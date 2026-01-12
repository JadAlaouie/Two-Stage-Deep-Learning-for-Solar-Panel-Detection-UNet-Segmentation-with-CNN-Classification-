"""
Solar Panel Segmentation - U-Net++ with EfficientNet-B7
Optimized for RTX A5000 (24GB VRAM)
GPU ID: 0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import time
import os
warnings.filterwarnings('ignore')


class SolarPanelDataset(Dataset):
    """Solar panel segmentation dataset"""
    
    def __init__(self, image_files, mask_dir, transform=None):
        self.image_files = image_files
        self.mask_dir = Path(mask_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)


class UNetPlusPlusTrainer:
    """
    U-Net++ Trainer with EfficientNet-B7 backbone
    Optimized for RTX A5000 (24GB VRAM)
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        gpu_id: int = 0,  # RTX A5000 #1
        
        # Model configuration
        encoder: str = "efficientnet-b7",
        encoder_weights: str = "imagenet",
        
        # Training configuration - OPTIMIZED FOR A5000
        batch_size: int = 48,  # Increased for better GPU utilization              # Large batch for 24GB VRAM
        num_epochs: int = 150,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-5,
        
        # Advanced optimization
        use_amp: bool = True,               # Mixed precision (FP16)
        gradient_accumulation: int = 1,     # No need with batch_size=32
        
        # Data split
        validation_split: float = 0.15,
        
        # Early stopping
        early_stopping_patience: int = 20,
        
        # Learning rate schedule
        lr_scheduler: str = "cosine",       # cosine, plateau, or none
        warmup_epochs: int = 5,
        
        # Test-time augmentation
        use_tta: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.gpu_id = gpu_id
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_amp = use_amp
        self.gradient_accumulation = gradient_accumulation
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.use_tta = use_tta
        
        # Set GPU
        self.device = torch.device(f'cuda:{gpu_id}')
        
        print("="*80)
        print("U-NET++ SOLAR PANEL SEGMENTATION TRAINER")
        print("="*80)
        print(f"\nGPU Configuration:")
        print(f"  GPU ID: {gpu_id}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print(f"\nModel Configuration:")
        print(f"  Architecture: U-Net++ (UnetPlusPlus)")
        print(f"  Encoder: {encoder}")
        print(f"  Pre-trained: {encoder_weights}")
        
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Mixed precision: {'✅ Enabled' if use_amp else '❌ Disabled'}")
        print(f"  Validation split: {validation_split*100:.0f}%")
        print(f"  LR scheduler: {lr_scheduler}")
        print(f"  TTA: {'✅ Enabled' if use_tta else '❌ Disabled'}")
        
        # Initialize
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if use_amp else None
        self.train_loader = None
        self.val_loader = None
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'learning_rate': []
        }
        
        self.best_val_iou = 0.0
        self.best_val_dice = 0.0
    
    def prepare_data(self):
        """Prepare training and validation data"""
        print("\n" + "="*80)
        print("PREPARING DATA")
        print("="*80)
        
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"
        
        # Get all images
        all_images = sorted(list(images_dir.glob("*.png")))
        print(f"\nTotal samples: {len(all_images):,}")
        
        # Train/val split
        train_files, val_files = train_test_split(
            all_images,
            test_size=self.validation_split,
            random_state=42,
            shuffle=True
        )
        
        print(f"Training samples: {len(train_files):,}")
        print(f"Validation samples: {len(val_files):,}")
        
        # Advanced augmentation for training
        train_transform = A.Compose([
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7
            ),
            
            # Photometric transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=25,
                    val_shift_limit=25,
                    p=1
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1
                ),
            ], p=0.8),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.MotionBlur(blur_limit=5, p=1),
            ], p=0.3),
            
            # Weather effects (simulate different conditions)
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=1),
            ], p=0.2),
            
            # Normalization (ImageNet stats)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
        
        # Validation transform (no augmentation)
        val_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
        
        # Create datasets
        train_dataset = SolarPanelDataset(train_files, masks_dir, transform=train_transform)
        val_dataset = SolarPanelDataset(val_files, masks_dir, transform=val_transform)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,  # More parallel data loading
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4  # Preload more batches
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,  # More parallel data loading
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4  # Preload more batches
        )
        
        print(f"\n✓ Data preparation complete")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
    
    def build_model(self):
        """Build U-Net++ model"""
        print("\n" + "="*80)
        print("BUILDING MODEL")
        print("="*80)
        
        # Create U-Net++ with EfficientNet-B7
        self.model = smp.UnetPlusPlus(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        
        self.model = self.model.to(self.device)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel: U-Net++ with {self.encoder}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB")
        
        # Optimizer - AdamW (better than Adam)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        if self.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-7
            )
            print(f"LR Scheduler: Cosine Annealing with Warm Restarts")
        elif self.lr_scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=7,
                verbose=True,
                min_lr=1e-7
            )
            print(f"LR Scheduler: Reduce on Plateau")
        else:
            self.scheduler = None
            print(f"LR Scheduler: None")
        
        print(f"\n✓ Model built successfully")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        # Losses
        dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        focal_loss_fn = smp.losses.FocalLoss(mode='binary')
        
        def combined_loss(pred, target):
            return (
                dice_loss_fn(pred, target) +
                bce_loss_fn(pred, target) +
                0.5 * focal_loss_fn(pred, target)
            )
        
        # Metrics - using manual IoU calculation (smp.utils removed in newer versions)
        def calculate_iou(pred, target, threshold=0.5):
            pred = (pred > threshold).float()
            target = (target > threshold).float()
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            return (intersection / (union + 1e-8)).item()
        
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_precision = 0.0
        running_recall = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Mixed precision training
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = combined_loss(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                iou = calculate_iou(preds, masks)
                
                # Dice coefficient
                intersection = (preds * masks).sum()
                dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-8)
                
                # Precision and recall
                tp = ((preds > 0.5) & (masks > 0.5)).sum().float()
                fp = ((preds > 0.5) & (masks < 0.5)).sum().float()
                fn = ((preds < 0.5) & (masks > 0.5)).sum().float()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
            
            running_loss += loss.item()
            running_iou += iou
            running_dice += dice.item()
            running_precision += precision.item()
            running_recall += recall.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice.item():.4f}'
            })
        
        # Average metrics
        n_batches = len(self.train_loader)
        return {
            'loss': running_loss / n_batches,
            'iou': running_iou / n_batches,
            'dice': running_dice / n_batches,
            'precision': running_precision / n_batches,
            'recall': running_recall / n_batches
        }
    
    def validate_epoch(self, epoch):
        """Validate one epoch"""
        self.model.eval()
        
        dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        focal_loss_fn = smp.losses.FocalLoss(mode='binary')
        
        def combined_loss(pred, target):
            return (
                dice_loss_fn(pred, target) +
                bce_loss_fn(pred, target) +
                0.5 * focal_loss_fn(pred, target)
            )
        
        # Metrics
        def calculate_iou(pred, target, threshold=0.5):
            pred = (pred > threshold).float()
            target = (target > threshold).float()
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            return (intersection / (union + 1e-8)).item()
        
        iou_metric = calculate_iou
        
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_precision = 0.0
        running_recall = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
        
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = combined_loss(outputs, masks)
                
                preds = torch.sigmoid(outputs)
                iou = calculate_iou(preds, masks)
                
                # Dice
                intersection = (preds * masks).sum()
                dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-8)
                
                # Precision and recall
                tp = ((preds > 0.5) & (masks > 0.5)).sum().float()
                fp = ((preds > 0.5) & (masks < 0.5)).sum().float()
                fn = ((preds < 0.5) & (masks > 0.5)).sum().float()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                
                running_loss += loss.item()
                running_iou += iou
                running_dice += dice.item()
                running_precision += precision.item()
                running_recall += recall.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}',
                    'dice': f'{dice.item():.4f}'
                })
        
        n_batches = len(self.val_loader)
        return {
            'loss': running_loss / n_batches,
            'iou': running_iou / n_batches,
            'dice': running_dice / n_batches,
            'precision': running_precision / n_batches,
            'recall': running_recall / n_batches
        }
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80 + "\n")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # LR scheduling
            if self.scheduler:
                if self.lr_scheduler == "plateau":
                    self.scheduler.step(val_metrics['iou'])
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary ({epoch_time:.1f}s)")
            print(f"{'='*80}")
            print(f"Train - Loss: {train_metrics['loss']:.4f} | IoU: {train_metrics['iou']:.4f} | "
                  f"Dice: {train_metrics['dice']:.4f} | Precision: {train_metrics['precision']:.4f} | "
                  f"Recall: {train_metrics['recall']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f} | IoU: {val_metrics['iou']:.4f} | "
                  f"Dice: {val_metrics['dice']:.4f} | Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model (based on IoU)
            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['iou']
                self.best_val_dice = val_metrics['dice']
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_iou': self.best_val_iou,
                    'best_val_dice': self.best_val_dice,
                    'history': self.history
                }
                
                torch.save(checkpoint, self.output_dir / 'unetplusplus_best.pth')
                print(f"✓ Best model saved! IoU: {self.best_val_iou:.4f}, Dice: {self.best_val_dice:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{self.early_stopping_patience})")
            
            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"{'='*80}")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }
                torch.save(checkpoint, self.output_dir / f'unetplusplus_epoch{epoch+1}.pth')
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation IoU: {self.best_val_iou:.4f}")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        
        # Save final model
        torch.save(self.model.state_dict(), self.output_dir / 'unetplusplus_final.pth')
        
        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IoU
        axes[0, 1].plot(self.history['train_iou'], label='Train')
        axes[0, 1].plot(self.history['val_iou'], label='Val')
        axes[0, 1].set_title('IoU Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Dice
        axes[1, 0].plot(self.history['train_dice'], label='Train')
        axes[1, 0].plot(self.history['val_dice'], label='Val')
        axes[1, 0].set_title('Dice Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to {self.output_dir / 'training_curves.png'}")


def main():
    """Run U-Net++ training"""
    
    # Configuration
    DATA_DIR = r"D:\Images\training_data"
    OUTPUT_DIR = r"D:\Images\models\unetplusplus"
    GPU_ID = 0  # First GPU (RTX A5000 #1)
    
    print("\n" + "="*80)
    print("U-NET++ TRAINING SCRIPT")
    print("Model: U-Net++ with EfficientNet-B7")
    print("GPU: RTX A5000 #1 (GPU ID: 0)")
    print("="*80)
    
    # Create trainer
    trainer = UNetPlusPlusTrainer(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        gpu_id=GPU_ID,
        
        # Model
        encoder="efficientnet-b7",
        encoder_weights="imagenet",
        
        # Training (optimized for A5000)
        batch_size=32,
        num_epochs=150,
        learning_rate=0.0001,
        weight_decay=1e-5,
        
        # Optimization
        use_amp=True,
        gradient_accumulation=1,
        
        # Validation
        validation_split=0.15,
        early_stopping_patience=20,
        
        # LR schedule
        lr_scheduler="cosine",
        warmup_epochs=5,
        
        # TTA
        use_tta=True,
    )
    
    # Prepare data
    trainer.prepare_data()
    
    # Build model
    trainer.build_model()
    
    # Train
    trainer.train()
    
    print("\n✓ U-Net++ training complete!")
    print(f"✓ Best model saved to: {OUTPUT_DIR}/unetplusplus_best.pth")


if __name__ == "__main__":
    main()