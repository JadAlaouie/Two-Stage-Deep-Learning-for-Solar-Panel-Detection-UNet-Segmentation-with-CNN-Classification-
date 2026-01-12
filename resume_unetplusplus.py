"""
Resume U-Net++ Training from Latest Checkpoint
Automatically finds and loads the most recent checkpoint
GPU ID: 0 (RTX A5000 #1)
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
import re
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
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = self.mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)


def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint file"""
    model_path = Path(model_dir)
    
    # Look for all checkpoint files
    checkpoints = []
    
    # Check for best checkpoint
    best_ckpt = model_path / "unetplusplus_best.pth"
    if best_ckpt.exists():
        checkpoints.append(('best', best_ckpt, best_ckpt.stat().st_mtime))
    
    # Check for epoch checkpoints
    for ckpt_file in model_path.glob("unetplusplus_epoch*.pth"):
        # Extract epoch number
        match = re.search(r'epoch(\d+)', ckpt_file.name)
        if match:
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, ckpt_file, ckpt_file.stat().st_mtime))
    
    if not checkpoints:
        return None
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda x: x[2], reverse=True)
    
    latest = checkpoints[0]
    return latest[1]


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load checkpoint and return starting epoch and history"""
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_iou = checkpoint.get('best_val_iou', 0.0)
    best_val_dice = checkpoint.get('best_val_dice', 0.0)
    history = checkpoint.get('history', {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'learning_rate': []
    })
    
    print(f"✓ Loaded checkpoint from epoch {start_epoch - 1}")
    print(f"  Best IoU: {best_val_iou:.4f}")
    print(f"  Best Dice: {best_val_dice:.4f}")
    
    return start_epoch, best_val_iou, best_val_dice, history


def prepare_data(data_dir, batch_size, validation_split=0.15):
    """Prepare data loaders"""
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)
    
    images_dir = Path(data_dir) / "images"
    masks_dir = Path(data_dir) / "masks"
    
    all_images = sorted(list(images_dir.glob("*.png")))
    print(f"\nTotal samples: {len(all_images):,}")
    
    # IMPORTANT: Use same random_state=42 to get same train/val split
    train_files, val_files = train_test_split(
        all_images,
        test_size=validation_split,
        random_state=42,
        shuffle=True
    )
    
    print(f"Training samples: {len(train_files):,}")
    print(f"Validation samples: {len(val_files):,}")
    
    # Training augmentation
    train_transform = A.Compose([
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
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=1),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
            A.MotionBlur(blur_limit=5, p=1),
        ], p=0.3),
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=1),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    train_dataset = SolarPanelDataset(train_files, masks_dir, transform=train_transform)
    val_dataset = SolarPanelDataset(val_files, masks_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    
    print(f"\n✓ Data preparation complete")
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scaler, device, epoch, total_epochs):
    """Train one epoch"""
    model.train()
    
    dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    focal_loss_fn = smp.losses.FocalLoss(mode='binary')
    
    def combined_loss(pred, target):
        return (
            dice_loss_fn(pred, target) +
            bce_loss_fn(pred, target) +
            0.5 * focal_loss_fn(pred, target)
        )
    
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
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast(enabled=True):
            outputs = model(images)
            loss = combined_loss(outputs, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            iou = calculate_iou(preds, masks)
            
            intersection = (preds * masks).sum()
            dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-8)
            
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
    
    n = len(train_loader)
    return {
        'loss': running_loss / n,
        'iou': running_iou / n,
        'dice': running_dice / n,
        'precision': running_precision / n,
        'recall': running_recall / n
    }


def validate_epoch(model, val_loader, device, epoch, total_epochs):
    """Validate one epoch"""
    model.eval()
    
    dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    focal_loss_fn = smp.losses.FocalLoss(mode='binary')
    
    def combined_loss(pred, target):
        return (
            dice_loss_fn(pred, target) +
            bce_loss_fn(pred, target) +
            0.5 * focal_loss_fn(pred, target)
        )
    
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
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} [Val]")
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            with autocast(enabled=True):
                outputs = model(images)
                loss = combined_loss(outputs, masks)
            
            preds = torch.sigmoid(outputs)
            iou = calculate_iou(preds, masks)
            
            intersection = (preds * masks).sum()
            dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-8)
            
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
    
    n = len(val_loader)
    return {
        'loss': running_loss / n,
        'iou': running_iou / n,
        'dice': running_dice / n,
        'precision': running_precision / n,
        'recall': running_recall / n
    }


def main():
    """Resume U-Net++ training from latest checkpoint"""
    
    # Configuration
    DATA_DIR = r"D:\Images\training_data"
    MODEL_DIR = r"D:\Images\models\unetplusplus"
    GPU_ID = 0
    
    # Training params
    BATCH_SIZE = 48
    TOTAL_EPOCHS = 150  # Total epochs (will continue from checkpoint epoch)
    LEARNING_RATE = 0.0001
    EARLY_STOPPING_PATIENCE = 20
    
    print("\n" + "="*80)
    print("RESUME U-NET++ TRAINING (AUTO-LATEST)")
    print("="*80)
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint(MODEL_DIR)
    
    if checkpoint_path is None:
        print("❌ No checkpoint found! Please train from scratch first.")
        return
    
    print(f"✓ Found latest checkpoint: {checkpoint_path.name}")
    print(f"GPU: {GPU_ID} - {torch.cuda.get_device_name(GPU_ID)}")
    
    device = torch.device(f'cuda:{GPU_ID}')
    
    # Build model
    print("\n🔨 Building model...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5
    )
    
    # Load checkpoint
    start_epoch, best_val_iou, best_val_dice, history = load_checkpoint(
        checkpoint_path, model, optimizer, device
    )
    
    # Prepare data
    train_loader, val_loader = prepare_data(DATA_DIR, BATCH_SIZE)
    
    # Scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-7
    )
    
    # Move scheduler to correct epoch
    for _ in range(start_epoch):
        scheduler.step()
    
    # Scaler for mixed precision
    scaler = GradScaler()
    
    print("\n" + "="*80)
    print("RESUMING TRAINING")
    print("="*80)
    print(f"Starting from epoch: {start_epoch}")
    print(f"Total epochs: {TOTAL_EPOCHS}")
    print(f"Remaining: {TOTAL_EPOCHS - start_epoch} epochs\n")
    
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, epoch + 1, TOTAL_EPOCHS)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, epoch + 1, TOTAL_EPOCHS)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # LR step
        scheduler.step()
        
        # Summary
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} Summary ({epoch_time:.1f}s)")
        print(f"{'='*80}")
        print(f"Train - Loss: {train_metrics['loss']:.4f} | IoU: {train_metrics['iou']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f} | Precision: {train_metrics['precision']:.4f} | "
              f"Recall: {train_metrics['recall']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | IoU: {val_metrics['iou']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f} | Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_val_dice = val_metrics['dice']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'best_val_dice': best_val_dice,
                'history': history
            }
            
            torch.save(checkpoint, Path(MODEL_DIR) / 'unetplusplus_best.pth')
            print(f"✓ Best model saved! IoU: {best_val_iou:.4f}, Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n{'='*80}")
            print(f"Early stopping at epoch {epoch+1}")
            print(f"{'='*80}")
            break
        
        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'best_val_dice': best_val_dice,
                'history': history
            }
            torch.save(checkpoint, Path(MODEL_DIR) / f'unetplusplus_epoch{epoch+1}.pth')
    
    # Complete
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    
    # Save final
    torch.save(model.state_dict(), Path(MODEL_DIR) / 'unetplusplus_final.pth')
    
    # Save history
    with open(Path(MODEL_DIR) / 'training_history_resumed.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training resumed and completed!")


if __name__ == "__main__":
    main()