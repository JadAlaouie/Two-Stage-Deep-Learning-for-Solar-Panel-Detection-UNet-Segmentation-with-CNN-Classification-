"""
CNN Binary Classifier Training for Solar Panel Verification
Stage 2 of the validation pipeline: Classifies extracted regions as panel/not-panel
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')


class SolarPanelDataset(Dataset):
    """Dataset for binary classification: solar panel vs not-panel"""

    def __init__(self, image_paths, labels, transform=None, input_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to input size
        image = cv2.resize(image, (self.input_size, self.input_size))

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels[idx]

        return image, label


class EfficientSolarClassifier(nn.Module):
    """Lightweight CNN classifier for solar panel verification"""

    def __init__(self, pretrained=True, dropout=0.3):
        super(EfficientSolarClassifier, self).__init__()

        # Use EfficientNet-B0 as backbone (lightweight but powerful)
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class CNNTrainer:
    """Trainer for CNN binary classifier"""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        gpu_id: int = 1,
        input_size: int = 224,
        batch_size: int = 32,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 10
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        print("="*80)
        print("CNN BINARY CLASSIFIER TRAINING")
        print("="*80)
        print(f"\nGPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"Input size: {input_size}x{input_size}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")

    def prepare_data(self, positive_ratio=0.7):
        """
        Prepare dataset from training_data folder
        Positive samples: chips with panels
        Negative samples: chips without panels (background)
        """
        print("\n" + "="*80)
        print("PREPARING DATA")
        print("="*80)

        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"

        all_images = list(images_dir.glob("*.png"))
        print(f"\nFound {len(all_images):,} total images")

        # Categorize based on label files
        positive_images = []
        negative_images = []

        print("Categorizing images...")
        for img_path in tqdm(all_images):
            label_path = labels_dir / f"{img_path.stem}.txt"

            if label_path.exists():
                with open(label_path, 'r') as f:
                    labels = f.read().strip()
                    if len(labels) > 0:  # Has labels = positive
                        positive_images.append(img_path)
                    else:  # Empty labels = negative
                        negative_images.append(img_path)
            else:
                negative_images.append(img_path)

        print(f"\n✓ Positive samples (with panels): {len(positive_images):,}")
        print(f"✓ Negative samples (no panels): {len(negative_images):,}")

        # Balance dataset
        n_positive = len(positive_images)
        n_negative = int(n_positive * (1 - positive_ratio) / positive_ratio)

        if n_negative > len(negative_images):
            n_negative = len(negative_images)
            print(f"\n⚠ Using all {n_negative} negative samples")
        else:
            # Randomly sample negatives
            import random
            negative_images = random.sample(negative_images, n_negative)
            print(f"\n✓ Sampled {n_negative} negative samples for balance")

        # Combine and create labels
        all_image_paths = positive_images + negative_images
        all_labels = [1] * len(positive_images) + [0] * len(negative_images)

        print(f"\nFinal dataset:")
        print(f"  Total: {len(all_image_paths):,}")
        print(f"  Positive: {sum(all_labels):,} ({sum(all_labels)/len(all_labels)*100:.1f}%)")
        print(f"  Negative: {len(all_labels) - sum(all_labels):,} ({(1-sum(all_labels)/len(all_labels))*100:.1f}%)")

        # Split into train/val/test
        train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
            all_image_paths, all_labels, test_size=0.3, random_state=42, stratify=all_labels
        )

        val_imgs, test_imgs, val_labels, test_labels = train_test_split(
            temp_imgs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )

        print(f"\nSplit:")
        print(f"  Train: {len(train_imgs):,}")
        print(f"  Val: {len(val_imgs):,}")
        print(f"  Test: {len(test_imgs):,}")

        return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)

    def get_transforms(self, augment=True):
        """Get data transforms"""
        if augment:
            train_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            train_transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        return train_transform, val_transform

    def train(self):
        """Train the classifier"""
        # Prepare data
        (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = self.prepare_data()

        # Create datasets
        train_transform, val_transform = self.get_transforms(augment=True)

        train_dataset = SolarPanelDataset(train_imgs, train_labels, train_transform, self.input_size)
        val_dataset = SolarPanelDataset(val_imgs, val_labels, val_transform, self.input_size)
        test_dataset = SolarPanelDataset(test_imgs, test_labels, val_transform, self.input_size)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        print("\n" + "="*80)
        print("BUILDING MODEL")
        print("="*80)

        # Create model
        model = EfficientSolarClassifier(pretrained=True, dropout=0.3)
        model = model.to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                         patience=5)

        print("✓ Model created")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Training loop
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)

        best_val_acc = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 80)

            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc="Training")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.4f}'
                })

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validating"):
                    images = images.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history
                }

                torch.save(checkpoint, self.output_dir / 'best_cnn_classifier.pth')
                print(f"  ✓ Saved best model (Val Acc: {best_val_acc:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{self.early_stopping_patience}")

                if patience_counter >= self.early_stopping_patience:
                    print("\n⚠ Early stopping triggered!")
                    break

        # Test on test set
        print("\n" + "="*80)
        print("TESTING ON TEST SET")
        print("="*80)

        # Load best model
        checkpoint = torch.load(self.output_dir / 'best_cnn_classifier.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                labels_gpu = labels.float().unsqueeze(1).to(self.device)

                outputs = model(images)
                predictions = (outputs > 0.5).float()

                test_correct += (predictions == labels_gpu).sum().item()
                test_total += labels.size(0)

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.numpy())

        test_acc = test_correct / test_total

        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )

        cm = confusion_matrix(all_labels, all_predictions)

        print(f"\n✓ Test Results:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")

        # Save final results
        results = {
            'test_accuracy': test_acc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'confusion_matrix': cm.tolist(),
            'best_val_accuracy': best_val_acc,
            'history': history
        }

        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Training complete!")
        print(f"✓ Best model saved to: {self.output_dir / 'best_cnn_classifier.pth'}")
        print(f"✓ Results saved to: {self.output_dir / 'training_results.json'}")


def main():
    """Train CNN classifier"""

    # Configuration
    DATA_DIR = r"D:\Images\training_data"
    OUTPUT_DIR = r"D:\Images\models\cnn_classifier"

    GPU_ID = 0  # Use GPU 0 for CNN training
    INPUT_SIZE = 224  # Standard for EfficientNet
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4

    trainer = CNNTrainer(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        gpu_id=GPU_ID,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=10
    )

    trainer.train()


if __name__ == "__main__":
    main()
