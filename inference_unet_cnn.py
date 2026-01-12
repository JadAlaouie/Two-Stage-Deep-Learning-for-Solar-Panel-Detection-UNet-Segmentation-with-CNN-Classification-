"""
Two-Stage Solar Panel Detection: UNet++ → CNN Classifier
Tests CNN classifier effectiveness on real-world Yemen city image
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import rasterio
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import json
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


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


class TwoStageDetector:
    """Two-stage solar panel detector: UNet++ segmentation + CNN classification"""

    def __init__(
        self,
        unet_model_path: str,
        cnn_model_path: str,
        output_dir: str,
        gpu_id: int = 0,
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 16,
        unet_threshold: float = 0.5,
        cnn_threshold: float = 0.7,
        min_region_area: int = 50,
        cnn_input_size: int = 224
    ):
        self.unet_model_path = Path(unet_model_path)
        self.cnn_model_path = Path(cnn_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.unet_threshold = unet_threshold
        self.cnn_threshold = cnn_threshold
        self.min_region_area = min_region_area
        self.cnn_input_size = cnn_input_size

        # Set GPU
        self.device = torch.device(f'cuda:{gpu_id}')

        print("="*80)
        print("TWO-STAGE SOLAR PANEL DETECTION (UNet++ + CNN)")
        print("="*80)
        print(f"\nGPU Configuration:")
        print(f"  GPU ID: {gpu_id}")
        print(f"  Device: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")

        print(f"\nStage 1 - UNet++ Segmentation:")
        print(f"  Model: {self.unet_model_path.name}")
        print(f"  Tile size: {tile_size}x{tile_size}")
        print(f"  Overlap: {overlap}px")
        print(f"  Threshold: {unet_threshold}")

        print(f"\nStage 2 - CNN Classification:")
        print(f"  Model: {self.cnn_model_path.name}")
        print(f"  Input size: {cnn_input_size}x{cnn_input_size}")
        print(f"  Threshold: {cnn_threshold}")
        print(f"  Min region area: {min_region_area}px")

        # Load models
        self.unet_model = None
        self.cnn_model = None
        self.unet_transform = None
        self.cnn_transform = None
        self._load_models()

    def _load_models(self):
        """Load both UNet++ and CNN models"""
        print("\n" + "="*80)
        print("LOADING MODELS")
        print("="*80)

        # 1. Load UNet++ model
        print(f"\n[Stage 1] Loading UNet++ model...")
        self.unet_model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b7',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

        checkpoint = torch.load(self.unet_model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.unet_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best IoU: {checkpoint.get('best_val_iou', 'N/A'):.4f}")
        else:
            self.unet_model.load_state_dict(checkpoint)

        self.unet_model = self.unet_model.to(self.device)
        self.unet_model.eval()

        self.unet_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

        # 2. Load CNN classifier
        print(f"\n[Stage 2] Loading CNN classifier...")
        self.cnn_model = EfficientSolarClassifier(pretrained=False, dropout=0.3)

        checkpoint = torch.load(self.cnn_model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best accuracy: {checkpoint.get('best_val_acc', 'N/A'):.4f}")
        else:
            self.cnn_model.load_state_dict(checkpoint)

        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()

        self.cnn_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

        print(f"\nBoth models loaded and ready")

    def _extract_tiles(self, image, tile_size, overlap):
        """Extract overlapping tiles from image"""
        h, w = image.shape[:2]
        stride = tile_size - overlap

        tiles = []
        positions = []

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((y, x))

        # Handle right edge
        if w % stride != 0:
            for y in range(0, h - tile_size + 1, stride):
                x = w - tile_size
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((y, x))

        # Handle bottom edge
        if h % stride != 0:
            for x in range(0, w - tile_size + 1, stride):
                y = h - tile_size
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((y, x))

        # Handle bottom-right corner
        if h % stride != 0 and w % stride != 0:
            y = h - tile_size
            x = w - tile_size
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))

        return tiles, positions

    def _predict_unet_tiles(self, tiles):
        """Predict on batch of tiles with UNet++"""
        # Transform tiles
        transformed = []
        for tile in tiles:
            augmented = self.unet_transform(image=tile)
            transformed.append(augmented['image'])

        # Stack into batch
        batch = torch.stack(transformed).to(self.device)

        # Predict
        with torch.no_grad():
            with autocast():
                outputs = self.unet_model(batch)
                preds = torch.sigmoid(outputs)

        return preds.cpu().numpy()

    def _merge_tiles(self, predictions, positions, output_shape, tile_size, overlap):
        """Merge overlapping tile predictions with averaging"""
        h, w = output_shape
        prediction_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        for pred, (y, x) in zip(predictions, positions):
            pred_tile = pred[0]  # Remove channel dimension

            # Add prediction to map
            prediction_map[y:y+tile_size, x:x+tile_size] += pred_tile
            count_map[y:y+tile_size, x:x+tile_size] += 1

        # Average overlapping predictions
        prediction_map = np.divide(
            prediction_map,
            count_map,
            out=np.zeros_like(prediction_map),
            where=count_map != 0
        )

        return prediction_map

    def _extract_regions(self, binary_mask, original_image):
        """Extract individual regions from binary mask"""
        # Label connected components
        labeled_mask, num_regions = ndimage.label(binary_mask)

        regions = []
        region_info = []

        for region_id in range(1, num_regions + 1):
            # Get region mask
            region_mask = (labeled_mask == region_id).astype(np.uint8)

            # Calculate area
            area = region_mask.sum()

            # Skip small regions
            if area < self.min_region_area:
                continue

            # Find bounding box
            coords = np.column_stack(np.where(region_mask > 0))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Add some padding
            pad = 10
            y_min = max(0, y_min - pad)
            x_min = max(0, x_min - pad)
            y_max = min(original_image.shape[0], y_max + pad)
            x_max = min(original_image.shape[1], x_max + pad)

            # Extract region from original image
            region_img = original_image[y_min:y_max, x_min:x_max]

            # Resize to CNN input size
            region_resized = cv2.resize(region_img, (self.cnn_input_size, self.cnn_input_size))

            regions.append(region_resized)
            region_info.append({
                'id': region_id,
                'bbox': (y_min, x_min, y_max, x_max),
                'area': area,
                'region_mask': region_mask
            })

        return regions, region_info

    def _classify_regions(self, regions):
        """Classify regions using CNN"""
        if len(regions) == 0:
            return []

        # Transform all regions
        transformed = []
        for region in regions:
            augmented = self.cnn_transform(image=region)
            transformed.append(augmented['image'])

        # Predict in batches
        predictions = []
        for i in range(0, len(transformed), self.batch_size):
            batch = torch.stack(transformed[i:i+self.batch_size]).to(self.device)

            with torch.no_grad():
                with autocast():
                    outputs = self.cnn_model(batch)
                    preds = outputs.squeeze().cpu().numpy()

            # Handle single prediction case
            if len(preds.shape) == 0:
                preds = np.array([preds])

            predictions.extend(preds)

        return predictions

    def process_image(self, image_path: str):
        """
        Process image with two-stage pipeline

        Args:
            image_path: Path to input GeoTIFF

        Returns:
            Dictionary with results and statistics
        """
        image_path = Path(image_path)
        print("\n" + "="*80)
        print(f"PROCESSING: {image_path.name}")
        print("="*80)

        # Read GeoTIFF
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]).transpose(1, 2, 0)
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'width': src.width,
                'height': src.height,
                'bounds': src.bounds
            }

            print(f"\nImage info:")
            print(f"  Size: {src.width} x {src.height}")
            print(f"  CRS: {src.crs}")

        # Normalize image to 0-255 range
        if image.max() > 255:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        h, w = image.shape[:2]

        # ====================================================================
        # STAGE 1: UNet++ Segmentation
        # ====================================================================
        print("\n" + "="*80)
        print("STAGE 1: UNet++ SEGMENTATION")
        print("="*80)

        # Extract tiles
        print(f"\nExtracting tiles...")
        tiles, positions = self._extract_tiles(image, self.tile_size, self.overlap)
        print(f"  Total tiles: {len(tiles)}")

        # Predict in batches
        print(f"\nRunning UNet++ inference...")
        all_predictions = []

        pbar = tqdm(range(0, len(tiles), self.batch_size), desc="Processing tiles")
        for i in pbar:
            batch_tiles = tiles[i:i+self.batch_size]
            batch_preds = self._predict_unet_tiles(batch_tiles)
            all_predictions.extend(batch_preds)

        # Merge predictions
        print(f"\nMerging predictions...")
        unet_probability_map = self._merge_tiles(
            all_predictions,
            positions,
            (h, w),
            self.tile_size,
            self.overlap
        )

        # Apply threshold
        unet_binary_mask = (unet_probability_map > self.unet_threshold).astype(np.uint8)

        # Calculate Stage 1 statistics
        unet_pixels = unet_binary_mask.sum()
        print(f"\nStage 1 Complete:")
        print(f"  Detected pixels: {unet_pixels:,}")
        print(f"  Coverage: {(unet_pixels / (h * w)) * 100:.2f}%")

        # ====================================================================
        # STAGE 2: CNN Classification
        # ====================================================================
        print("\n" + "="*80)
        print("STAGE 2: CNN CLASSIFICATION")
        print("="*80)

        # Extract regions from UNet++ mask
        print(f"\nExtracting regions from UNet++ detections...")
        regions, region_info = self._extract_regions(unet_binary_mask, image)
        print(f"  Total regions (area >= {self.min_region_area}px): {len(regions)}")

        if len(regions) == 0:
            print("\nWarning: No regions found to classify!")
            final_mask = np.zeros_like(unet_binary_mask)
            cnn_predictions = []
        else:
            # Classify regions
            print(f"\nClassifying regions with CNN...")
            cnn_predictions = self._classify_regions(regions)

            # Filter regions by CNN confidence
            print(f"\nFiltering by CNN confidence (threshold: {self.cnn_threshold})...")
            final_mask = np.zeros_like(unet_binary_mask)

            passed_regions = 0
            rejected_regions = 0

            for info, confidence in zip(region_info, cnn_predictions):
                if confidence >= self.cnn_threshold:
                    # Keep this region
                    final_mask += info['region_mask']
                    passed_regions += 1
                else:
                    rejected_regions += 1

            print(f"  Passed: {passed_regions} regions")
            print(f"  Rejected: {rejected_regions} regions")
            print(f"  Rejection rate: {(rejected_regions / len(regions)) * 100:.1f}%")

        # Calculate final statistics
        final_pixels = final_mask.sum()

        print(f"\nStage 2 Complete:")
        print(f"  Final detected pixels: {final_pixels:,}")
        print(f"  Coverage: {(final_pixels / (h * w)) * 100:.2f}%")
        print(f"  False positive reduction: {((unet_pixels - final_pixels) / unet_pixels * 100) if unet_pixels > 0 else 0:.1f}%")

        # ====================================================================
        # Save Results
        # ====================================================================
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)

        output_name = image_path.stem

        # 1. Save UNet++ outputs
        unet_mask_path = self.output_dir / f"{output_name}_unet_mask.tif"
        self._save_geotiff(unet_binary_mask, unet_mask_path, metadata)
        print(f"\nSaved Saved UNet++ mask: {unet_mask_path}")

        unet_prob_path = self.output_dir / f"{output_name}_unet_probability.tif"
        self._save_geotiff(unet_probability_map, unet_prob_path, metadata)
        print(f"Saved Saved UNet++ probability: {unet_prob_path}")

        # 2. Save final CNN-filtered outputs
        final_mask_path = self.output_dir / f"{output_name}_final_mask.tif"
        self._save_geotiff(final_mask, final_mask_path, metadata)
        print(f"Saved Saved final mask: {final_mask_path}")

        # 3. Save visualizations
        print(f"\nGenerating visualizations...")

        # Stage 1 overlay
        unet_overlay = self._create_overlay(image, unet_binary_mask, color=(255, 0, 0), alpha=0.5)
        unet_overlay_path = self.output_dir / f"{output_name}_stage1_unet.png"
        cv2.imwrite(str(unet_overlay_path), cv2.cvtColor(unet_overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved Saved Stage 1 overlay: {unet_overlay_path}")

        # Final overlay
        final_overlay = self._create_overlay(image, final_mask, color=(0, 255, 0), alpha=0.5)
        final_overlay_path = self.output_dir / f"{output_name}_stage2_final.png"
        cv2.imwrite(str(final_overlay_path), cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved Saved Stage 2 overlay: {final_overlay_path}")

        # Side-by-side comparison
        comparison = self._create_comparison(image, unet_binary_mask, final_mask)
        comparison_path = self.output_dir / f"{output_name}_comparison.png"
        cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"Saved Saved comparison: {comparison_path}")

        # 4. Save statistics
        stats = {
            'image': str(image_path),
            'image_size': {'width': w, 'height': h},
            'total_pixels': int(h * w),
            'stage1_unet': {
                'detected_pixels': int(unet_pixels),
                'coverage_percent': float((unet_pixels / (h * w)) * 100),
                'threshold': float(self.unet_threshold)
            },
            'stage2_cnn': {
                'total_regions': len(regions),
                'passed_regions': int(np.sum([c >= self.cnn_threshold for c in cnn_predictions])) if len(cnn_predictions) > 0 else 0,
                'rejected_regions': int(np.sum([c < self.cnn_threshold for c in cnn_predictions])) if len(cnn_predictions) > 0 else 0,
                'rejection_rate_percent': float((np.sum([c < self.cnn_threshold for c in cnn_predictions]) / len(regions) * 100)) if len(regions) > 0 else 0,
                'threshold': float(self.cnn_threshold)
            },
            'final_output': {
                'detected_pixels': int(final_pixels),
                'coverage_percent': float((final_pixels / (h * w)) * 100),
                'false_positive_reduction_percent': float(((unet_pixels - final_pixels) / unet_pixels * 100)) if unet_pixels > 0 else 0
            }
        }

        stats_path = self.output_dir / f"{output_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved Saved statistics: {stats_path}")

        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)

        return {
            'unet_mask': unet_binary_mask,
            'unet_probability': unet_probability_map,
            'final_mask': final_mask,
            'stats': stats,
            'metadata': metadata
        }

    def _save_geotiff(self, data, output_path, metadata):
        """Save array as georeferenced GeoTIFF"""
        # Ensure data is 2D
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=metadata['height'],
            width=metadata['width'],
            count=data.shape[0],
            dtype=data.dtype,
            crs=metadata['crs'],
            transform=metadata['transform'],
            compress='lzw'
        ) as dst:
            dst.write(data)

    def _create_overlay(self, image, mask, color=(255, 0, 0), alpha=0.5):
        """Create overlay of mask on image"""
        overlay = image.copy()

        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 0] = mask * color[0]
        colored_mask[:, :, 1] = mask * color[1]
        colored_mask[:, :, 2] = mask * color[2]

        # Blend
        overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)

        return overlay

    def _create_comparison(self, image, unet_mask, final_mask):
        """Create side-by-side comparison"""
        h, w = image.shape[:2]

        # Resize for display if too large
        max_size = 2000
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            unet_mask = cv2.resize(unet_mask.astype(np.uint8), (new_w, new_h))
            final_mask = cv2.resize(final_mask.astype(np.uint8), (new_w, new_h))
            h, w = new_h, new_w

        # Create 3-panel comparison
        comparison = np.zeros((h, w*3, 3), dtype=np.uint8)

        # Original image
        comparison[:, :w] = image

        # Stage 1: UNet++ (Red)
        unet_overlay = self._create_overlay(image, unet_mask, color=(255, 0, 0), alpha=0.5)
        comparison[:, w:2*w] = unet_overlay

        # Stage 2: Final (Green)
        final_overlay = self._create_overlay(image, final_mask, color=(0, 255, 0), alpha=0.5)
        comparison[:, 2*w:3*w] = final_overlay

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Stage 1: UNet++', (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Stage 2: UNet++ + CNN', (2*w + 10, 30), font, 1, (255, 255, 255), 2)

        return comparison


def main():
    """Run two-stage inference on Yemen city image"""

    print("\n" + "="*80)
    print("TWO-STAGE SOLAR PANEL DETECTION PIPELINE")
    print("UNet++ Segmentation + CNN Classification")
    print("="*80)

    # Configuration
    UNET_MODEL_PATH = r"D:\Images\models\unetplusplus\unetplusplus_epoch90.pth"
    CNN_MODEL_PATH = r"D:\Images\models\cnn_classifier\best_cnn_classifier.pth"
    INPUT_IMAGE = r"D:\Images\huraida.tif"  # Yemen city image
    OUTPUT_DIR = r"D:\Images\PV-Panel-Detection\unet_cnn"

    GPU_ID = 0

    # UNet++ parameters
    TILE_SIZE = 512
    OVERLAP = 64
    BATCH_SIZE = 16
    UNET_THRESHOLD = 0.5

    # CNN parameters
    CNN_THRESHOLD = 0.7
    CNN_INPUT_SIZE = 224
    MIN_REGION_AREA = 50

    print(f"\nConfiguration:")
    print(f"  Input image: {INPUT_IMAGE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  GPU: {GPU_ID}")
    print(f"  UNet++ threshold: {UNET_THRESHOLD}")
    print(f"  CNN threshold: {CNN_THRESHOLD}")

    # Create detector
    detector = TwoStageDetector(
        unet_model_path=UNET_MODEL_PATH,
        cnn_model_path=CNN_MODEL_PATH,
        output_dir=OUTPUT_DIR,
        gpu_id=GPU_ID,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        batch_size=BATCH_SIZE,
        unet_threshold=UNET_THRESHOLD,
        cnn_threshold=CNN_THRESHOLD,
        min_region_area=MIN_REGION_AREA,
        cnn_input_size=CNN_INPUT_SIZE
    )

    # Process image
    results = detector.process_image(INPUT_IMAGE)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    stats = results['stats']
    print(f"\nStage 1 (UNet++):")
    print(f"  Detected pixels: {stats['stage1_unet']['detected_pixels']:,}")
    print(f"  Coverage: {stats['stage1_unet']['coverage_percent']:.2f}%")

    print(f"\nStage 2 (CNN):")
    print(f"  Total regions: {stats['stage2_cnn']['total_regions']}")
    print(f"  Passed: {stats['stage2_cnn']['passed_regions']}")
    print(f"  Rejected: {stats['stage2_cnn']['rejected_regions']}")
    print(f"  Rejection rate: {stats['stage2_cnn']['rejection_rate_percent']:.1f}%")

    print(f"\nFinal Output:")
    print(f"  Detected pixels: {stats['final_output']['detected_pixels']:,}")
    print(f"  Coverage: {stats['final_output']['coverage_percent']:.2f}%")
    print(f"  False positive reduction: {stats['final_output']['false_positive_reduction_percent']:.1f}%")

    print(f"\nSaved All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
