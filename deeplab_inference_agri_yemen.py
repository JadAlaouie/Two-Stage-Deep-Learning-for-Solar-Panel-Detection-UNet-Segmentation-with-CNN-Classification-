"""
DeepLabV3+ Solar Panel Inference on City
Test trained model on large GeoTIFF images with visualization
SAVES INDIVIDUAL DETECTIONS AS PNG FILES
IMPROVED VERSION - Better detection with increased overlap and lower threshold
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import json
import warnings
warnings.filterwarnings('ignore')


class CityInference:
    """Run inference on large city GeoTIFF files"""
    
    def __init__(
        self,
        model_path: str,
        gpu_id: int = 1,
        tile_size: int = 512,
        overlap: int = 256,  # Increased from 64 to 256 (50% overlap)
        batch_size: int = 16,
        confidence_threshold: float = 0.35,  # Lowered from 0.5 to 0.35
        use_amp: bool = True,
        save_individual_detections: bool = True,
        detection_window_size: int = 512,
        use_tta: bool = False  # Test-Time Augmentation
    ):
        self.model_path = Path(model_path)
        self.gpu_id = gpu_id
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.use_amp = use_amp
        self.save_individual_detections = save_individual_detections
        self.detection_window_size = detection_window_size
        self.use_tta = use_tta
        
        self.device = torch.device(f'cuda:{gpu_id}')
        
        print("="*80)
        print("DEEPLABV3+ CITY INFERENCE - IMPROVED VERSION")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  GPU ID: {gpu_id}")
        print(f"  Device: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  Model: {model_path}")
        print(f"  Tile size: {tile_size}x{tile_size}")
        print(f"  Overlap: {overlap}px ({overlap/tile_size*100:.0f}% overlap)")
        print(f"  Batch size: {batch_size}")
        print(f"  Threshold: {confidence_threshold}")
        print(f"  Mixed precision: {'✅ Enabled' if use_amp else '❌ Disabled'}")
        print(f"  Test-Time Augmentation: {'✅ Enabled' if use_tta else '❌ Disabled'}")
        print(f"  Save individual detections: {'✅ Enabled' if save_individual_detections else '❌ Disabled'}")
        print(f"  Detection window size: {detection_window_size}x{detection_window_size}")
        
        # Load model
        self.model = self.load_model()
    
    def load_model(self):
        """Load trained DeepLabV3+ model"""
        print("\nLoading model...")
        
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None,
            encoder_depth=5,
            in_channels=3,
            classes=1,
            activation=None,
            upsampling=4
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best IoU: {checkpoint.get('best_val_iou', 'unknown')}")
            print(f"  Best Dice: {checkpoint.get('best_val_dice', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print("✓ Model loaded successfully")
        return model
    
    def preprocess_tile(self, tile):
        """Preprocess tile for inference"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        tile = tile.astype(np.float32) / 255.0
        tile = (tile - mean) / std
        tile = tile.transpose(2, 0, 1)
        
        return torch.from_numpy(tile).float()
    
    def extract_tiles(self, image):
        """Extract overlapping tiles from image"""
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap
        
        tiles = []
        positions = []
        
        for y in range(0, h - self.tile_size + 1, stride):
            for x in range(0, w - self.tile_size + 1, stride):
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append(self.preprocess_tile(tile))
                positions.append((y, x))
        
        # Handle remaining edges
        if w % stride != 0:
            x = w - self.tile_size
            for y in range(0, h - self.tile_size + 1, stride):
                tile = image[y:y+self.tile_size, x:w]
                if tile.shape[1] == self.tile_size:
                    tiles.append(self.preprocess_tile(tile))
                    positions.append((y, x))
        
        if h % stride != 0:
            y = h - self.tile_size
            for x in range(0, w - self.tile_size + 1, stride):
                tile = image[y:h, x:x+self.tile_size]
                if tile.shape[0] == self.tile_size:
                    tiles.append(self.preprocess_tile(tile))
                    positions.append((y, x))
        
        if h % stride != 0 and w % stride != 0:
            y = h - self.tile_size
            x = w - self.tile_size
            tile = image[y:h, x:w]
            if tile.shape[0] == self.tile_size and tile.shape[1] == self.tile_size:
                tiles.append(self.preprocess_tile(tile))
                positions.append((y, x))
        
        return tiles, positions
    
    def predict_with_tta(self, batch):
        """Predict with test-time augmentation (flips)"""
        predictions = []
        
        # Original
        with autocast(enabled=self.use_amp):
            pred = torch.sigmoid(self.model(batch))
            predictions.append(pred)
        
        # Horizontal flip
        batch_hflip = torch.flip(batch, dims=[3])
        with autocast(enabled=self.use_amp):
            pred = torch.sigmoid(self.model(batch_hflip))
            pred = torch.flip(pred, dims=[3])
            predictions.append(pred)
        
        # Vertical flip
        batch_vflip = torch.flip(batch, dims=[2])
        with autocast(enabled=self.use_amp):
            pred = torch.sigmoid(self.model(batch_vflip))
            pred = torch.flip(pred, dims=[2])
            predictions.append(pred)
        
        # Both flips
        batch_hvflip = torch.flip(batch, dims=[2, 3])
        with autocast(enabled=self.use_amp):
            pred = torch.sigmoid(self.model(batch_hvflip))
            pred = torch.flip(pred, dims=[2, 3])
            predictions.append(pred)
        
        # Average all predictions
        return torch.stack(predictions).mean(dim=0)
    
    def stitch_tiles(self, predictions, positions, image_shape):
        """Stitch tile predictions with weighted blending"""
        h, w = image_shape[:2]
        output = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        center_weight = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
        
        fade = self.overlap // 2
        for i in range(fade):
            weight = i / fade
            center_weight[i, :] *= weight
            center_weight[-i-1, :] *= weight
            center_weight[:, i] *= weight
            center_weight[:, -i-1] *= weight
        
        for pred, (y, x) in zip(predictions, positions):
            pred_h, pred_w = pred.shape
            output[y:y+pred_h, x:x+pred_w] += pred * center_weight[:pred_h, :pred_w]
            weight_map[y:y+pred_h, x:x+pred_w] += center_weight[:pred_h, :pred_w]
        
        output = np.divide(output, weight_map, where=weight_map > 0)
        
        return output
    
    def post_process_mask(self, binary_mask):
        """Clean up mask with morphological operations"""
        # Close small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        return cleaned
    
    def test_thresholds(self, prediction_map, image, output_dir, city_name):
        """Test multiple confidence thresholds for comparison"""
        print("\nGenerating threshold comparison...")
        
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()
        
        for idx, thresh in enumerate(thresholds):
            binary_mask = (prediction_map > thresh).astype(np.uint8)
            detected = binary_mask.sum()
            coverage = (detected / binary_mask.size) * 100
            
            # Create overlay
            viz_scale = 0.1
            viz_h = int(image.shape[0] * viz_scale)
            viz_w = int(image.shape[1] * viz_scale)
            
            image_small = cv2.resize(image, (viz_w, viz_h))
            mask_small = cv2.resize(binary_mask, (viz_w, viz_h))
            
            overlay = image_small.copy()
            overlay[mask_small > 0] = [255, 0, 0]
            blended = cv2.addWeighted(image_small, 0.6, overlay, 0.4, 0)
            
            axes[idx].imshow(blended)
            axes[idx].set_title(
                f'Threshold: {thresh}\n'
                f'Coverage: {coverage:.4f}%\n'
                f'Pixels: {detected:,}', 
                fontsize=10, fontweight='bold'
            )
            axes[idx].axis('off')
        
        axes[-1].axis('off')
        
        plt.suptitle(f'Confidence Threshold Comparison - {city_name.upper()}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / f"{city_name}_threshold_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Threshold comparison: {output_path}")
    
    def visualize_tile_boundaries(self, image, output_dir, city_name):
        """Visualize tile grid to debug boundary issues"""
        print("\nGenerating tile boundary visualization...")
        
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap
        
        viz_scale = 0.05  # 5% for very large images
        viz_img = cv2.resize(image, (int(w*viz_scale), int(h*viz_scale)))
        
        # Draw tile boundaries
        tile_count = 0
        for y in range(0, h - self.tile_size + 1, stride):
            for x in range(0, w - self.tile_size + 1, stride):
                y_viz = int(y * viz_scale)
                x_viz = int(x * viz_scale)
                h_viz = int(self.tile_size * viz_scale)
                w_viz = int(self.tile_size * viz_scale)
                
                cv2.rectangle(viz_img, (x_viz, y_viz), 
                             (x_viz + w_viz, y_viz + h_viz), 
                             (0, 255, 0), 1)
                tile_count += 1
        
        plt.figure(figsize=(20, 20))
        plt.imshow(viz_img)
        plt.title(f'Tile Grid Overlay - {city_name.upper()}\n'
                  f'{tile_count} tiles | Tile size: {self.tile_size}x{self.tile_size} | '
                  f'Overlap: {self.overlap}px | Stride: {stride}px', 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        
        output_path = output_dir / f"{city_name}_tile_grid.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Tile grid visualization: {output_path}")
    
    def save_individual_detections_from_mask(self, image, binary_mask, output_dir, city_name, 
                                            min_detection_pixels=50):
        """
        Save individual detection regions as separate PNG files
        
        Args:
            image: Original RGB image
            binary_mask: Binary detection mask
            output_dir: Output directory path
            city_name: Name of the city for file naming
            min_detection_pixels: Minimum number of detected pixels to save
        """
        print("\nSaving individual detections...")
        
        # Create detections folder
        detections_dir = output_dir / "detections"
        detections_dir.mkdir(parents=True, exist_ok=True)
        
        h, w = image.shape[:2]
        window_size = self.detection_window_size
        stride = window_size // 2  # 50% overlap
        
        saved_count = 0
        detection_metadata = []
        
        # Scan image with sliding window
        for y in tqdm(range(0, h - window_size + 1, stride), desc="Scanning for detections"):
            for x in range(0, w - window_size + 1, stride):
                # Extract window
                mask_window = binary_mask[y:y+window_size, x:x+window_size]
                
                # Check if window contains detections
                num_detected = mask_window.sum()
                
                if num_detected >= min_detection_pixels:
                    # Extract image window
                    img_window = image[y:y+window_size, x:x+window_size]
                    
                    # Create overlay
                    overlay = img_window.copy()
                    overlay[mask_window > 0] = [255, 0, 0]  # Red for detections
                    blended = cv2.addWeighted(img_window, 0.6, overlay, 0.4, 0)
                    
                    # Save original image
                    img_filename = f"{city_name}_det_{saved_count:05d}_orig.png"
                    img_path = detections_dir / img_filename
                    cv2.imwrite(str(img_path), cv2.cvtColor(img_window, cv2.COLOR_RGB2BGR))
                    
                    # Save overlay
                    overlay_filename = f"{city_name}_det_{saved_count:05d}_overlay.png"
                    overlay_path = detections_dir / overlay_filename
                    cv2.imwrite(str(overlay_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                    
                    # Save mask only
                    mask_filename = f"{city_name}_det_{saved_count:05d}_mask.png"
                    mask_path = detections_dir / mask_filename
                    cv2.imwrite(str(mask_path), mask_window * 255)
                    
                    # Store metadata
                    detection_metadata.append({
                        'id': saved_count,
                        'x': int(x),
                        'y': int(y),
                        'width': window_size,
                        'height': window_size,
                        'detected_pixels': int(num_detected),
                        'coverage_percent': float(num_detected / (window_size * window_size) * 100),
                        'files': {
                            'original': img_filename,
                            'overlay': overlay_filename,
                            'mask': mask_filename
                        }
                    })
                    
                    saved_count += 1
        
        # Handle edges (right and bottom)
        # Right edge
        if w % stride != 0:
            x = w - window_size
            for y in range(0, h - window_size + 1, stride):
                mask_window = binary_mask[y:y+window_size, x:x+window_size]
                num_detected = mask_window.sum()
                
                if num_detected >= min_detection_pixels:
                    img_window = image[y:y+window_size, x:x+window_size]
                    overlay = img_window.copy()
                    overlay[mask_window > 0] = [255, 0, 0]
                    blended = cv2.addWeighted(img_window, 0.6, overlay, 0.4, 0)
                    
                    img_filename = f"{city_name}_det_{saved_count:05d}_orig.png"
                    overlay_filename = f"{city_name}_det_{saved_count:05d}_overlay.png"
                    mask_filename = f"{city_name}_det_{saved_count:05d}_mask.png"
                    
                    cv2.imwrite(str(detections_dir / img_filename), cv2.cvtColor(img_window, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(detections_dir / overlay_filename), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(detections_dir / mask_filename), mask_window * 255)
                    
                    detection_metadata.append({
                        'id': saved_count,
                        'x': int(x),
                        'y': int(y),
                        'width': window_size,
                        'height': window_size,
                        'detected_pixels': int(num_detected),
                        'coverage_percent': float(num_detected / (window_size * window_size) * 100),
                        'files': {
                            'original': img_filename,
                            'overlay': overlay_filename,
                            'mask': mask_filename
                        }
                    })
                    saved_count += 1
        
        # Bottom edge
        if h % stride != 0:
            y = h - window_size
            for x in range(0, w - window_size + 1, stride):
                mask_window = binary_mask[y:y+window_size, x:x+window_size]
                num_detected = mask_window.sum()
                
                if num_detected >= min_detection_pixels:
                    img_window = image[y:y+window_size, x:x+window_size]
                    overlay = img_window.copy()
                    overlay[mask_window > 0] = [255, 0, 0]
                    blended = cv2.addWeighted(img_window, 0.6, overlay, 0.4, 0)
                    
                    img_filename = f"{city_name}_det_{saved_count:05d}_orig.png"
                    overlay_filename = f"{city_name}_det_{saved_count:05d}_overlay.png"
                    mask_filename = f"{city_name}_det_{saved_count:05d}_mask.png"
                    
                    cv2.imwrite(str(detections_dir / img_filename), cv2.cvtColor(img_window, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(detections_dir / overlay_filename), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(detections_dir / mask_filename), mask_window * 255)
                    
                    detection_metadata.append({
                        'id': saved_count,
                        'x': int(x),
                        'y': int(y),
                        'width': window_size,
                        'height': window_size,
                        'detected_pixels': int(num_detected),
                        'coverage_percent': float(num_detected / (window_size * window_size) * 100),
                        'files': {
                            'original': img_filename,
                            'overlay': overlay_filename,
                            'mask': mask_filename
                        }
                    })
                    saved_count += 1
        
        # Bottom-right corner
        if h % stride != 0 and w % stride != 0:
            y = h - window_size
            x = w - window_size
            mask_window = binary_mask[y:y+window_size, x:x+window_size]
            num_detected = mask_window.sum()
            
            if num_detected >= min_detection_pixels:
                img_window = image[y:y+window_size, x:x+window_size]
                overlay = img_window.copy()
                overlay[mask_window > 0] = [255, 0, 0]
                blended = cv2.addWeighted(img_window, 0.6, overlay, 0.4, 0)
                
                img_filename = f"{city_name}_det_{saved_count:05d}_orig.png"
                overlay_filename = f"{city_name}_det_{saved_count:05d}_overlay.png"
                mask_filename = f"{city_name}_det_{saved_count:05d}_mask.png"
                
                cv2.imwrite(str(detections_dir / img_filename), cv2.cvtColor(img_window, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(detections_dir / overlay_filename), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(detections_dir / mask_filename), mask_window * 255)
                
                detection_metadata.append({
                    'id': saved_count,
                    'x': int(x),
                    'y': int(y),
                    'width': window_size,
                    'height': window_size,
                    'detected_pixels': int(num_detected),
                    'coverage_percent': float(num_detected / (window_size * window_size) * 100),
                    'files': {
                        'original': img_filename,
                        'overlay': overlay_filename,
                        'mask': mask_filename
                    }
                })
                saved_count += 1
        
        # Save metadata
        metadata_file = detections_dir / f"{city_name}_detections_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'total_detections': saved_count,
                'window_size': window_size,
                'min_detection_pixels': min_detection_pixels,
                'detections': detection_metadata
            }, f, indent=2)
        
        print(f"\n  ✓ Saved {saved_count} detection windows")
        print(f"  ✓ Location: {detections_dir}")
        print(f"  ✓ Each detection has 3 files: *_orig.png, *_overlay.png, *_mask.png")
        print(f"  ✓ Metadata: {metadata_file.name}")
        
        return saved_count, detection_metadata
    
    def create_sample_visualizations(self, image, prediction_map, binary_mask, output_dir, city_name, num_samples=9):
        """Create grid of sample detections"""
        print("\nCreating sample visualizations...")
        
        h, w = image.shape[:2]
        
        # Find regions with detections
        detection_coords = np.column_stack(np.where(binary_mask > 0))
        
        if len(detection_coords) == 0:
            print("  ⚠ No detections found for visualization")
            return
        
        # Sample random detection locations
        sample_size = 2048  # Size of each sample window
        samples = []
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(detection_coords), min(num_samples * 10, len(detection_coords)), replace=False)
        
        for idx in sample_indices:
            y, x = detection_coords[idx]
            
            # Center window on detection
            y_start = max(0, y - sample_size // 2)
            x_start = max(0, x - sample_size // 2)
            y_end = min(h, y_start + sample_size)
            x_end = min(w, x_start + sample_size)
            
            # Adjust if at edge
            if y_end - y_start < sample_size:
                y_start = max(0, y_end - sample_size)
            if x_end - x_start < sample_size:
                x_start = max(0, x_end - sample_size)
            
            img_crop = image[y_start:y_end, x_start:x_end]
            mask_crop = binary_mask[y_start:y_end, x_start:x_end]
            prob_crop = prediction_map[y_start:y_end, x_start:x_end]
            
            # Only keep samples with sufficient detections
            if mask_crop.sum() > 1000:  # At least 1000 detected pixels
                samples.append({
                    'image': img_crop,
                    'mask': mask_crop,
                    'prob': prob_crop,
                    'coords': (y_start, x_start)
                })
            
            if len(samples) >= num_samples:
                break
        
        if len(samples) == 0:
            print("  ⚠ No suitable samples found")
            return
        
        # Create grid visualization
        rows = int(np.ceil(np.sqrt(len(samples))))
        cols = int(np.ceil(len(samples) / rows))
        
        fig = plt.figure(figsize=(cols * 6, rows * 6))
        
        for idx, sample in enumerate(samples):
            # Original image
            ax1 = plt.subplot(rows, cols * 3, idx * 3 + 1)
            ax1.imshow(sample['image'])
            ax1.set_title(f'Sample {idx+1}\nCoords: ({sample["coords"][1]}, {sample["coords"][0]})', fontsize=10)
            ax1.axis('off')
            
            # Overlay
            ax2 = plt.subplot(rows, cols * 3, idx * 3 + 2)
            overlay = sample['image'].copy()
            overlay[sample['mask'] > 0] = [255, 0, 0]
            blended = cv2.addWeighted(sample['image'], 0.6, overlay, 0.4, 0)
            ax2.imshow(blended)
            ax2.set_title(f'Detection Overlay\nPixels: {sample["mask"].sum():,}', fontsize=10)
            ax2.axis('off')
            
            # Probability heatmap
            ax3 = plt.subplot(rows, cols * 3, idx * 3 + 3)
            im = ax3.imshow(sample['prob'], cmap='hot', vmin=0, vmax=1)
            ax3.set_title(f'Confidence Map\nMax: {sample["prob"].max():.3f}', fontsize=10)
            ax3.axis('off')
            plt.colorbar(im, ax=ax3, fraction=0.046)
        
        plt.tight_layout()
        samples_output = output_dir / f"{city_name}_samples.png"
        plt.savefig(samples_output, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Sample visualizations: {samples_output}")
        print(f"    Generated {len(samples)} samples")
    
    def create_statistics_plot(self, stats, output_dir, city_name):
        """Create visual statistics summary"""
        print("\nCreating statistics visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Solar Panel Detection Statistics - {city_name.upper()}', fontsize=20, fontweight='bold')
        
        # 1. Coverage pie chart
        ax1 = axes[0, 0]
        coverage = stats['coverage_percent']
        sizes = [coverage, 100 - coverage]
        colors = ['#ff6b6b', '#f0f0f0']
        explode = (0.1, 0)
        
        ax1.pie(sizes, explode=explode, labels=['Solar Panels', 'Other'], colors=colors,
                autopct='%1.4f%%', startangle=90, textprops={'fontsize': 12})
        ax1.set_title(f'Coverage: {coverage:.4f}%', fontsize=14, fontweight='bold')
        
        # 2. Detection metrics
        ax2 = axes[0, 1]
        ax2.axis('off')
        metrics_text = f"""
DETECTION METRICS

Total Pixels: {stats['total_pixels']:,}
Detected Pixels: {stats['detected_pixels']:,}
Coverage: {stats['coverage_percent']:.4f}%

AREA COVERAGE

Detected Area: {stats['detected_area_m2']:,.2f} m²
Detected Area: {stats['detected_area_km2']:.4f} km²

INDIVIDUAL DETECTIONS

Detection Windows Saved: {stats.get('num_detection_windows', 'N/A')}

MODEL PERFORMANCE

Confidence Threshold: {stats['confidence_threshold']}
Test-Time Augmentation: {stats.get('use_tta', False)}
Processing Time: {stats['processing_time_seconds']/60:.2f} minutes

IMAGE DETAILS

Image Size: {stats['image_size']}
Total Area: {stats['total_pixels'] * 0.09 / 1_000_000:.2f} km²
"""
        ax2.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 3. Area bar chart
        ax3 = axes[1, 0]
        areas = [stats['detected_area_m2'], stats['total_pixels'] * 0.09]  # 0.3m x 0.3m = 0.09 m²/pixel
        labels = ['Solar Panels', 'Total Area']
        colors_bar = ['#ff6b6b', '#4ecdc4']
        
        bars = ax3.bar(labels, areas, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Area (m²)', fontsize=12, fontweight='bold')
        ax3.set_title('Area Comparison', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f} m²',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Detection density
        ax4 = axes[1, 1]
        density = (stats['detected_pixels'] / stats['total_pixels']) * 100
        x = ['Image']
        y = [density]
        
        bars = ax4.bar(x, y, color='#ff6b6b', alpha=0.7, edgecolor='black', linewidth=2, width=0.5)
        ax4.set_ylabel('Solar Panel Density (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Detection Density', fontsize=14, fontweight='bold')
        ax4.set_ylim([0, max(0.1, density * 1.5)])
        ax4.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        stats_plot = output_dir / f"{city_name}_statistics.png"
        plt.savefig(stats_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Statistics plot: {stats_plot}")
    
    def predict_city(self, image_path: str, output_dir: str):
        """Run prediction on entire city image"""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Processing: {image_path.name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Read image
        print("\nReading image...")
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]).transpose(1, 2, 0)
            profile = src.profile
            transform = src.transform
            crs = src.crs
            
            print(f"  Image size: {image.shape[1]}x{image.shape[0]} pixels")
            print(f"  Bands: {image.shape[2]}")
            print(f"  CRS: {crs}")
            print(f"  Size: {image.nbytes / 1024**3:.2f} GB")
        
        # Extract tiles
        print("\nExtracting tiles...")
        tiles, positions = self.extract_tiles(image)
        print(f"  Total tiles: {len(tiles):,}")
        print(f"  Estimated batches: {len(tiles) // self.batch_size + 1}")
        
        # Generate tile grid visualization
        self.visualize_tile_boundaries(image, output_dir, image_path.stem)
        
        # Run inference
        print("\nRunning inference...")
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(tiles), self.batch_size), desc="Processing"):
                batch = tiles[i:i+self.batch_size]
                batch = torch.stack(batch).to(self.device)
                
                if self.use_tta:
                    # Use test-time augmentation
                    preds = self.predict_with_tta(batch)
                else:
                    # Standard inference
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(batch)
                        preds = torch.sigmoid(outputs)
                
                preds = preds.cpu().numpy()
                
                for pred in preds:
                    predictions.append(pred.squeeze())
        
        # Stitch tiles
        print("\nStitching tiles...")
        prediction_map = self.stitch_tiles(predictions, positions, image.shape)
        
        # Generate threshold comparison
        self.test_thresholds(prediction_map, image, output_dir, image_path.stem)
        
        # Apply threshold
        binary_mask = (prediction_map > self.confidence_threshold).astype(np.uint8)
        
        # Apply post-processing
        print("\nApplying post-processing...")
        binary_mask = self.post_process_mask(binary_mask)
        
        # Calculate statistics
        print("\nCalculating statistics...")
        total_pixels = binary_mask.size
        detected_pixels = binary_mask.sum()
        coverage_percent = (detected_pixels / total_pixels) * 100
        
        pixel_area_m2 = 0.3 * 0.3
        detected_area_m2 = detected_pixels * pixel_area_m2
        detected_area_km2 = detected_area_m2 / 1_000_000
        
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {image_path.stem.upper()}")
        print(f"{'='*80}")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Detected pixels: {detected_pixels:,}")
        print(f"Coverage: {coverage_percent:.4f}%")
        print(f"Detected area: {detected_area_m2:,.2f} m² ({detected_area_km2:.4f} km²)")
        
        # Save individual detections
        num_detection_windows = 0
        if self.save_individual_detections:
            num_detection_windows, _ = self.save_individual_detections_from_mask(
                image, binary_mask, output_dir, image_path.stem, 
                min_detection_pixels=50
            )
        
        # Save outputs
        print("\nSaving outputs...")
        
        # 1. Probability map (with BIGTIFF support)
        prob_profile = profile.copy()
        prob_profile.update(
            count=1,
            dtype=rasterio.float32,
            compress='lzw',
            BIGTIFF='YES'
        )
        
        prob_output = output_dir / f"{image_path.stem}_probability.tif"
        with rasterio.open(prob_output, 'w', **prob_profile) as dst:
            dst.write(prediction_map.astype(np.float32), 1)
        print(f"  ✓ Probability map: {prob_output}")
        
        # 2. Binary mask (with BIGTIFF support)
        mask_profile = profile.copy()
        mask_profile.update(
            count=1,
            dtype=rasterio.uint8,
            compress='lzw',
            BIGTIFF='YES'
        )
        
        mask_output = output_dir / f"{image_path.stem}_mask.tif"
        with rasterio.open(mask_output, 'w', **mask_profile) as dst:
            dst.write(binary_mask, 1)
        print(f"  ✓ Binary mask: {mask_output}")
        
        # 3. Full overlay (downsampled)
        print("\nCreating full image visualization...")
        viz_scale = 0.1  # 10% for very large images
        viz_h = int(image.shape[0] * viz_scale)
        viz_w = int(image.shape[1] * viz_scale)
        
        image_small = cv2.resize(image, (viz_w, viz_h))
        mask_small = cv2.resize(binary_mask, (viz_w, viz_h))
        
        overlay = image_small.copy()
        overlay[mask_small > 0] = [255, 0, 0]
        
        alpha = 0.4
        blended = cv2.addWeighted(image_small, 1-alpha, overlay, alpha, 0)
        
        viz_output = output_dir / f"{image_path.stem}_full_overlay.png"
        cv2.imwrite(str(viz_output), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        print(f"  ✓ Full overlay: {viz_output}")
        
        # 4. Sample visualizations
        self.create_sample_visualizations(
            image, prediction_map, binary_mask, 
            output_dir, image_path.stem, num_samples=9
        )
        
        # 5. Statistics
        stats = {
            'city': image_path.stem,
            'image_size': f"{image.shape[1]}x{image.shape[0]}",
            'total_pixels': int(total_pixels),
            'detected_pixels': int(detected_pixels),
            'coverage_percent': float(coverage_percent),
            'detected_area_m2': float(detected_area_m2),
            'detected_area_km2': float(detected_area_km2),
            'confidence_threshold': float(self.confidence_threshold),
            'use_tta': self.use_tta,
            'overlap': self.overlap,
            'processing_time_seconds': float(time.time() - start_time),
            'num_detection_windows': int(num_detection_windows)
        }
        
        stats_output = output_dir / f"{image_path.stem}_stats.json"
        with open(stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Statistics JSON: {stats_output}")
        
        # 6. Statistics visualization
        self.create_statistics_plot(stats, output_dir, image_path.stem)
        
        # Processing time
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Processing complete!")
        print(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
        print(f"{'='*80}\n")
        
        return {
            'probability_map': prob_output,
            'binary_mask': mask_output,
            'full_overlay': viz_output,
            'stats': stats,
            'num_detections': num_detection_windows
        }


def main():
    """Run inference on AGRI City YEMEN"""
    
    MODEL_PATH = r"D:\Images\models\deeplabv3plus\deeplabv3plus_best.pth"
    IMAGE_PATH = r"D:\Images\agri_yemen.tif"
    OUTPUT_DIR = r"D:\Images\results_yemen\agri_yemen.tif"
    GPU_ID = 1
    
    print("\n" + "="*80)
    print("DEEPLABV3+ CITY INFERENCE - YEMEN - IMPROVED VERSION")
    print("="*80)
    print(f"\nModel: DeepLabV3+ with ResNet101")
    print(f"Best IoU: 0.7534 | Best Dice: 0.8521")
    print(f"GPU: RTX A5000 #2 (GPU ID: 1)")
    print(f"\nIMPROVEMENTS:")
    print(f"  ✅ Increased tile overlap: 64px → 256px (50% overlap)")
    print(f"  ✅ Lowered confidence threshold: 0.5 → 0.35")
    print(f"  ✅ Added morphological post-processing")
    print(f"  ✅ Added threshold comparison visualization")
    print(f"  ✅ Added tile grid visualization")
    print(f"  ✅ Optional: Test-Time Augmentation (TTA)")
    
    inference = CityInference(
        model_path=MODEL_PATH,
        gpu_id=GPU_ID,
        tile_size=512,
        overlap=256,  # INCREASED from 64 to 256 (50% overlap)
        batch_size=16,  # REDUCED from 32 due to more tiles
        confidence_threshold=0.35,  # LOWERED from 0.5 to 0.35
        use_amp=True,
        save_individual_detections=True,
        detection_window_size=512,
        use_tta=False  # Set to True for even better results (4x slower)
    )
    
    results = inference.predict_city(IMAGE_PATH, OUTPUT_DIR)
    
    print("\n✓ Inference complete!")
    print(f"✓ Results saved to: {OUTPUT_DIR}")
    print(f"\nOutputs:")
    print(f"  - Probability map (GeoTIFF): {results['probability_map'].name}")
    print(f"  - Binary mask (GeoTIFF): {results['binary_mask'].name}")
    print(f"  - Full overlay (PNG): {results['full_overlay'].name}")
    print(f"  - Sample detections (PNG): {Path(IMAGE_PATH).stem}_samples.png")
    print(f"  - Statistics plot (PNG): {Path(IMAGE_PATH).stem}_statistics.png")
    print(f"  - Statistics (JSON): {Path(IMAGE_PATH).stem}_stats.json")
    print(f"  - Threshold comparison (PNG): {Path(IMAGE_PATH).stem}_threshold_comparison.png")
    print(f"  - Tile grid (PNG): {Path(IMAGE_PATH).stem}_tile_grid.png")
    print(f"\n  🎯 Individual detections saved: {results['num_detections']} windows")
    print(f"     Location: {OUTPUT_DIR}/detections/")
    print(f"     Each detection has 3 files:")
    print(f"       - *_orig.png (original image)")
    print(f"       - *_overlay.png (with red detection overlay)")
    print(f"       - *_mask.png (binary mask only)")
    print(f"\n💡 TIP: Check the threshold_comparison.png to see if you need to adjust")
    print(f"   the confidence threshold further!")


if __name__ == "__main__":
    main()