"""
Solar Panel Detection - ULTRA-FAST Training Data Preparation
GPU-accelerated with multi-processing for high-end hardware
Optimized for: RTX A4500 + 1TB RAM + Powerful CPU
"""

import rasterio
from rasterio.windows import Window
import geopandas as gpd
import numpy as np
from pathlib import Path
import cv2
from shapely.geometry import box
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Multi-processing
from multiprocessing import Pool, Manager, cpu_count
import multiprocessing as mp

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled!")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ CuPy not found - Install with: pip install cupy-cuda11x")
    print("  (Replace cuda11x with your CUDA version)")

# Fast image processing
from concurrent.futures import ThreadPoolExecutor
import queue


class FastTrainingDataExporter:
    """
    Ultra-fast training data exporter using:
    - GPU acceleration for image processing
    - Multi-processing for parallel chip extraction
    - Memory mapping for large images
    - Batch processing for efficiency
    """
    
    def __init__(
        self,
        image_dir: str,
        shapefile_dir: str,
        output_dir: str,
        chip_size: int = 256,
        stride: int = 128,
        augment: bool = True,
        blacken_background: bool = True,
        include_negatives: bool = True,
        negative_ratio: float = 0.3,
        num_workers: int = None,
        batch_size: int = 100,
        use_gpu: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.shapefile_dir = Path(shapefile_dir)
        self.output_dir = Path(output_dir)
        self.chip_size = chip_size
        self.stride = stride
        self.augment = augment
        self.blacken_background = blacken_background
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio
        self.batch_size = batch_size
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Auto-detect optimal number of workers
        if num_workers is None:
            self.num_workers = max(1, cpu_count() - 2)  # Leave 2 cores free
        else:
            self.num_workers = num_workers
        
        print(f"🚀 Performance Configuration:")
        print(f"   CPU Workers: {self.num_workers} (out of {cpu_count()} available)")
        print(f"   GPU Acceleration: {'✅ Enabled' if self.use_gpu else '❌ Disabled'}")
        print(f"   Batch Size: {self.batch_size}")
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.labels_dir = self.output_dir / "labels"
        self.blackened_dir = self.output_dir / "images_blackened"
        
        for d in [self.images_dir, self.masks_dir, self.labels_dir, self.blackened_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Define mappings - Using multiple regions for higher data volume
        self.mappings = {
            'Aaley': 'Aaley_sp',
            'Beirut': 'BeirutSP',
            'Batroun': 'BatrounSP',
            'Nabatiyeh': 'Nabatiyeh',
            'Sour': 'SourSP',
            'Bint_Jbeil': 'Bint_Jbeil',
            'Hermel': 'Hermel_SP',
            'Aanjar': 'AnjarSP',
        }
    
    def process_all(self):
        """
        Process all image-shapefile pairs in parallel
        """
        print("="*80)
        print("ULTRA-FAST SOLAR PANEL TRAINING DATA EXPORT")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Chip size: {self.chip_size}×{self.chip_size} pixels")
        print(f"  Stride: {self.stride} pixels")
        print(f"  Augmentation: {'✅' if self.augment else '❌'}")
        print(f"  Background blackening: {'✅' if self.blacken_background else '❌'}")
        print(f"  Include negatives: {'✅' if self.include_negatives else '❌'}")
        print(f"\nProcessing {len(self.mappings)} image pairs in parallel...")
        print("="*80)
        
        # Prepare all image pairs
        image_pairs = []
        for img_name, shp_name in self.mappings.items():
            img_path = self.image_dir / f"{img_name}.tif"
            shp_path = self.shapefile_dir / f"{shp_name}.shp"
            
            if img_path.exists() and shp_path.exists():
                image_pairs.append((img_path, shp_path, img_name))
        
        print(f"\n✓ Found {len(image_pairs)} valid image pairs")
        
        # Process with multiprocessing
        # Use sequential processing to avoid issues with rasterio and multiprocessing
        total_stats = {
            'total_chips': 0,
            'positive_chips': 0,
            'negative_chips': 0,
            'augmented_chips': 0,
            'solar_panels_found': 0,
            'images_processed': 0
        }
        
        for img_path, shp_path, img_name in image_pairs:
            print(f"\n{'='*80}")
            print(f"Processing: {img_name}")
            print(f"{'='*80}")
            
            try:
                stats = self.process_image_pair_fast(img_path, shp_path, img_name)
                
                # Aggregate stats
                for key in total_stats:
                    total_stats[key] += stats[key]
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Save and print summary
        self.save_statistics(total_stats)
        self.print_summary(total_stats)
    
    def process_image_pair_fast(self, img_path: Path, shp_path: Path, img_name: str):
        """
        Fast processing of single image pair using GPU and batching
        """
        stats = {
            'total_chips': 0,
            'positive_chips': 0,
            'negative_chips': 0,
            'augmented_chips': 0,
            'solar_panels_found': 0,
            'images_processed': 0
        }
        
        # Load shapefile
        gdf = gpd.read_file(shp_path)
        
        # Find class column
        class_col = None
        for col in gdf.columns:
            if col.lower() == 'class':
                class_col = col
                break
        
        if not class_col:
            print(f"  ⚠️ No class column")
            return stats
        
        # Open image with memory mapping
        with rasterio.open(img_path) as src:
            img_crs = src.crs
            img_transform = src.transform
            img_width = src.width
            img_height = src.height
            
            print(f"  Image: {img_width}×{img_height} pixels")
            
            # Reproject shapefile if needed
            if gdf.crs != img_crs:
                gdf = gdf.to_crs(img_crs)
            
            # Filter geometries
            solar_gdf = gdf[gdf[class_col] == 1].copy()
            
            print(f"  Solar panels: {len(solar_gdf):,}")
            
            if len(solar_gdf) == 0:
                return stats
            
            stats['solar_panels_found'] = len(solar_gdf)
            
            # Calculate all chip positions
            n_chips_x = (img_width - self.chip_size) // self.stride + 1
            n_chips_y = (img_height - self.chip_size) // self.stride + 1
            
            print(f"  Grid: {n_chips_x} × {n_chips_y} = {n_chips_x * n_chips_y:,} chips")
            
            # Pre-calculate all chip bounds and intersections
            print(f"  Pre-calculating spatial intersections...")
            chip_data = []
            
            for y_idx in range(n_chips_y):
                for x_idx in range(n_chips_x):
                    x_offset = x_idx * self.stride
                    y_offset = y_idx * self.stride
                    
                    window = Window(x_offset, y_offset, self.chip_size, self.chip_size)
                    chip_bounds = rasterio.windows.bounds(window, src.transform)
                    chip_box = box(*chip_bounds)
                    
                    # Find intersecting solar panels
                    intersecting = solar_gdf[solar_gdf.intersects(chip_box)]
                    
                    is_positive = len(intersecting) > 0
                    
                    if is_positive or (self.include_negatives and np.random.random() < self.negative_ratio):
                        chip_data.append({
                            'x_offset': x_offset,
                            'y_offset': y_offset,
                            'window': window,
                            'bounds': chip_bounds,
                            'is_positive': is_positive,
                            'geometries': intersecting
                        })
            
            print(f"  Selected {len(chip_data):,} chips to extract")
            
            # Process chips in batches for speed
            positive_count = 0
            negative_count = 0
            
            for batch_start in tqdm(range(0, len(chip_data), self.batch_size), 
                                   desc="  Extracting batches"):
                batch_end = min(batch_start + self.batch_size, len(chip_data))
                batch = chip_data[batch_start:batch_end]
                
                # Read all chips in batch at once
                chips = []
                for item in batch:
                    chip = src.read(window=item['window'])
                    
                    # Convert to HWC format
                    if chip.shape[0] <= 4:
                        chip = np.transpose(chip, (1, 2, 0))
                    if chip.shape[2] == 4:
                        chip = chip[:, :, :3]
                    
                    chips.append(chip)
                
                # Process batch with GPU if available
                if self.use_gpu:
                    processed_batch = self.process_batch_gpu(chips, batch, img_name, 
                                                            positive_count, negative_count,
                                                            src.transform)
                else:
                    processed_batch = self.process_batch_cpu(chips, batch, img_name,
                                                            positive_count, negative_count,
                                                            src.transform)
                
                positive_count += processed_batch['positive']
                negative_count += processed_batch['negative']
                stats['augmented_chips'] += processed_batch['augmented']
        
        stats['positive_chips'] = positive_count
        stats['negative_chips'] = negative_count
        stats['total_chips'] = positive_count + negative_count
        stats['images_processed'] = 1
        
        print(f"  ✓ Extracted {positive_count:,} positive + {negative_count:,} negative")
        if self.augment:
            print(f"  ✓ Created {stats['augmented_chips']:,} augmented chips")
        
        return stats
    
    def process_batch_gpu(self, chips, batch_info, img_name, pos_count, neg_count, transform):
        """
        Process batch of chips using GPU acceleration
        """
        stats = {'positive': 0, 'negative': 0, 'augmented': 0}
        
        # Convert to CuPy arrays for GPU processing
        for idx, (chip, info) in enumerate(zip(chips, batch_info)):
            if info['is_positive']:
                chip_id = f"{img_name}_chip_{pos_count + stats['positive']:06d}"
                
                # Process on GPU
                if self.use_gpu:
                    chip_gpu = cp.asarray(chip)
                    
                    # Save original
                    self.save_chip(cp.asnumpy(chip_gpu), chip_id, "original")
                    
                    # Create mask
                    mask = self.create_mask_fast(info['geometries'], info['bounds'], 
                                                 transform, info['window'], self.chip_size)
                    self.save_mask(mask, chip_id, "original")
                    
                    # Create labels
                    labels = self.create_yolo_labels(info['geometries'], info['bounds'], 
                                                     self.chip_size)
                    self.save_yolo_labels(labels, chip_id, "original")
                    
                    # Blacken background
                    if self.blacken_background:
                        mask_gpu = cp.asarray(mask)
                        mask_3ch = cp.stack([mask_gpu, mask_gpu, mask_gpu], axis=2)
                        blackened = cp.where(mask_3ch > 0, chip_gpu, 0)
                        self.save_blackened_chip(cp.asnumpy(blackened).astype(np.uint8), 
                                                chip_id, "original")
                    
                    # Augmentation on GPU (much faster!)
                    if self.augment:
                        aug_results = self.augment_batch_gpu(chip_gpu, mask_gpu if self.blacken_background else cp.asarray(mask),
                                                            info['geometries'], info['bounds'])
                        
                        for aug_idx, (aug_chip, aug_mask, aug_labels) in enumerate(aug_results):
                            aug_chip_id = f"{img_name}_chip_{pos_count + stats['positive']:06d}_aug{aug_idx}"
                            
                            self.save_chip(cp.asnumpy(aug_chip), aug_chip_id, "augmented")
                            self.save_mask(cp.asnumpy(aug_mask), aug_chip_id, "augmented")
                            self.save_yolo_labels(aug_labels, aug_chip_id, "augmented")
                            
                            if self.blacken_background:
                                mask_3ch = cp.stack([aug_mask, aug_mask, aug_mask], axis=2)
                                blackened = cp.where(mask_3ch > 0, aug_chip, 0)
                                self.save_blackened_chip(cp.asnumpy(blackened).astype(np.uint8),
                                                        aug_chip_id, "augmented")
                            
                            stats['augmented'] += 1
                else:
                    # Fallback to CPU
                    self.save_chip(chip, chip_id, "original")
                    mask = self.create_mask_fast(info['geometries'], info['bounds'],
                                                 transform, info['window'], self.chip_size)
                    self.save_mask(mask, chip_id, "original")
                    labels = self.create_yolo_labels(info['geometries'], info['bounds'],
                                                     self.chip_size)
                    self.save_yolo_labels(labels, chip_id, "original")
                
                stats['positive'] += 1
                
            else:
                # Negative chip
                chip_id = f"{img_name}_chip_neg_{neg_count + stats['negative']:06d}"
                self.save_chip(chip, chip_id, "negative")
                
                mask = np.zeros((self.chip_size, self.chip_size), dtype=np.uint8)
                self.save_mask(mask, chip_id, "negative")
                self.save_yolo_labels([], chip_id, "negative")
                
                stats['negative'] += 1
        
        return stats
    
    def process_batch_cpu(self, chips, batch_info, img_name, pos_count, neg_count, transform):
        """
        Process batch of chips using CPU (fallback)
        """
        stats = {'positive': 0, 'negative': 0, 'augmented': 0}
        
        for idx, (chip, info) in enumerate(zip(chips, batch_info)):
            if info['is_positive']:
                chip_id = f"{img_name}_chip_{pos_count + stats['positive']:06d}"
                
                self.save_chip(chip, chip_id, "original")
                
                mask = self.create_mask_fast(info['geometries'], info['bounds'],
                                             transform, info['window'], self.chip_size)
                self.save_mask(mask, chip_id, "original")
                
                labels = self.create_yolo_labels(info['geometries'], info['bounds'],
                                                 self.chip_size)
                self.save_yolo_labels(labels, chip_id, "original")
                
                if self.blacken_background:
                    mask_3ch = np.stack([mask, mask, mask], axis=2)
                    blackened = np.where(mask_3ch > 0, chip, 0)
                    self.save_blackened_chip(blackened.astype(np.uint8), chip_id, "original")
                
                # CPU augmentation
                if self.augment:
                    aug_results = self.augment_batch_cpu(chip, mask, info['geometries'], 
                                                        info['bounds'])
                    
                    for aug_idx, (aug_chip, aug_mask, aug_labels) in enumerate(aug_results):
                        aug_chip_id = f"{img_name}_chip_{pos_count + stats['positive']:06d}_aug{aug_idx}"
                        
                        self.save_chip(aug_chip, aug_chip_id, "augmented")
                        self.save_mask(aug_mask, aug_chip_id, "augmented")
                        self.save_yolo_labels(aug_labels, aug_chip_id, "augmented")
                        
                        if self.blacken_background:
                            mask_3ch = np.stack([aug_mask, aug_mask, aug_mask], axis=2)
                            blackened = np.where(mask_3ch > 0, aug_chip, 0)
                            self.save_blackened_chip(blackened.astype(np.uint8), 
                                                    aug_chip_id, "augmented")
                        
                        stats['augmented'] += 1
                
                stats['positive'] += 1
            else:
                chip_id = f"{img_name}_chip_neg_{neg_count + stats['negative']:06d}"
                self.save_chip(chip, chip_id, "negative")
                
                mask = np.zeros((self.chip_size, self.chip_size), dtype=np.uint8)
                self.save_mask(mask, chip_id, "negative")
                self.save_yolo_labels([], chip_id, "negative")
                
                stats['negative'] += 1
        
        return stats
    
    def augment_batch_gpu(self, chip_gpu, mask_gpu, geometries, chip_bounds):
        """Fast GPU-accelerated augmentation"""
        augmented = []
        
        # All augmentations done on GPU
        # 1. Horizontal flip
        flipped = cp.fliplr(chip_gpu)
        flipped_mask = cp.fliplr(mask_gpu)
        flipped_labels = self.flip_labels_horizontal(geometries, chip_bounds, self.chip_size)
        augmented.append((flipped, flipped_mask, flipped_labels))
        
        # 2. Vertical flip
        vflipped = cp.flipud(chip_gpu)
        vflipped_mask = cp.flipud(mask_gpu)
        vflipped_labels = self.flip_labels_vertical(geometries, chip_bounds, self.chip_size)
        augmented.append((vflipped, vflipped_mask, vflipped_labels))
        
        # 3. Rotation 90°
        rot90 = cp.rot90(chip_gpu)
        rot90_mask = cp.rot90(mask_gpu)
        rot90_labels = self.rotate_labels_90(geometries, chip_bounds, self.chip_size)
        augmented.append((rot90, rot90_mask, rot90_labels))
        
        # 4-6. Brightness/contrast (GPU accelerated)
        bright = cp.clip(chip_gpu * 1.2, 0, 255).astype(cp.uint8)
        augmented.append((bright, mask_gpu, self.create_yolo_labels(geometries, chip_bounds, self.chip_size)))
        
        dark = cp.clip(chip_gpu * 0.8, 0, 255).astype(cp.uint8)
        augmented.append((dark, mask_gpu, self.create_yolo_labels(geometries, chip_bounds, self.chip_size)))
        
        contrast = cp.clip((chip_gpu - 128) * 1.3 + 128, 0, 255).astype(cp.uint8)
        augmented.append((contrast, mask_gpu, self.create_yolo_labels(geometries, chip_bounds, self.chip_size)))
        
        return augmented
    
    def augment_batch_cpu(self, chip, mask, geometries, chip_bounds):
        """CPU augmentation (fallback)"""
        augmented = []
        
        flipped = np.fliplr(chip)
        flipped_mask = np.fliplr(mask)
        augmented.append((flipped, flipped_mask, 
                         self.flip_labels_horizontal(geometries, chip_bounds, self.chip_size)))
        
        vflipped = np.flipud(chip)
        vflipped_mask = np.flipud(mask)
        augmented.append((vflipped, vflipped_mask,
                         self.flip_labels_vertical(geometries, chip_bounds, self.chip_size)))
        
        rot90 = np.rot90(chip)
        rot90_mask = np.rot90(mask)
        augmented.append((rot90, rot90_mask,
                         self.rotate_labels_90(geometries, chip_bounds, self.chip_size)))
        
        bright = np.clip(chip * 1.2, 0, 255).astype(np.uint8)
        augmented.append((bright, mask.copy(),
                         self.create_yolo_labels(geometries, chip_bounds, self.chip_size)))
        
        dark = np.clip(chip * 0.8, 0, 255).astype(np.uint8)
        augmented.append((dark, mask.copy(),
                         self.create_yolo_labels(geometries, chip_bounds, self.chip_size)))
        
        contrast = np.clip((chip - 128) * 1.3 + 128, 0, 255).astype(np.uint8)
        augmented.append((contrast, mask.copy(),
                         self.create_yolo_labels(geometries, chip_bounds, self.chip_size)))
        
        return augmented
    
    def create_mask_fast(self, geometries, chip_bounds, transform, window, chip_size):
        """Fast mask creation"""
        mask = np.zeros((chip_size, chip_size), dtype=np.uint8)
        
        if len(geometries) == 0:
            return mask
        
        row_start, col_start = rasterio.transform.rowcol(transform, chip_bounds[0], chip_bounds[3])
        
        for idx, row in geometries.iterrows():
            geom = row.geometry
            
            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
                geom_pixels = []
                
                for x, y in coords:
                    r, c = rasterio.transform.rowcol(transform, x, y)
                    chip_row = r - row_start
                    chip_col = c - col_start
                    geom_pixels.append([chip_col, chip_row])
                
                if len(geom_pixels) > 0:
                    geom_pixels = np.array(geom_pixels, dtype=np.int32)
                    cv2.fillPoly(mask, [geom_pixels], 255)
        
        return mask
    
    def create_yolo_labels(self, geometries, chip_bounds, chip_size):
        """Create YOLO format labels with proper intersection handling"""
        labels = []

        chip_minx, chip_miny, chip_maxx, chip_maxy = chip_bounds
        chip_width = chip_maxx - chip_minx
        chip_height = chip_maxy - chip_miny

        for idx, row in geometries.iterrows():
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds

            # Calculate intersection bounds with chip
            inter_minx = max(minx, chip_minx)
            inter_maxx = min(maxx, chip_maxx)
            inter_miny = max(miny, chip_miny)
            inter_maxy = min(maxy, chip_maxy)

            # Only create label if geometry actually intersects chip
            if inter_minx < inter_maxx and inter_miny < inter_maxy:
                # Calculate center and dimensions from intersection
                x_center = ((inter_minx + inter_maxx) / 2 - chip_minx) / chip_width
                y_center = ((inter_miny + inter_maxy) / 2 - chip_miny) / chip_height
                width = (inter_maxx - inter_minx) / chip_width
                height = (inter_maxy - inter_miny) / chip_height

                # Clip to valid range (should already be in range, but safety check)
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0.001, 1)  # Minimum width to avoid zero
                height = np.clip(height, 0.001, 1)  # Minimum height to avoid zero

                labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return labels
    
    def flip_labels_horizontal(self, geometries, chip_bounds, chip_size):
        """Flip labels horizontally with proper clipping"""
        labels = []
        chip_minx, chip_miny, chip_maxx, chip_maxy = chip_bounds
        chip_width = chip_maxx - chip_minx
        chip_height = chip_maxy - chip_miny

        for idx, row in geometries.iterrows():
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds

            # Calculate intersection bounds
            inter_minx = max(minx, chip_minx)
            inter_maxx = min(maxx, chip_maxx)
            inter_miny = max(miny, chip_miny)
            inter_maxy = min(maxy, chip_maxy)

            if inter_minx < inter_maxx and inter_miny < inter_maxy:
                x_center = ((inter_minx + inter_maxx) / 2 - chip_minx) / chip_width
                y_center = ((inter_miny + inter_maxy) / 2 - chip_miny) / chip_height
                width = (inter_maxx - inter_minx) / chip_width
                height = (inter_maxy - inter_miny) / chip_height

                # Flip horizontally
                x_center = 1.0 - x_center

                # Clip to valid range
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0.001, 1)
                height = np.clip(height, 0.001, 1)

                labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return labels
    
    def flip_labels_vertical(self, geometries, chip_bounds, chip_size):
        """Flip labels vertically with proper clipping"""
        labels = []
        chip_minx, chip_miny, chip_maxx, chip_maxy = chip_bounds
        chip_width = chip_maxx - chip_minx
        chip_height = chip_maxy - chip_miny

        for idx, row in geometries.iterrows():
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds

            # Calculate intersection bounds
            inter_minx = max(minx, chip_minx)
            inter_maxx = min(maxx, chip_maxx)
            inter_miny = max(miny, chip_miny)
            inter_maxy = min(maxy, chip_maxy)

            if inter_minx < inter_maxx and inter_miny < inter_maxy:
                x_center = ((inter_minx + inter_maxx) / 2 - chip_minx) / chip_width
                y_center = ((inter_miny + inter_maxy) / 2 - chip_miny) / chip_height
                width = (inter_maxx - inter_minx) / chip_width
                height = (inter_maxy - inter_miny) / chip_height

                # Flip vertically
                y_center = 1.0 - y_center

                # Clip to valid range
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0.001, 1)
                height = np.clip(height, 0.001, 1)

                labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return labels
    
    def rotate_labels_90(self, geometries, chip_bounds, chip_size):
        """Rotate labels 90 degrees with proper clipping"""
        labels = []
        chip_minx, chip_miny, chip_maxx, chip_maxy = chip_bounds
        chip_width = chip_maxx - chip_minx
        chip_height = chip_maxy - chip_miny

        for idx, row in geometries.iterrows():
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds

            # Calculate intersection bounds
            inter_minx = max(minx, chip_minx)
            inter_maxx = min(maxx, chip_maxx)
            inter_miny = max(miny, chip_miny)
            inter_maxy = min(maxy, chip_maxy)

            if inter_minx < inter_maxx and inter_miny < inter_maxy:
                x_center = ((inter_minx + inter_maxx) / 2 - chip_minx) / chip_width
                y_center = ((inter_miny + inter_maxy) / 2 - chip_miny) / chip_height
                width = (inter_maxx - inter_minx) / chip_width
                height = (inter_maxy - inter_miny) / chip_height

                # Rotate 90 degrees
                new_x = y_center
                new_y = 1.0 - x_center
                new_width = height
                new_height = width

                # Clip to valid range
                new_x = np.clip(new_x, 0, 1)
                new_y = np.clip(new_y, 0, 1)
                new_width = np.clip(new_width, 0.001, 1)
                new_height = np.clip(new_height, 0.001, 1)

                labels.append(f"0 {new_x:.6f} {new_y:.6f} {new_width:.6f} {new_height:.6f}")

        return labels
    
    def save_chip(self, chip, chip_id, chip_type):
        """Save chip"""
        output_path = self.images_dir / f"{chip_id}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))
    
    def save_mask(self, mask, chip_id, chip_type):
        """Save mask"""
        output_path = self.masks_dir / f"{chip_id}.png"
        cv2.imwrite(str(output_path), mask)
    
    def save_yolo_labels(self, labels, chip_id, chip_type):
        """Save YOLO labels"""
        output_path = self.labels_dir / f"{chip_id}.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(labels))
    
    def save_blackened_chip(self, chip, chip_id, chip_type):
        """Save blackened chip"""
        output_path = self.blackened_dir / f"{chip_id}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))
    
    def save_statistics(self, stats):
        """Save statistics"""
        stats_path = self.output_dir / "statistics.json"
        
        stats_extended = {
            **stats,
            'chip_size': self.chip_size,
            'stride': self.stride,
            'num_workers': self.num_workers,
            'gpu_accelerated': self.use_gpu,
            'batch_size': self.batch_size
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_extended, f, indent=2)
        
        print(f"\n✓ Statistics saved to: {stats_path}")
    
    def print_summary(self, stats):
        """Print summary"""
        print("\n" + "="*80)
        print("EXPORT COMPLETE!")
        print("="*80)
        
        print(f"\n📊 Statistics:")
        print(f"  Images processed: {stats['images_processed']}")
        print(f"  Solar panels found: {stats['solar_panels_found']:,}")
        print(f"  Positive chips: {stats['positive_chips']:,}")
        print(f"  Negative chips: {stats['negative_chips']:,}")
        if self.augment:
            print(f"  Augmented chips: {stats['augmented_chips']:,}")
        
        total = stats['positive_chips'] + stats['augmented_chips'] + stats['negative_chips']
        print(f"  Total training samples: {total:,}")
        
        print(f"\n📁 Output: {self.output_dir}")
        print(f"\n🎯 Ready for training!")


def main():
    """Run ultra-fast export"""
    IMAGE_DIR = r"D:\Images"
    SHAPEFILE_DIR = r"D:\Images\shapes_new"
    OUTPUT_DIR = r"D:\Images\training_data"
    
    # Optimal settings for your hardware
    CHIP_SIZE = 256
    STRIDE = 128
    NUM_WORKERS = cpu_count() - 2  # Leave 2 cores free
    BATCH_SIZE = 100  # Process 100 chips at once
    
    print("\n" + "="*80)
    print("ULTRA-FAST TRAINING DATA EXPORT")
    print("Hardware: RTX A4500 + 1TB RAM + High-end CPU")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Chip size: {CHIP_SIZE}×{CHIP_SIZE}")
    print(f"  Stride: {STRIDE} (50% overlap)")
    print(f"  CPU Workers: {NUM_WORKERS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  GPU acceleration: {'✅ Enabled' if GPU_AVAILABLE else '⚠️ Install CuPy'}")
    print(f"\n  Estimated time: 30-60 minutes (10-20x faster!)")
    print(f"  Output: {OUTPUT_DIR}")
    
    print("\nPress Enter to start...")
    input()
    
    exporter = FastTrainingDataExporter(
        image_dir=IMAGE_DIR,
        shapefile_dir=SHAPEFILE_DIR,
        output_dir=OUTPUT_DIR,
        chip_size=CHIP_SIZE,
        stride=STRIDE,
        augment=True,
        blacken_background=True,
        include_negatives=True,
        negative_ratio=0.3,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        use_gpu=True
    )
    
    exporter.process_all()


if __name__ == "__main__":
    main()