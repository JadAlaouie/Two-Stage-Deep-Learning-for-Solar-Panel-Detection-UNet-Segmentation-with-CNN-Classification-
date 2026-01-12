"""
U-Net++ Inference Script for Solar Panel Detection
Outputs: Segmented images + Shapefile
GPU ID: 1 (as requested)
Model: unetplusplus_epoch90.pth
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import shape, Polygon
import warnings
import time
warnings.filterwarnings('ignore')


class UNetPlusPlusInference:
    """
    U-Net++ Inference for Large GeoTIFF Images
    Outputs segmentation masks and shapefiles
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        gpu_id: int = 1,
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 16,
        threshold: float = 0.5,
        encoder: str = "efficientnet-b7",
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_id = gpu_id
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.threshold = threshold
        self.encoder = encoder

        # Set GPU
        self.device = torch.device(f'cuda:{gpu_id}')

        print("="*80)
        print("U-NET++ SOLAR PANEL INFERENCE")
        print("="*80)
        print(f"\nGPU Configuration:")
        print(f"  GPU ID: {gpu_id}")
        print(f"  Device: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")

        print(f"\nModel Configuration:")
        print(f"  Model: {self.model_path.name}")
        print(f"  Encoder: {encoder}")
        print(f"  Tile size: {tile_size}x{tile_size}")
        print(f"  Overlap: {overlap}px")
        print(f"  Batch size: {batch_size}")
        print(f"  Threshold: {threshold}")

        # Load model
        self.model = None
        self.transform = None
        self._build_model()

    def _build_model(self):
        """Build and load model"""
        print("\n" + "="*80)
        print("LOADING MODEL")
        print("="*80)

        # Create model
        self.model = smp.UnetPlusPlus(
            encoder_name=self.encoder,
            encoder_weights=None,  # We'll load trained weights
            in_channels=3,
            classes=1,
            activation=None,
        )

        # Load checkpoint
        print(f"\nLoading checkpoint: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best IoU: {checkpoint.get('best_val_iou', 'N/A'):.4f}")
            print(f"  Best Dice: {checkpoint.get('best_val_dice', 'N/A'):.4f}")
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Validation transform (same as training)
        self.transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

        print(f"✓ Model loaded and ready for inference")

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

    def _predict_tiles(self, tiles):
        """Predict on batch of tiles"""
        # Transform tiles
        transformed = []
        for tile in tiles:
            augmented = self.transform(image=tile)
            transformed.append(augmented['image'])

        # Stack into batch
        batch = torch.stack(transformed).to(self.device)

        # Predict
        with torch.no_grad():
            with autocast():
                outputs = self.model(batch)
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

    def predict_image(self, image_path: str, save_visualizations: bool = True):
        """
        Predict solar panels on large GeoTIFF image

        Args:
            image_path: Path to input GeoTIFF
            save_visualizations: Whether to save visualization images

        Returns:
            prediction_mask: Binary mask of predictions
            metadata: GeoTIFF metadata for georeferencing
        """
        image_path = Path(image_path)
        print("\n" + "="*80)
        print(f"PROCESSING: {image_path.name}")
        print("="*80)

        # Read GeoTIFF with rasterio
        with rasterio.open(image_path) as src:
            # Read RGB bands (assuming bands 1,2,3 are RGB)
            image = src.read([1, 2, 3]).transpose(1, 2, 0)  # (H, W, 3)

            # Get metadata for georeferencing
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
            print(f"  Bounds: {src.bounds}")

        # Normalize image to 0-255 range if needed
        if image.max() > 255:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        h, w = image.shape[:2]
        print(f"  Image shape: {h} x {w} x {image.shape[2]}")

        # Extract tiles
        print(f"\nExtracting tiles...")
        tiles, positions = self._extract_tiles(image, self.tile_size, self.overlap)
        print(f"  Total tiles: {len(tiles)}")

        # Predict in batches
        print(f"\nRunning inference...")
        all_predictions = []

        pbar = tqdm(range(0, len(tiles), self.batch_size), desc="Processing batches")
        for i in pbar:
            batch_tiles = tiles[i:i+self.batch_size]
            batch_preds = self._predict_tiles(batch_tiles)
            all_predictions.extend(batch_preds)

        # Merge predictions
        print(f"\nMerging predictions...")
        prediction_map = self._merge_tiles(
            all_predictions,
            positions,
            (h, w),
            self.tile_size,
            self.overlap
        )

        # Apply threshold
        binary_mask = (prediction_map > self.threshold).astype(np.uint8)

        # Calculate statistics
        total_pixels = h * w
        panel_pixels = binary_mask.sum()
        coverage_pct = (panel_pixels / total_pixels) * 100

        print(f"\nPrediction Statistics:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Panel pixels: {panel_pixels:,}")
        print(f"  Coverage: {coverage_pct:.2f}%")

        # Save outputs
        output_name = image_path.stem

        # 1. Save binary mask as GeoTIFF
        mask_path = self.output_dir / f"{output_name}_mask.tif"
        self._save_geotiff(binary_mask, mask_path, metadata)
        print(f"\n✓ Saved mask: {mask_path}")

        # 2. Save probability map as GeoTIFF
        prob_path = self.output_dir / f"{output_name}_probability.tif"
        self._save_geotiff(prediction_map, prob_path, metadata)
        print(f"✓ Saved probability map: {prob_path}")

        # 3. Save visualizations if requested
        if save_visualizations:
            # Overlay visualization
            overlay = self._create_overlay(image, binary_mask)
            overlay_path = self.output_dir / f"{output_name}_overlay.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved overlay: {overlay_path}")

            # Side-by-side comparison
            comparison = self._create_comparison(image, binary_mask, prediction_map)
            comparison_path = self.output_dir / f"{output_name}_comparison.png"
            cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved comparison: {comparison_path}")

        return binary_mask, prediction_map, metadata

    def _save_geotiff(self, data, output_path, metadata):
        """Save array as georeferenced GeoTIFF"""
        # Ensure data is 2D
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]  # Add band dimension

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

    def _create_overlay(self, image, mask, alpha=0.5):
        """Create overlay of mask on image"""
        overlay = image.copy()

        # Create red mask
        red_mask = np.zeros_like(image)
        red_mask[:, :, 0] = mask * 255  # Red channel

        # Blend
        overlay = cv2.addWeighted(overlay, 1-alpha, red_mask, alpha, 0)

        return overlay

    def _create_comparison(self, image, mask, probability_map):
        """Create side-by-side comparison"""
        h, w = image.shape[:2]

        # Resize for display if too large
        max_size = 2000
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h))
            probability_map = cv2.resize(probability_map, (new_w, new_h))
            h, w = new_h, new_w

        # Create 3-panel comparison
        comparison = np.zeros((h, w*3, 3), dtype=np.uint8)

        # Original image
        comparison[:, :w] = image

        # Binary mask
        mask_rgb = np.stack([mask*255]*3, axis=-1)
        comparison[:, w:2*w] = mask_rgb

        # Probability heatmap
        prob_colored = cv2.applyColorMap((probability_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        prob_colored = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)
        comparison[:, 2*w:3*w] = prob_colored

        return comparison

    def create_shapefile(self, mask, metadata, output_path, min_area=10):
        """
        Convert binary mask to shapefile of polygons

        Args:
            mask: Binary mask (H, W)
            metadata: GeoTIFF metadata with CRS and transform
            output_path: Path to output shapefile
            min_area: Minimum polygon area in pixels
        """
        print("\n" + "="*80)
        print("CREATING SHAPEFILE")
        print("="*80)

        # Extract shapes (polygons) from mask
        print(f"\nExtracting polygons...")
        mask_uint8 = mask.astype(np.uint8)

        # Get shapes with rasterio
        results = list(shapes(mask_uint8, transform=metadata['transform']))

        # Filter and convert to GeoDataFrame
        polygons = []
        for geom, value in results:
            if value == 1:  # Only keep panel polygons (value=1)
                poly = shape(geom)
                if poly.area >= min_area:  # Filter small polygons
                    polygons.append(poly)

        print(f"  Found {len(polygons)} polygons (min area: {min_area} px)")

        if len(polygons) == 0:
            print("⚠ No polygons found! Skipping shapefile creation.")
            return

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'geometry': polygons},
            crs=metadata['crs']
        )

        # Add attributes
        gdf['area_px'] = gdf.geometry.area  # Area in pixel units
        gdf['area_m2'] = gdf.geometry.area  # Will be in map units if CRS is projected
        gdf['panel_id'] = range(1, len(gdf) + 1)

        # Calculate total area
        total_area = gdf['area_m2'].sum()

        print(f"\nShapefile Statistics:")
        print(f"  Total polygons: {len(gdf)}")
        print(f"  Total area: {total_area:.2f} square units")
        print(f"  Mean polygon area: {gdf['area_m2'].mean():.2f}")
        print(f"  Min polygon area: {gdf['area_m2'].min():.2f}")
        print(f"  Max polygon area: {gdf['area_m2'].max():.2f}")

        # Save shapefile
        output_path = Path(output_path)
        gdf.to_file(output_path)
        print(f"\n✓ Shapefile saved: {output_path}")

        # Also save as GeoJSON for easier viewing
        geojson_path = output_path.with_suffix('.geojson')
        gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"✓ GeoJSON saved: {geojson_path}")

        return gdf


def main():
    """Run inference on huraida.tif"""

    # Configuration
    INPUT_IMAGE = r"D:\Images\huraida.tif"
    MODEL_PATH = r"D:\Images\models\unetplusplus\unetplusplus_epoch90.pth"
    OUTPUT_DIR = r"D:\Images\PV-Panel-Detection\inference_results"

    GPU_ID = 1  # Use GPU 1 as requested
    TILE_SIZE = 512
    OVERLAP = 64
    BATCH_SIZE = 16
    THRESHOLD = 0.5
    MIN_POLYGON_AREA = 10  # Minimum area in pixels

    print("\n" + "="*80)
    print("U-NET++ SOLAR PANEL INFERENCE SCRIPT")
    print("="*80)
    print(f"\nInput: {INPUT_IMAGE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"GPU ID: {GPU_ID}")

    # Create inference engine
    inference = UNetPlusPlusInference(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        gpu_id=GPU_ID,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        batch_size=BATCH_SIZE,
        threshold=THRESHOLD,
    )

    # Run prediction
    start_time = time.time()

    binary_mask, probability_map, metadata = inference.predict_image(
        image_path=INPUT_IMAGE,
        save_visualizations=True
    )

    # Create shapefile
    output_name = Path(INPUT_IMAGE).stem
    shapefile_path = Path(OUTPUT_DIR) / f"{output_name}_panels.shp"

    gdf = inference.create_shapefile(
        mask=binary_mask,
        metadata=metadata,
        output_path=shapefile_path,
        min_area=MIN_POLYGON_AREA
    )

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"\nProcessing time: {elapsed_time/60:.2f} minutes")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - Binary mask: {output_name}_mask.tif")
    print(f"  - Probability map: {output_name}_probability.tif")
    print(f"  - Overlay image: {output_name}_overlay.png")
    print(f"  - Comparison image: {output_name}_comparison.png")
    print(f"  - Shapefile: {output_name}_panels.shp")
    print(f"  - GeoJSON: {output_name}_panels.geojson")

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
