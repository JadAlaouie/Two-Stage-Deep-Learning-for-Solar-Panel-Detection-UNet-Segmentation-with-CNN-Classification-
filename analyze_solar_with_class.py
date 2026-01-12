"""
Solar Panel Detection - Analysis with Class Filtering
Analyzes building shapefiles and filters for solar panels (class == 1)
Includes negative samples (class != 1) for robust training
"""

import rasterio
import geopandas as gpd
from pathlib import Path
import numpy as np
from shapely.geometry import box
import pandas as pd
from tabulate import tabulate


class SolarPanelClassAnalyzer:
    """
    Analyzer that handles shapefiles with class column
    class == 1: Solar panels (positive samples)
    class != 1: Non-solar buildings (negative samples)
    """
    
    def __init__(self, image_dir: str, shapefile_dir: str):
        self.image_dir = Path(image_dir)
        self.shapefile_dir = Path(shapefile_dir)
        self.image_info = {}
        self.shapefile_info = {}
        self.matches = {}
    
    def analyze_all(self):
        """
        Run complete analysis
        """
        print("="*80)
        print("SOLAR PANEL DETECTION - CLASS-BASED ANALYSIS")
        print("="*80)
        
        # Analyze images
        print("\n" + "="*80)
        print("1. ANALYZING GEOTIFF IMAGES")
        print("="*80)
        self.analyze_images()
        
        # Analyze shapefiles
        print("\n" + "="*80)
        print("2. ANALYZING SHAPEFILES WITH CLASS COLUMN")
        print("="*80)
        self.analyze_shapefiles()
        
        # Match images with shapefiles
        print("\n" + "="*80)
        print("3. MATCHING IMAGES WITH LABELS")
        print("="*80)
        self.match_images_to_shapefiles()
        
        # Summary
        print("\n" + "="*80)
        print("4. SUMMARY AND RECOMMENDATIONS")
        print("="*80)
        self.print_summary()
    
    def analyze_images(self):
        """
        Analyze GeoTIFF images (excluding specified ones)
        """
        # Images to exclude
        exclude_images = ['alghab.tif', 'yafi.tif', 'huraida.tif']
        
        tif_files = sorted(list(self.image_dir.glob("*.tif")))
        tif_files = [f for f in tif_files if f.name.lower() not in 
                    [e.lower() for e in exclude_images]]
        
        print(f"\nAnalyzing {len(tif_files)} images (excluded: {', '.join(exclude_images)})\n")
        
        for tif_path in tif_files:
            try:
                with rasterio.open(tif_path) as src:
                    info = {
                        'path': tif_path,
                        'name': tif_path.name,
                        'crs': src.crs,
                        'crs_string': str(src.crs),
                        'bounds': src.bounds,
                        'width': src.width,
                        'height': src.height,
                        'count': src.count,
                        'dtype': src.dtypes[0],
                        'transform': src.transform,
                        'resolution': (src.transform[0], abs(src.transform[4])),
                        'size_mb': tif_path.stat().st_size / (1024 * 1024),
                    }
                    
                    # Calculate area
                    width_meters = info['width'] * info['resolution'][0]
                    height_meters = info['height'] * info['resolution'][1]
                    info['area_km2'] = (width_meters * height_meters) / 1_000_000
                    
                    self.image_info[tif_path.stem] = info
                    
                    # Print brief info
                    print(f"📷 {info['name']:<20} | CRS: {info['crs_string']:<12} | "
                          f"{info['width']:>5}x{info['height']:<5} | "
                          f"{info['count']} bands | {info['area_km2']:>6.1f} km²")
                    
            except Exception as e:
                print(f"❌ Error reading {tif_path.name}: {e}")
        
        print(f"\n✓ Successfully analyzed {len(self.image_info)} images")
    
    def analyze_shapefiles(self):
        """
        Analyze shapefiles and count solar panels vs non-solar buildings
        """
        shp_files = sorted(list(self.shapefile_dir.glob("*.shp")))
        
        # Exclude Solar_* shapefiles (we're using the building shapefiles with class column)
        shp_files = [f for f in shp_files if not f.stem.startswith('Solar')]
        
        print(f"\nAnalyzing {len(shp_files)} building shapefiles\n")
        print(f"{'Shapefile':<30} | {'CRS':<12} | {'Total':>7} | {'Solar':>6} | {'Non-Solar':>10}")
        print("-" * 80)
        
        for shp_path in shp_files:
            try:
                gdf = gpd.read_file(shp_path)
                
                # Check if 'class' column exists
                has_class = 'class' in gdf.columns or 'Class' in gdf.columns
                class_col = 'class' if 'class' in gdf.columns else 'Class' if 'Class' in gdf.columns else None
                
                if has_class and class_col:
                    # Count solar panels (class == 1) vs non-solar (class != 1)
                    solar_count = (gdf[class_col] == 1).sum()
                    non_solar_count = (gdf[class_col] != 1).sum()
                else:
                    solar_count = 0
                    non_solar_count = len(gdf)
                
                info = {
                    'path': shp_path,
                    'name': shp_path.name,
                    'crs': gdf.crs,
                    'crs_string': str(gdf.crs),
                    'total_features': len(gdf),
                    'solar_count': int(solar_count),
                    'non_solar_count': int(non_solar_count),
                    'has_class_column': has_class,
                    'class_column': class_col,
                    'bounds': gdf.total_bounds,
                    'geometry_type': gdf.geom_type.unique().tolist(),
                    'columns': gdf.columns.tolist(),
                }
                
                # Calculate areas
                if gdf.crs and gdf.crs.is_projected:
                    info['total_area_m2'] = gdf.geometry.area.sum()
                    if solar_count > 0:
                        solar_gdf = gdf[gdf[class_col] == 1]
                        info['solar_area_m2'] = solar_gdf.geometry.area.sum()
                    else:
                        info['solar_area_m2'] = 0
                else:
                    info['total_area_m2'] = None
                    info['solar_area_m2'] = 0
                
                self.shapefile_info[shp_path.stem] = info
                
                # Print info
                print(f"{info['name']:<30} | {info['crs_string']:<12} | "
                      f"{info['total_features']:>7} | {solar_count:>6} | {non_solar_count:>10}")
                    
            except Exception as e:
                print(f"❌ Error reading {shp_path.name}: {e}")
        
        print(f"\n✓ Successfully analyzed {len(self.shapefile_info)} shapefiles")
    
    def match_images_to_shapefiles(self):
        """
        Match images with their corresponding shapefiles
        """
        print("\nMatching Images with Building Labels:\n")
        
        # Define name mappings
        name_mappings = {
            'Aaley': 'Aaley_sp',
            'Aanjar': 'AnjarSP',
            'Barr_Elias': 'BarEliasSP',
            'Batroun': 'BatrounSP',
            'Bcharre': 'BcharreSP',
            'Beirut': 'BeirutSP',
            'Beit_Eddine': 'BeitEddineSP',
            'Beskinta': 'Beskinta',
            'Bhamdoun': 'bhamdoun_sp',
            'Bikfayya': 'Bikfayya_SP',
            'Bint_Jbeil': 'Bint_Jbeil',
            'Britel': 'BritelSP',
            'Chmistar': 'ChmistarSP',
            'Damour': 'DamourSP',
            'Deir_AlAhmar': 'Deir_AlAhmar_SP',
            'Ehden': 'Ehden_SP',
            'Ezzrareyye': 'Ezzrareyye',
            'Hermel': 'Hermel_SP',
            'Nabatiyeh': 'Nabatiyeh',
            'Ras_Baalback': 'Ras_BaalbakSP',
            'Rayak': 'Rayak_SP',
            'Saida': 'SaidaSP1',
            'Sour': 'SourSP',
            'Zahli': None,
        }
        
        matches_found = 0
        crs_mismatches = 0
        no_class_col = []
        
        for img_name, img_info in self.image_info.items():
            print(f"\n📷 {img_name}.tif")
            print("-" * 70)
            
            # Get expected shapefile name
            expected_shp = name_mappings.get(img_name)
            
            if expected_shp is None:
                expected_shp = self._find_matching_shapefile(img_name)
            
            if expected_shp and expected_shp in self.shapefile_info:
                shp_info = self.shapefile_info[expected_shp]
                
                # Check if shapefile has class column
                if not shp_info['has_class_column']:
                    print(f"   ⚠️  Matched with: {expected_shp}")
                    print(f"   ❌ NO CLASS COLUMN - Cannot identify solar panels")
                    no_class_col.append((img_name, expected_shp))
                    continue
                
                # Check CRS
                crs_match = img_info['crs'] == shp_info['crs']
                
                print(f"   ✓ Matched with: {expected_shp}")
                print(f"   Total Buildings: {shp_info['total_features']:,}")
                print(f"   Solar Panels (class=1): {shp_info['solar_count']:,}")
                print(f"   Non-Solar (class!=1): {shp_info['non_solar_count']:,}")
                
                if shp_info['solar_count'] == 0:
                    print(f"   ⚠️  NO SOLAR PANELS FOUND (all class != 1)")
                    continue
                
                print(f"   CRS Match: {'✅' if crs_match else '⚠️  NO - Will reproject'}")
                
                if not crs_match:
                    print(f"      Image CRS: {img_info['crs_string']}")
                    print(f"      Shape CRS: {shp_info['crs_string']}")
                    crs_mismatches += 1
                
                # Check spatial overlap
                overlap_pct = self._check_spatial_overlap(img_info, shp_info)
                print(f"   Spatial Overlap: {overlap_pct:.1f}%")
                
                if overlap_pct > 50:
                    print(f"   ✅ READY FOR TRAINING")
                    matches_found += 1
                    
                    self.matches[img_name] = {
                        'image': img_info,
                        'shapefile': shp_info,
                        'shapefile_name': expected_shp,
                        'crs_match': crs_match,
                        'overlap_pct': overlap_pct
                    }
                elif overlap_pct > 0:
                    print(f"   ⚠️  Partial overlap - verify alignment")
                else:
                    print(f"   ❌ No spatial overlap - check data")
            else:
                print(f"   ❌ No matching shapefile found")
        
        print(f"\n{'='*70}")
        print(f"Matching Summary:")
        print(f"  ✓ Matches found: {matches_found}")
        print(f"  ⚠️  CRS mismatches: {crs_mismatches} (will auto-reproject)")
        if no_class_col:
            print(f"  ⚠️  No class column: {len(no_class_col)} shapefiles")
            for img, shp in no_class_col:
                print(f"     {img} → {shp}")
    
    def _find_matching_shapefile(self, img_name: str) -> str:
        """
        Try to find matching shapefile using fuzzy name matching
        """
        img_name_lower = img_name.lower().replace('_', '').replace('-', '')
        
        for shp_name in self.shapefile_info.keys():
            shp_name_clean = shp_name.lower().replace('_', '').replace('-', '').replace('sp', '')
            
            if img_name_lower in shp_name_clean or shp_name_clean in img_name_lower:
                return shp_name
        
        return None
    
    def _check_spatial_overlap(self, img_info, shp_info):
        """
        Calculate percentage of shapefile features within image bounds
        """
        try:
            img_bounds = box(*img_info['bounds'])
            shp_bounds = box(*shp_info['bounds'])
            
            if img_bounds.intersects(shp_bounds):
                intersection = img_bounds.intersection(shp_bounds)
                overlap = (intersection.area / shp_bounds.area) * 100
                return overlap
            else:
                return 0
        except:
            return 0
    
    def print_summary(self):
        """
        Print summary and recommendations
        """
        print("\n📊 FINAL SUMMARY")
        print("="*70)
        
        total_images = len(self.image_info)
        matched_images = len(self.matches)
        
        # Calculate totals
        total_solar = sum(m['shapefile']['solar_count'] for m in self.matches.values())
        total_non_solar = sum(m['shapefile']['non_solar_count'] for m in self.matches.values())
        total_buildings = total_solar + total_non_solar
        
        print(f"\n✓ Images analyzed: {total_images}")
        print(f"✓ Matched pairs ready: {matched_images}")
        print(f"✓ Total buildings: {total_buildings:,}")
        print(f"✓ Solar panels (class=1): {total_solar:,}")
        print(f"✓ Non-solar buildings (class!=1): {total_non_solar:,}")
        print(f"✓ Solar panel ratio: {total_solar/total_buildings*100:.2f}%")
        
        # List all ready pairs
        if self.matches:
            print(f"\n📋 READY FOR TRAINING:")
            print("="*70)
            
            # Create table
            table_data = []
            for img_name, match_info in sorted(self.matches.items()):
                shp_name = match_info['shapefile_name']
                solar_count = match_info['shapefile']['solar_count']
                non_solar_count = match_info['shapefile']['non_solar_count']
                total_count = solar_count + non_solar_count
                crs_ok = "✅" if match_info['crs_match'] else "⚠️"
                
                table_data.append([
                    f"{img_name}.tif",
                    f"{shp_name}.shp",
                    f"{total_count:,}",
                    f"{solar_count:,}",
                    f"{non_solar_count:,}",
                    crs_ok
                ])
            
            headers = ["Image", "Shapefile", "Total", "Solar", "Non-Solar", "CRS"]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))
        
        # Training data estimation
        print(f"\n📈 TRAINING DATA ESTIMATION:")
        print("="*70)
        
        chip_size = 512
        stride = 256
        
        total_chips = 0
        for img_name, match_info in self.matches.items():
            img_info = match_info['image']
            n_chips_x = (img_info['width'] - chip_size) // stride + 1
            n_chips_y = (img_info['height'] - chip_size) // stride + 1
            img_chips = n_chips_x * n_chips_y
            total_chips += img_chips
        
        print(f"  Chip size: {chip_size}x{chip_size} pixels")
        print(f"  Stride: {stride} pixels (50% overlap)")
        print(f"  Total possible chips: ~{total_chips:,}")
        print(f"  Estimated chips with buildings: ~{total_chips // 5:,}")
        print(f"  Solar panels: {total_solar:,}")
        print(f"  Non-solar buildings: {total_non_solar:,}")
        
        # Model recommendations
        print(f"\n🎯 RECOMMENDED WORKFLOW:")
        print("="*70)
        print("""
Since you have BOTH positive (solar) and negative (non-solar) samples:

1. DATA PREPARATION (Following ArcGIS workflow):
   • Metadata Format: Classified_Tiles (for segmentation) or PASCAL_VOC_rectangles (for detection)
   • Extract chips from images
   • Filter labels by class column (class == 1 for solar panels)
   • Include negative samples for robust training
   
2. MODEL ARCHITECTURE (Recommended):
  
   OPTION A: U-Net (Semantic Segmentation) ⭐ RECOMMENDED
   • Best for: Binary classification (solar vs non-solar)
   • Metadata: Classified_Tiles
   • Output: Pixel-wise mask
   • Training time: 2-4 hours with GPU
   • Advantage: Handles class imbalance well
   
   OPTION B: Mask R-CNN (Instance Segmentation)
   • Best for: Detecting individual panels
   • Metadata: RCNN_Masks  
   • Output: Polygon per panel
   • Training time: 6-12 hours with GPU
   • Advantage: Precise boundaries, counts each panel
   
3. CLASS IMBALANCE HANDLING:
   • Solar panels: {total_solar:,} ({total_solar/total_buildings*100:.1f}%)
   • Non-solar: {total_non_solar:,} ({total_non_solar/total_buildings*100:.1f}%)
   • Use weighted loss function
   • Sample more chips with solar panels
   • Data augmentation for solar panel class
        """)
        
        print(f"\n✅ Analysis complete! Ready to prepare training data.")
        print(f"\nNext step: Run prepare_data_with_class.py to extract training chips")


def main():
    """
    Run the analysis
    """
    # UPDATE THESE PATHS
    IMAGE_DIR = r"D:\Images"
    SHAPEFILE_DIR = r"D:\Images\shapes_new"
    
    analyzer = SolarPanelClassAnalyzer(IMAGE_DIR, SHAPEFILE_DIR)
    analyzer.analyze_all()


if __name__ == "__main__":
    main()