"""
Data Structure Verification Script
Confirms understanding of solar panel data before preparing training dataset
"""

import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def verify_data_understanding(shapefile_dir: str, image_dir: str):
    """
    Verify critical aspects of the data
    """
    shapefile_dir = Path(shapefile_dir)
    image_dir = Path(image_dir)
    
    print("="*80)
    print("DATA STRUCTURE VERIFICATION")
    print("="*80)
    
    # 1. Verify Class column interpretation
    print("\n1️⃣  VERIFYING CLASS COLUMN INTERPRETATION")
    print("-"*80)
    
    # Pick a few sample shapefiles to verify
    test_files = ['Aaley_sp.shp', 'BeirutSP.shp', 'Nabatiyeh.shp']
    
    for shp_name in test_files:
        shp_path = shapefile_dir / shp_name
        if shp_path.exists():
            gdf = gpd.read_file(shp_path)
            
            # Find class column (case insensitive)
            class_col = None
            for col in gdf.columns:
                if col.lower() == 'class':
                    class_col = col
                    break
            
            if class_col:
                print(f"\n📋 {shp_name}:")
                value_counts = gdf[class_col].value_counts().sort_index()
                
                for class_val, count in value_counts.items():
                    pct = count / len(gdf) * 100
                    if class_val == 1:
                        print(f"   Class {class_val}: {count:>6,} features ({pct:>5.1f}%) ← SOLAR PANELS")
                    else:
                        print(f"   Class {class_val}: {count:>6,} features ({pct:>5.1f}%) ← non-solar")
                
                # Check if Class column is numeric or string
                print(f"   Data type: {gdf[class_col].dtype}")
                
                # Show sample geometries for Class 1
                solar_samples = gdf[gdf[class_col] == 1].head(3)
                print(f"   Sample solar panel areas (m²):")
                for idx, row in solar_samples.iterrows():
                    if gdf.crs and gdf.crs.is_projected:
                        area = row.geometry.area
                        print(f"      {area:.2f} m²")
    
    # 2. Verify spatial alignment
    print("\n\n2️⃣  VERIFYING SPATIAL ALIGNMENT")
    print("-"*80)
    
    import rasterio
    
    # Check a few image-shapefile pairs
    test_pairs = [
        ('Aaley.tif', 'Aaley_sp.shp'),
        ('Beirut.tif', 'BeirutSP.shp'),
        ('Nabatiyeh.tif', 'Nabatiyeh.shp')
    ]
    
    for img_name, shp_name in test_pairs:
        img_path = image_dir / img_name
        shp_path = shapefile_dir / shp_name
        
        if img_path.exists() and shp_path.exists():
            try:
                with rasterio.open(img_path) as src:
                    img_bounds = src.bounds
                    img_crs = src.crs
                
                gdf = gpd.read_file(shp_path)
                shp_bounds = gdf.total_bounds
                shp_crs = gdf.crs
                
                print(f"\n📷 {img_name} ↔ 🗺️  {shp_name}")
                print(f"   Image CRS: {img_crs}")
                print(f"   Shape CRS: {shp_crs}")
                print(f"   CRS Match: {'✅ YES' if img_crs == shp_crs else '⚠️  NO (will reproject)'}")
                
                # Check bounds overlap
                from shapely.geometry import box
                img_box = box(img_bounds.left, img_bounds.bottom, 
                             img_bounds.right, img_bounds.top)
                shp_box = box(shp_bounds[0], shp_bounds[1], 
                             shp_bounds[2], shp_bounds[3])
                
                if img_box.intersects(shp_box):
                    overlap = img_box.intersection(shp_box).area / shp_box.area * 100
                    print(f"   Spatial Overlap: {overlap:.1f}% ✅")
                else:
                    print(f"   Spatial Overlap: 0% ❌")
                    
            except Exception as e:
                print(f"   Error: {e}")
    
    # 3. Verify solar panel characteristics
    print("\n\n3️⃣  ANALYZING SOLAR PANEL CHARACTERISTICS")
    print("-"*80)
    
    all_solar_areas = []
    all_solar_perimeters = []
    
    for shp_path in shapefile_dir.glob("*.shp"):
        if 'Solar_' in shp_path.stem:  # Skip standalone solar files
            continue
            
        try:
            gdf = gpd.read_file(shp_path)
            
            # Find class column
            class_col = None
            for col in gdf.columns:
                if col.lower() == 'class':
                    class_col = col
                    break
            
            if class_col and gdf.crs and gdf.crs.is_projected:
                solar_gdf = gdf[gdf[class_col] == 1]
                
                if len(solar_gdf) > 0:
                    areas = solar_gdf.geometry.area.values
                    all_solar_areas.extend(areas)
                    
        except Exception as e:
            continue
    
    if all_solar_areas:
        areas = np.array(all_solar_areas)
        
        print(f"\n📊 Solar Panel Size Statistics (from {len(areas):,} panels):")
        print(f"   Min area: {areas.min():.2f} m²")
        print(f"   Max area: {areas.max():.2f} m²")
        print(f"   Mean area: {areas.mean():.2f} m²")
        print(f"   Median area: {np.median(areas):.2f} m²")
        print(f"   Std dev: {areas.std():.2f} m²")
        
        # Show distribution
        print(f"\n   Size Distribution:")
        bins = [0, 20, 50, 100, 200, 500, 1000, areas.max()]
        hist, _ = np.histogram(areas, bins=bins)
        
        for i in range(len(bins)-1):
            count = hist[i]
            pct = count / len(areas) * 100
            print(f"      {bins[i]:>6.0f} - {bins[i+1]:>6.0f} m²: {count:>7,} panels ({pct:>5.1f}%)")
    
    # 4. Class value meanings verification
    print("\n\n4️⃣  CONFIRMING CLASS VALUE MEANINGS")
    print("-"*80)
    
    print("\nBased on the data analysis, please confirm:")
    print("\n   Class 1 = Solar panels (positive samples)")
    print("   Class 2-6 = Non-solar features (negative samples)")
    print("\n   Is this correct? (This is critical for model training!)")
    
    # 5. Check for edge cases
    print("\n\n5️⃣  CHECKING FOR EDGE CASES")
    print("-"*80)
    
    issues_found = []
    
    for shp_path in shapefile_dir.glob("*.shp"):
        if 'Solar_' in shp_path.stem:
            continue
            
        try:
            gdf = gpd.read_file(shp_path)
            
            # Check for class column
            class_col = None
            for col in gdf.columns:
                if col.lower() == 'class':
                    class_col = col
                    break
            
            if not class_col:
                issues_found.append(f"⚠️  {shp_path.name}: No 'class' column found")
                continue
            
            # Check if Class is numeric
            if gdf[class_col].dtype == 'object':
                # Try to convert to numeric
                try:
                    gdf[class_col] = pd.to_numeric(gdf[class_col])
                except:
                    issues_found.append(f"⚠️  {shp_path.name}: Class column is string and can't convert to numeric")
            
            # Check for unexpected class values
            unique_classes = gdf[class_col].unique()
            unexpected = [c for c in unique_classes if c not in [1, 2, 3, 4, 5, 6]]
            if unexpected:
                issues_found.append(f"⚠️  {shp_path.name}: Unexpected class values: {unexpected}")
            
            # Check for null geometries
            null_geoms = gdf.geometry.isna().sum()
            if null_geoms > 0:
                issues_found.append(f"⚠️  {shp_path.name}: {null_geoms} null geometries")
            
            # Check for very small or very large polygons (potential errors)
            if gdf.crs and gdf.crs.is_projected:
                areas = gdf.geometry.area
                very_small = (areas < 1).sum()  # < 1 m²
                very_large = (areas > 10000).sum()  # > 1 hectare
                
                if very_small > len(gdf) * 0.1:  # More than 10% very small
                    issues_found.append(f"⚠️  {shp_path.name}: {very_small} very small features (< 1 m²)")
                
                if very_large > 0:
                    issues_found.append(f"ℹ️  {shp_path.name}: {very_large} very large features (> 10,000 m²)")
                    
        except Exception as e:
            issues_found.append(f"❌ {shp_path.name}: Error reading file - {e}")
    
    if issues_found:
        print("\n⚠️  Issues found:")
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print("\n✅ No issues found - data looks clean!")
    
    # 6. Summary
    print("\n\n6️⃣  FINAL SUMMARY")
    print("="*80)
    
    # Count totals
    total_solar = 0
    total_non_solar = 0
    total_files_with_class = 0
    total_files_no_class = 0
    
    for shp_path in shapefile_dir.glob("*.shp"):
        if 'Solar_' in shp_path.stem:
            continue
            
        try:
            gdf = gpd.read_file(shp_path)
            
            class_col = None
            for col in gdf.columns:
                if col.lower() == 'class':
                    class_col = col
                    break
            
            if class_col:
                total_files_with_class += 1
                
                # Convert to numeric if string
                if gdf[class_col].dtype == 'object':
                    try:
                        gdf[class_col] = pd.to_numeric(gdf[class_col])
                    except:
                        pass
                
                solar_count = (gdf[class_col] == 1).sum()
                non_solar_count = (gdf[class_col] != 1).sum()
                
                total_solar += solar_count
                total_non_solar += non_solar_count
            else:
                total_files_no_class += 1
                
        except:
            continue
    
    print(f"\n✅ Data Understanding Confirmed:")
    print(f"   • Total shapefiles: {total_files_with_class + total_files_no_class}")
    print(f"   • Shapefiles with 'Class' column: {total_files_with_class}")
    print(f"   • Shapefiles without 'Class' column: {total_files_no_class}")
    print(f"   • Total solar panels (Class=1): {total_solar:,}")
    print(f"   • Total non-solar features (Class≠1): {total_non_solar:,}")
    print(f"   • Solar panel ratio: {total_solar/(total_solar+total_non_solar)*100:.1f}%")
    
    print(f"\n📋 Training Data Will Include:")
    print(f"   • Positive samples: {total_solar:,} solar panels")
    print(f"   • Negative samples: {total_non_solar:,} non-solar features")
    print(f"   • Class balance: {'Good' if total_solar/(total_solar+total_non_solar) > 0.3 else 'Imbalanced (will use weighted loss)'}")
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE - Ready to prepare training data!")
    print("="*80)


def main():
    """
    Run verification
    """
    IMAGE_DIR = r"D:\Images"
    SHAPEFILE_DIR = r"D:\Images\shapes_new"
    
    verify_data_understanding(SHAPEFILE_DIR, IMAGE_DIR)
    
    print("\n\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. If everything looks correct above, proceed with:
   python analyze_solar_with_class.py
   
2. Then prepare training data:
   python prepare_data_with_class.py
   
3. Then train your model!

If you see any unexpected results above, please review before proceeding.
    """)


if __name__ == "__main__":
    import pandas as pd
    main()
