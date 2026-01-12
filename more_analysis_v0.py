"""
Fix Data Issues Found in Verification
Handles edge cases before preparing training data
"""

import geopandas as gpd
from pathlib import Path
import numpy as np
import shutil


def fix_data_issues(shapefile_dir: str, backup: bool = True):
    """
    Fix identified issues:
    1. bhamdoun_sp.shp: Has Class=8 and NaN values + 4 null geometries
    2. Bint_Jbeil.shp: No 'class' column (has 'Class_01' instead)
    3. Ezzrareyye.shp: No 'class' column (has 'Class_01' instead)
    4. Very large features (>10,000 m²) - these are likely solar farms, keep them
    """
    shapefile_dir = Path(shapefile_dir)
    
    print("="*80)
    print("FIXING DATA ISSUES")
    print("="*80)
    
    if backup:
        backup_dir = shapefile_dir / "backup_original"
        backup_dir.mkdir(exist_ok=True)
        print(f"\n📦 Creating backups in: {backup_dir}")
    
    # Issue 1: Fix bhamdoun_sp.shp
    print("\n1️⃣  Fixing bhamdoun_sp.shp")
    print("-"*80)
    
    bhamdoun_path = shapefile_dir / "bhamdoun_sp.shp"
    if bhamdoun_path.exists():
        if backup:
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                src = shapefile_dir / f"bhamdoun_sp{ext}"
                if src.exists():
                    shutil.copy(src, backup_dir / f"bhamdoun_sp{ext}")
        
        gdf = gpd.read_file(bhamdoun_path)
        
        print(f"   Original: {len(gdf)} features")
        print(f"   Columns: {list(gdf.columns)}")
        
        # Find class column (case insensitive)
        class_col = None
        for col in gdf.columns:
            if col.lower() == 'class':
                class_col = col
                break
        
        if not class_col:
            print(f"   ⚠️  No class column found! Available columns: {list(gdf.columns)}")
            return
        
        print(f"   Using column: '{class_col}'")
        print(f"   Class values: {gdf[class_col].unique()}")
        print(f"   Null geometries: {gdf.geometry.isna().sum()}")
        
        # Remove null geometries
        gdf = gdf[~gdf.geometry.isna()]
        
        # Convert Class to numeric (it's stored as strings!)
        gdf[class_col] = pd.to_numeric(gdf[class_col], errors='coerce')
        
        # Fix Class column - treat Class=8 as non-solar (Class=2)
        # Replace NaN with 2 (non-solar)
        gdf[class_col] = gdf[class_col].fillna(2)
        gdf.loc[gdf[class_col] == 8, class_col] = 2
        
        # Convert to int
        gdf[class_col] = gdf[class_col].astype(np.int32)
        
        # Standardize to lowercase 'class'
        if class_col != 'class':
            gdf.rename(columns={class_col: 'class'}, inplace=True)
            class_col = 'class'
        
        print(f"   After fix: {len(gdf)} features")
        print(f"   Class values: {gdf[class_col].unique()}")
        print(f"   Solar panels (Class=1): {(gdf[class_col] == 1).sum()}")
        print(f"   Non-solar (Class≠1): {(gdf[class_col] != 1).sum()}")
        
        # Save fixed version
        gdf.to_file(bhamdoun_path)
        print(f"   ✅ Fixed and saved")
    else:
        print(f"   ⚠️  File not found")
    
    # Issue 2: Fix Bint_Jbeil.shp
    print("\n2️⃣  Fixing Bint_Jbeil.shp")
    print("-"*80)
    
    bint_jbeil_path = shapefile_dir / "Bint_Jbeil.shp"
    if bint_jbeil_path.exists():
        if backup:
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                src = shapefile_dir / f"Bint_Jbeil{ext}"
                if src.exists():
                    shutil.copy(src, backup_dir / f"Bint_Jbeil{ext}")
        
        gdf = gpd.read_file(bint_jbeil_path)
        
        print(f"   Original columns: {list(gdf.columns)}")
        
        # Check for Class_01 or similar
        if 'Class_01' in gdf.columns:
            gdf['class'] = gdf['Class_01']
            print(f"   Found 'Class_01' column, renamed to 'class'")
        elif 'class_01' in gdf.columns:
            gdf['class'] = gdf['class_01']
            print(f"   Found 'class_01' column, renamed to 'class'")
        else:
            # Check all columns for class-like names
            class_col = None
            for col in gdf.columns:
                if 'class' in col.lower():
                    class_col = col
                    break
            
            if class_col:
                gdf['class'] = gdf[class_col]
                print(f"   Found '{class_col}' column, renamed to 'class'")
            else:
                # Assume all are solar panels (Class=1)
                gdf['class'] = 1
                print(f"   No class column found, assuming all are solar panels (Class=1)")
        
        # Ensure numeric
        if gdf['class'].dtype == 'object':
            gdf['class'] = pd.to_numeric(gdf['class'], errors='coerce').fillna(1)
        
        gdf['class'] = gdf['class'].astype(np.int32)
        
        print(f"   Class values: {gdf['class'].unique()}")
        print(f"   Solar panels (Class=1): {(gdf['class'] == 1).sum()}")
        print(f"   Non-solar (Class≠1): {(gdf['class'] != 1).sum()}")
        
        # Save fixed version
        gdf.to_file(bint_jbeil_path)
        print(f"   ✅ Fixed and saved")
    else:
        print(f"   ⚠️  File not found")
    
    # Issue 3: Fix Ezzrareyye.shp
    print("\n3️⃣  Fixing Ezzrareyye.shp")
    print("-"*80)
    
    ezzrareyye_path = shapefile_dir / "Ezzrareyye.shp"
    if ezzrareyye_path.exists():
        if backup:
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                src = shapefile_dir / f"Ezzrareyye{ext}"
                if src.exists():
                    shutil.copy(src, backup_dir / f"Ezzrareyye{ext}")
        
        gdf = gpd.read_file(ezzrareyye_path)
        
        print(f"   Original columns: {list(gdf.columns)}")
        
        # Check for Class_01 or similar
        if 'Class_01' in gdf.columns:
            gdf['class'] = gdf['Class_01']
            print(f"   Found 'Class_01' column, renamed to 'class'")
        elif 'class_01' in gdf.columns:
            gdf['class'] = gdf['class_01']
            print(f"   Found 'class_01' column, renamed to 'class'")
        else:
            # Check all columns for class-like names
            class_col = None
            for col in gdf.columns:
                if 'class' in col.lower():
                    class_col = col
                    break
            
            if class_col:
                gdf['class'] = gdf[class_col]
                print(f"   Found '{class_col}' column, renamed to 'class'")
            else:
                # Assume all are solar panels (Class=1)
                gdf['class'] = 1
                print(f"   No class column found, assuming all are solar panels (Class=1)")
        
        # Ensure numeric
        if gdf['class'].dtype == 'object':
            gdf['class'] = pd.to_numeric(gdf['class'], errors='coerce').fillna(1)
        
        gdf['class'] = gdf['class'].astype(np.int32)
        
        print(f"   Class values: {gdf['class'].unique()}")
        print(f"   Solar panels (Class=1): {(gdf['class'] == 1).sum()}")
        print(f"   Non-solar (Class≠1): {(gdf['class'] != 1).sum()}")
        
        # Save fixed version
        gdf.to_file(ezzrareyye_path)
        print(f"   ✅ Fixed and saved")
    else:
        print(f"   ⚠️  File not found")
    
    # Issue 4: Very large features - just report, don't fix
    print("\n4️⃣  Very Large Features (>10,000 m²)")
    print("-"*80)
    print("   These are likely solar farms (ground-mounted arrays)")
    print("   Keeping them as valid training samples")
    print("   Files with large features:")
    print("      • BarEliasSP.shp: 10 large features")
    print("      • BatrounSP.shp: 1 large feature")
    print("      • BeirutSP.shp: 1 large feature")
    print("      • ChmistarSP.shp: 1 large feature")
    print("      • Rayak_SP.shp: 1 large feature")
    print("   ✅ No action needed")
    
    print("\n" + "="*80)
    print("✅ ALL ISSUES FIXED!")
    print("="*80)
    
    # Verify fixes
    print("\n📊 Re-counting totals after fixes...")
    
    total_solar = 0
    total_non_solar = 0
    total_files = 0
    
    for shp_path in shapefile_dir.glob("*.shp"):
        if 'Solar_' in shp_path.stem or shp_path.parent.name == 'backup_original':
            continue
        
        try:
            gdf = gpd.read_file(shp_path)
            
            # Find class column (case insensitive)
            class_col = None
            for col in gdf.columns:
                if col.lower() == 'class':
                    class_col = col
                    break
            
            if class_col:
                total_files += 1
                
                # Convert to numeric if needed
                if gdf[class_col].dtype == 'object':
                    gdf[class_col] = pd.to_numeric(gdf[class_col], errors='coerce').fillna(1)
                
                solar = (gdf[class_col] == 1).sum()
                non_solar = (gdf[class_col] != 1).sum()
                
                total_solar += solar
                total_non_solar += non_solar
                
        except Exception as e:
            print(f"   ⚠️  Error reading {shp_path.name}: {e}")
    
    print(f"\n   Total shapefiles: {total_files}")
    print(f"   Total solar panels (Class=1): {total_solar:,}")
    print(f"   Total non-solar (Class≠1): {total_non_solar:,}")
    print(f"   Solar ratio: {total_solar/(total_solar+total_non_solar)*100:.1f}%")
    
    print("\n" + "="*80)
    print("🎯 READY FOR TRAINING DATA PREPARATION!")
    print("="*80)
    print("\nNext step: Run analyze_solar_with_class.py")


def main():
    """
    Run fixes
    """
    SHAPEFILE_DIR = r"D:\Images\shapes_new"
    
    print("\n⚠️  This script will modify shapefiles to fix issues.")
    print("   Original files will be backed up to 'backup_original' folder.")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    fix_data_issues(SHAPEFILE_DIR, backup=True)


if __name__ == "__main__":
    import pandas as pd
    main()