"""
Shapefile Inspector
Shows the structure and sample data from all shapefiles
"""

import geopandas as gpd
from pathlib import Path
import pandas as pd


def inspect_shapefiles(shapefile_dir: str, max_files: int = None):
    """
    Inspect all shapefiles and show their structure
    """
    shapefile_dir = Path(shapefile_dir)
    shp_files = sorted(list(shapefile_dir.glob("*.shp")))
    
    if max_files:
        shp_files = shp_files[:max_files]
    
    print("="*80)
    print(f"SHAPEFILE INSPECTION - {len(shp_files)} files")
    print("="*80)
    
    for idx, shp_path in enumerate(shp_files, 1):
        print(f"\n{'='*80}")
        print(f"{idx}. {shp_path.name}")
        print(f"{'='*80}")
        
        try:
            gdf = gpd.read_file(shp_path)
            
            print(f"\n📊 BASIC INFO:")
            print(f"   Total Features: {len(gdf):,}")
            print(f"   CRS: {gdf.crs}")
            print(f"   Geometry Type: {gdf.geom_type.unique().tolist()}")
            
            print(f"\n📋 ALL COLUMNS ({len(gdf.columns)} total):")
            print(f"   {', '.join(gdf.columns.tolist())}")
            
            # Show first 5 rows with first 5 columns (excluding geometry for readability)
            cols_to_show = [col for col in gdf.columns if col != 'geometry'][:5]
            
            if cols_to_show:
                print(f"\n📝 SAMPLE DATA (first 5 rows, first 5 non-geometry columns):")
                print("-"*80)
                
                # Create a clean display
                sample_df = gdf[cols_to_show].head(5)
                
                # Format the output nicely
                for col_idx, col in enumerate(cols_to_show):
                    print(f"\n   Column {col_idx+1}: '{col}' (dtype: {sample_df[col].dtype})")
                    
                    # Show unique values if categorical or boolean
                    if sample_df[col].dtype in ['object', 'bool', 'category']:
                        unique_vals = sample_df[col].unique()
                        if len(unique_vals) <= 10:
                            print(f"   Unique values: {unique_vals.tolist()}")
                    
                    # Show min/max if numeric
                    elif sample_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        print(f"   Range: {gdf[col].min()} to {gdf[col].max()}")
                    
                    # Show first 5 values
                    print(f"   First 5 values: {sample_df[col].tolist()}")
                
                # Check for 'class' column specifically
                if 'class' in gdf.columns or 'Class' in gdf.columns:
                    class_col = 'class' if 'class' in gdf.columns else 'Class'
                    print(f"\n🎯 CLASS COLUMN FOUND: '{class_col}'")
                    print(f"   Value counts:")
                    value_counts = gdf[class_col].value_counts()
                    for val, count in value_counts.items():
                        print(f"      {val}: {count:,} features ({count/len(gdf)*100:.1f}%)")
                    
                    # Check if class==1 exists (solar panels)
                    if 1 in gdf[class_col].values:
                        solar_count = (gdf[class_col] == 1).sum()
                        print(f"\n   ✅ SOLAR PANELS (class=1): {solar_count:,}")
                        print(f"   ✅ NON-SOLAR (class!=1): {(len(gdf) - solar_count):,}")
                else:
                    print(f"\n   ⚠️  NO 'class' COLUMN FOUND")
            else:
                print(f"\n   ⚠️  No non-geometry columns found")
            
            # Show geometry bounds
            bounds = gdf.total_bounds
            print(f"\n🗺️  SPATIAL EXTENT:")
            print(f"   Bounds: [{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]")
            
            # Calculate area if projected CRS
            if gdf.crs and gdf.crs.is_projected:
                total_area = gdf.geometry.area.sum()
                mean_area = gdf.geometry.area.mean()
                print(f"   Total Area: {total_area:.2f} m²")
                print(f"   Mean Feature Area: {mean_area:.2f} m²")
            
        except Exception as e:
            print(f"   ❌ ERROR reading shapefile: {e}")
    
    print(f"\n{'='*80}")
    print("INSPECTION COMPLETE")
    print(f"{'='*80}")


def create_summary_table(shapefile_dir: str):
    """
    Create a summary table of all shapefiles
    """
    shapefile_dir = Path(shapefile_dir)
    shp_files = sorted(list(shapefile_dir.glob("*.shp")))
    
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE - ALL SHAPEFILES")
    print(f"{'='*80}\n")
    
    summary_data = []
    
    for shp_path in shp_files:
        try:
            gdf = gpd.read_file(shp_path)
            
            # Check for class column
            has_class = 'class' in gdf.columns or 'Class' in gdf.columns
            class_col = 'class' if 'class' in gdf.columns else 'Class' if 'Class' in gdf.columns else None
            
            if has_class and class_col:
                solar_count = (gdf[class_col] == 1).sum()
                non_solar_count = (gdf[class_col] != 1).sum()
            else:
                solar_count = 0
                non_solar_count = len(gdf)
            
            summary_data.append({
                'Shapefile': shp_path.name,
                'Total': len(gdf),
                'Solar (class=1)': solar_count,
                'Non-Solar': non_solar_count,
                'Has Class Column': '✅' if has_class else '❌',
                'CRS': str(gdf.crs)
            })
            
        except Exception as e:
            summary_data.append({
                'Shapefile': shp_path.name,
                'Total': 0,
                'Solar (class=1)': 0,
                'Non-Solar': 0,
                'Has Class Column': '❌',
                'CRS': 'ERROR'
            })
    
    # Create DataFrame and print
    summary_df = pd.DataFrame(summary_data)
    
    # Print as table
    print(summary_df.to_string(index=False))
    
    # Print totals
    print(f"\n{'='*80}")
    print("TOTALS:")
    print(f"{'='*80}")
    print(f"Total Shapefiles: {len(summary_df)}")
    print(f"Shapefiles with 'class' column: {(summary_df['Has Class Column'] == '✅').sum()}")
    print(f"Total Buildings: {summary_df['Total'].sum():,}")
    print(f"Total Solar Panels (class=1): {summary_df['Solar (class=1)'].sum():,}")
    print(f"Total Non-Solar Buildings: {summary_df['Non-Solar'].sum():,}")
    
    if summary_df['Solar (class=1)'].sum() > 0:
        solar_pct = summary_df['Solar (class=1)'].sum() / summary_df['Total'].sum() * 100
        print(f"Solar Panel Percentage: {solar_pct:.2f}%")


def main():
    """
    Main execution
    """
    SHAPEFILE_DIR = r"D:\Images\shapes_new"
    
    print("="*80)
    print("SHAPEFILE STRUCTURE INSPECTOR")
    print("="*80)
    print(f"\nDirectory: {SHAPEFILE_DIR}\n")
    
    # Ask user if they want to see all files or just a sample
    print("Options:")
    print("  1. Inspect ALL shapefiles (detailed)")
    print("  2. Inspect first 10 shapefiles (detailed)")
    print("  3. Show summary table only (quick)")
    print()
    
    try:
        choice = input("Enter choice (1-3) or press Enter for option 3: ").strip()
        
        if choice == '1':
            inspect_shapefiles(SHAPEFILE_DIR)
            create_summary_table(SHAPEFILE_DIR)
        elif choice == '2':
            inspect_shapefiles(SHAPEFILE_DIR, max_files=10)
            create_summary_table(SHAPEFILE_DIR)
        else:  # Default to 3
            create_summary_table(SHAPEFILE_DIR)
            
    except KeyboardInterrupt:
        print("\n\nInspection cancelled by user")


if __name__ == "__main__":
    main()