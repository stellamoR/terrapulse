#!/usr/bin/env python3
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path for config import
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from config import DATA_DIR, CRS_UTM

def rasterize_nuremberg_stats():
    # Paths
    stats_path = DATA_DIR / 'nuremberg_stats' / 'bezirke_stats' / 'processed_nuremberg_data.csv'
    geo_path = DATA_DIR / 'nuremberg_stats' / 'geojsons_stats' / 'nuremberg_stat_bezirke_wgs84.geojson'
    s2_ref_path = DATA_DIR / 'sentinel2' / 's2_median_2020.tif'
    out_path = DATA_DIR / 'nuremberg_stats' / 'rasterized_stats.tif'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load reference metadata
    with rasterio.open(s2_ref_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        shape = src.shape

    # Load data
    gdf = gpd.read_file(geo_path).to_crs(CRS_UTM)
    df = pd.read_csv(stats_path)

    # Select columns
    col_map = {
        'bevölkerung_mit_hauptwohnung_2020': 'pop_2020',
        'wohnungen_insg._2020': 'housing_2020',
        'nutzfläche_insg._(in_100\xa0m2)_2020': 'commercial_2020',
        'pkw_je_1_000_einwohner_2020': 'cars_2020'
    }
    
    # Check if they exist (handling the non-breaking space \xa0)
    available_cols = df.columns.tolist()
    final_cols = {}
    for k, v in col_map.items():
        if k in available_cols:
            final_cols[k] = v
        else:
            # Try to find partial match
            base = k.replace('\xa0', ' ')
            matches = [c for c in available_cols if base in c.replace('\xa0', ' ')]
            if matches:
                final_cols[matches[0]] = v
            else:
                print(f"Warning: Column {k} not found.")

    # Merge
    gdf['join_id'] = gdf['BEZ_ID'].astype(int)
    df['join_id'] = df['id'].astype(int)
    merged = gdf.merge(df[['join_id'] + list(final_cols.keys())], on='join_id', how='left')
    
    # Fill NAs (districts without data) with 0 or local mean? Let's use 0 for now as it's safe for population/housing
    merged = merged.fillna(0)

    # Update metadata for multi-band output
    meta.update({
        'count': len(final_cols),
        'dtype': 'float32',
        'nodata': -1
    })

    with rasterio.open(out_path, 'w', **meta) as dst:
        for i, (orig_col, new_name) in enumerate(final_cols.items(), 1):
            print(f"Rasterizing {new_name}...")
            # Create list of (geometry, value) pairs
            shapes = ((geom, value) for geom, value in zip(merged.geometry, merged[orig_col]))
            
            # Rasterize
            burned = rasterize(
                shapes=shapes,
                out_shape=shape,
                transform=transform,
                fill=-1,
                all_touched=True,
                dtype='float32'
            )
            dst.write(burned, i)
            dst.set_band_description(i, new_name)

    print(f"✓ Saved {out_path} with {len(final_cols)} bands: {list(final_cols.values())}")

if __name__ == "__main__":
    rasterize_nuremberg_stats()
