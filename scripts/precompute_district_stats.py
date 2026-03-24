#!/usr/bin/env python3
"""
Precompute per-district pixel class counts for the Nuremberg dashboard.

Reads the boundary GeoJSON and label/prediction .bin files, rasterizes each
district polygon onto the pixel grid, and writes a JSON file with per-district class counts.

Output format:
{
    "<district_id>": {
        "name": "Altstadt, St. Lorenz",
        "labels_2020": {"tree_cover": 1234, "grassland": 567, ...},
        "labels_2021": {"tree_cover": 1200, ...},
        "predictions_2021": {"tree_cover": 1180, ...},
        "total_pixels": 5000
    },
    ...
}
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DASHBOARD_DATA = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_dashboard"
BOUNDARY_PATH = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_boundary.geojson"

CLASS_ORDER = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]

def load_meta():
    with open(DASHBOARD_DATA / "nuremberg_dashboard_meta.json") as f:
        return json.load(f)

def load_boundary():
    with open(BOUNDARY_PATH) as f:
        return json.load(f)

def point_in_polygon(px, py, polygon_coords):
    """Ray casting algorithm for point-in-polygon test."""
    n = len(polygon_coords)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_coords[i]
        xj, yj = polygon_coords[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def rasterize_district(feature, meta, width, height):
    """Create a boolean mask for a district polygon on the raster grid."""
    bounds = meta["wgs84_bounds"]
    west, south, east, north = bounds

    mask = np.zeros((height, width), dtype=bool)

    geometry = feature["geometry"]
    if geometry["type"] == "Polygon":
        rings = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        rings = geometry["coordinates"]
    else:
        return mask

    for polygon in rings:
        outer_ring = polygon[0]  # Outer ring only

        # Get bounding box of polygon in pixel coords
        lngs = [p[0] for p in outer_ring]
        lats = [p[1] for p in outer_ring]
        min_lng, max_lng = min(lngs), max(lngs)
        min_lat, max_lat = min(lats), max(lats)

        # Convert to pixel coords
        px_min = max(0, int((min_lng - west) / (east - west) * width))
        px_max = min(width - 1, int((max_lng - west) / (east - west) * width))
        py_min = max(0, int((north - max_lat) / (north - south) * height))
        py_max = min(height - 1, int((north - min_lat) / (north - south) * height))

        for py in range(py_min, py_max + 1):
            lat = north - (py + 0.5) / height * (north - south)
            for px in range(px_min, px_max + 1):
                lng = west + (px + 0.5) / width * (east - west)
                if point_in_polygon(lng, lat, outer_ring):
                    mask[py, px] = True

    return mask

def count_classes(data, mask, width):
    """Count pixels of each class within the mask."""
    flat_mask = mask.flatten()
    masked_vals = data[flat_mask]
    counts = {}
    for i, cls_name in enumerate(CLASS_ORDER):
        counts[cls_name] = int((masked_vals == i).sum())
    return counts

def main():
    meta = load_meta()
    boundary = load_boundary()

    # Use resolution 5 (50m) for a good balance of speed and accuracy
    res = 5
    res_key = f"res{res}"
    dims = meta["resolutions"][res_key]
    width, height = dims["width"], dims["height"]

    print(f"Using resolution {res} ({width}×{height})")

    # Load label data
    datasets = {}
    for name, fname in [
        ("labels_2020", f"nuremberg_labels_2020_res{res}.bin"),
        ("labels_2021", f"nuremberg_labels_2021_res{res}.bin"),
        ("predictions_2021", f"nuremberg_pred_2021_res{res}.bin"),
        ("experimental_2021", f"experimental_pred_2021_res{res}.bin"),
    ]:
        fpath = DASHBOARD_DATA / fname
        if fpath.exists():
            datasets[name] = np.fromfile(fpath, dtype=np.uint8)
            print(f"  Loaded {name}: {len(datasets[name]):,} pixels")
        else:
            print(f"  Skipped {name}: {fname} not found")

    result = {}
    features = boundary["features"]
    print(f"\nProcessing {len(features)} districts...")

    for idx, feature in enumerate(features):
        props = feature["properties"]
        district_id = props.get("KRG_DISS", str(idx))
        district_name = props.get("KRG_BEZ", f"District {district_id}")

        mask = rasterize_district(feature, meta, width, height)
        total = int(mask.sum())

        if total == 0:
            continue

        entry = {
            "name": district_name,
            "id": district_id,
            "total_pixels": total,
        }

        for ds_name, data in datasets.items():
            entry[ds_name] = count_classes(data, mask, width)

        result[district_id] = entry
        print(f"  [{idx+1}/{len(features)}] {district_id} {district_name}: {total:,} pixels")

    # Write output
    out_path = DASHBOARD_DATA / "district_stats.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Wrote {out_path} ({len(result)} districts)")

if __name__ == "__main__":
    main()
