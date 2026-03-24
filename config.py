"""
config.py – Central configuration for the Nürnberg satellite land-use project.

This module provides a strictly defined source of truth for the entire pipeline.
It ensures spatial consistency (CRS, Bounding Box), temporal bounds (Years 2020/2021),
and thematic consistency (WorldCover classes).

Key Components:
  • AOI_BBOX: The study area in WGS84 coordinates [min_lon, min_lat, max_lon, max_lat].
  • CRS_UTM: The local projection (EPSG:25832) used for all area-based calculations.
  • WORLDCOVER_CLASSES: Dictionary mapping IDs to descriptive names and hex colors.
  • S2_BANDS: Definitions for Sentinel-2 spectral channels and composite years.
  • Paths: Standardized local directory structure for raw and processed data.

Integration:
  Used by data acquisition, feature extraction, and visualization scripts to 
  maintain identical data alignment and class legend handling.
"""

from pathlib import Path
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
WC_DIR = DATA_DIR / "worldcover"
S2_DIR = DATA_DIR / "sentinel2"
DEM_DIR = DATA_DIR / "dem"
FIG_DIR = PROJECT_DIR / "figures"

for d in [WC_DIR, S2_DIR, DEM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Area of Interest – Nürnberg
# ──────────────────────────────────────────────
# Bounding box [west, south, east, north] in EPSG:4326
AOI_BBOX = [10.9430, 49.3057, 11.3197, 49.5642]  # expanded to cover full dashboard anchor

# CRS
CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32632"  # UTM zone 32N – covers Nürnberg

# ──────────────────────────────────────────────
# ESA WorldCover 10 m – class legend
# ──────────────────────────────────────────────
# Official class IDs, names, and RGB colours
WORLDCOVER_CLASSES = {
    10: {"name": "Tree cover",              "color": "#006400"},
    20: {"name": "Shrubland",               "color": "#FFBB22"},
    30: {"name": "Grassland",               "color": "#FFFF4C"},
    40: {"name": "Cropland",                "color": "#F096FF"},
    50: {"name": "Built-up",                "color": "#FA0000"},
    60: {"name": "Bare / sparse vegetation","color": "#8E44AD"},
    70: {"name": "Snow and ice",            "color": "#F0F0F0"},
    80: {"name": "Permanent water bodies",  "color": "#0064C8"},
    90: {"name": "Herbaceous wetland",      "color": "#0096A0"},
    95: {"name": "Mangroves",               "color": "#00CF75"},
    100:{"name": "Moss and lichen",         "color": "#FAE6A0"},
}

# Sorted class IDs for consistent ordering
WC_CLASS_IDS = sorted(WORLDCOVER_CLASSES.keys())
WC_CLASS_NAMES = [WORLDCOVER_CLASSES[c]["name"] for c in WC_CLASS_IDS]
WC_CLASS_COLORS = [WORLDCOVER_CLASSES[c]["color"] for c in WC_CLASS_IDS]


def get_worldcover_cmap():
    """Return a (cmap, norm) pair for plotting WorldCover rasters."""
    # We map each class value to a colour.  Values outside the known
    # classes are mapped to transparent.
    from matplotlib.colors import hex2color

    boundaries = WC_CLASS_IDS + [WC_CLASS_IDS[-1] + 1]
    colours = [hex2color(c) for c in WC_CLASS_COLORS]
    cmap = ListedColormap(colours, name="worldcover")
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


# ──────────────────────────────────────────────
# Sentinel-2 Level-2A bands
# ──────────────────────────────────────────────
# Bands we download (10 m and 20 m native resolution; we resample 20 m → 10 m)
S2_BANDS = [
    "B02",  # Blue        490 nm   10 m
    "B03",  # Green       560 nm   10 m
    "B04",  # Red         665 nm   10 m
    "B05",  # Veg Red Edge 705 nm  20 m
    "B06",  # Veg Red Edge 740 nm  20 m
    "B07",  # Veg Red Edge 783 nm  20 m
    "B08",  # NIR         842 nm   10 m
    "B8A",  # NIR narrow  865 nm   20 m
    "B11",  # SWIR-1      1610 nm  20 m
    "B12",  # SWIR-2      2190 nm  20 m
]

S2_BAND_WAVELENGTHS = {
    "B02": 490, "B03": 560, "B04": 665,
    "B05": 705, "B06": 740, "B07": 783,
    "B08": 842, "B8A": 865, "B11": 1610, "B12": 2190,
}

# ──────────────────────────────────────────────
# Sentinel-2 time windows per year
# ──────────────────────────────────────────────
S2_YEARS = [2019, 2020, 2021]
S2_MONTHS = (6, 9)  # Jun – Sep (summer composite)
S2_MAX_CLOUD = 20   # max cloud cover %

# ──────────────────────────────────────────────
# WorldCover available years
# ──────────────────────────────────────────────
WC_YEARS = [2020, 2021]

# ──────────────────────────────────────────────
# Spectral index formulas (for reference / feature eng.)
# ──────────────────────────────────────────────
SPECTRAL_INDICES = {
    "NDVI":  "(B08 - B04) / (B08 + B04)",
    "NDBI":  "(B11 - B08) / (B11 + B08)",
    "NDWI":  "(B03 - B08) / (B03 + B08)",
    "NDMI":  "(B08 - B11) / (B08 + B11)",
    "EVI":   "2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)",
    "SAVI":  "1.5 * (B08 - B04) / (B08 + B04 + 0.5)",
    "BSI":   "((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))",
}
