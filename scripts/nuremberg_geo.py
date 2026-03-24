import geojson
from shapely.geometry import Point, shape, MultiPolygon
from shapely.ops import unary_union
from typing import Optional, Dict
import geopandas as gpd
from geopandas import GeoDataFrame
import os

from nuremberg_geo_stats_pipeline import GEOJSON_NUREMBERG_FILEPATH
from nuremberg_geo_stats_pipeline import GEOJSONS_OUTPUT_FOLDER


def __load_geojson_from_file(file_path: str) -> dict:
    """Loads GeoJSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return geojson.load(f)


def __find_district(lat: float, lon: float, data: dict) -> Optional[str]:
    """Determines the district containing the given coordinates."""
    point = Point(lon, lat)
    
    for feature in data['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            return f"{feature['properties']['KRG_DISS']} {feature['properties']['KRG_BEZ']}"
    
    return None


def get_district_str(lat: float, lon: float) -> str:
    """Returns the district for given coordinates as a string."""
    data = __load_geojson_from_file(GEOJSON_NUREMBERG_FILEPATH)
    district_info = __find_district(lat, lon, data)
    
    if district_info:
        return district_info
    else:
        return "No district found"
    

def get_nuremberg_outer_contour() -> Dict:
    """Generates the combined outer boundary of Nuremberg."""
    data = __load_geojson_from_file(GEOJSON_NUREMBERG_FILEPATH)
    polygons = []
    
    for feature in data['features']:
        polygon = shape(feature['geometry'])
        polygons.append(polygon)
    
    combined_polygon = unary_union(polygons)
    
    if isinstance(combined_polygon, MultiPolygon):
        combined_polygon = combined_polygon.geoms[0]
    
    combined_geojson = geojson.Feature(
        geometry=combined_polygon.__geo_interface__,
        properties={"description": "Combined outer boundary of Nuremberg"}
    )
    
    return geojson.FeatureCollection([combined_geojson])


def find_nearest_points(geojson_file, lat, lon, n=10) -> GeoDataFrame:
    """Finds the n nearest points to a given coordinate in a GeoJSON layer."""
    gdf = gpd.read_file(geojson_file)

    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=25832)

    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    point_gdf = point_gdf.to_crs(gdf.crs)

    gdf['distance'] = gdf.geometry.distance(point_gdf.geometry.iloc[0])
    nearest_points = gdf.nsmallest(n, 'distance')

    return nearest_points


def find_types(geojson_file, key="type") -> list[str]:
    """Returns all distinct types present in a GeoJSON file."""
    gdf = gpd.read_file(geojson_file)
    feature_classes = gdf[key].unique().tolist()
    return feature_classes


def find_points_within_radius(geojson_file, lat, lon, radius=100) -> GeoDataFrame:
    """Returns all points within a given radius from a coordinate."""
    gdf = gpd.read_file(geojson_file)

    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=25832)

    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    point_gdf = point_gdf.to_crs(gdf.crs)

    gdf['distance'] = gdf.geometry.distance(point_gdf.geometry.iloc[0])
    points_within_radius = gdf[gdf['distance'] <= radius]

    return points_within_radius


def get_available_geojsons() -> list[str]:
    """Searches a directory recursively and returns all GeoJSON file paths."""
    ignore_list = [GEOJSON_NUREMBERG_FILEPATH]
    geojson_files = []

    for root, dirs, files in os.walk(GEOJSONS_OUTPUT_FOLDER):
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in ignore_list]

        for file in files:
            if file.endswith('.geojson') and os.path.join(root, file) not in ignore_list:
                geojson_files.append(os.path.join(root, file))

    return geojson_files