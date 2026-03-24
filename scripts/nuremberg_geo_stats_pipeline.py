import os
import requests
import zipfile
import shutil
import geopandas as gpd

OUTPUT_FOLDER = os.path.join("data", "nuremberg_stats")
GEOJSONS_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "geojsons_stats")
ZIP_URL_GEOJSON_NUREMBERG = "https://www.nuernberg.de/imperia/md/statistik/dokumente/karten/geometrie_nue_stat_bezirke.zip"
ZIP_PATH_GEOJSON_NUREMBERG = os.path.join(GEOJSONS_OUTPUT_FOLDER, "geometrie_bezirke.zip")
GEOJSON_NUREMBERG_FILEPATH = os.path.join(GEOJSONS_OUTPUT_FOLDER, "nuremberg_stat_bezirke_wgs84.geojson")
ZIP_URL_GEOJSONS_OSM = 'https://download.geofabrik.de/europe/germany/bayern/mittelfranken-latest-free.shp.zip'
GEJSONS_OSM_EXTRACTED_FILEPATH = os.path.join(GEOJSONS_OUTPUT_FOLDER, "extracted")


def __download_file(url: str, output_path: str) -> bool:
    """Downloads a file if it does not exist."""
    if os.path.exists(output_path):
        print(f"✅ Already exists: {output_path}")
        return False
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False


def __process_shapefile(zip_path: str, output_folder: str) -> None:
    """Converts a shapefile ZIP to GeoJSON."""
    temp_dir = os.path.join(output_folder, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print("✅ ZIP extracted")

        shp_file = None
        for f in os.listdir(temp_dir):
            if f.lower().endswith(".shp"):
                shp_file = os.path.join(temp_dir, f)
                break
        if shp_file is None:
            raise FileNotFoundError("❌ No shapefile found")

        gdf = gpd.read_file(shp_file)
        gdf_wgs = gdf.to_crs(epsg=4326)
        geojson_path = GEOJSON_NUREMBERG_FILEPATH
        gdf_wgs.to_file(geojson_path, driver="GeoJSON")
        print(f"✅ GeoJSON saved: {geojson_path}")
    finally:
        shutil.rmtree(temp_dir)
        print(f"🗑️ TEMP folder deleted: {temp_dir}")


def download_and_extract(url, download_folder, extract_folder):
    """Downloads and extracts a ZIP file."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    zip_filename = os.path.join(download_folder, os.path.basename(url))
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    
    with open(zip_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✅ Download completed! The ZIP file was saved to '{zip_filename}'.")
    
    print(f"Extracting {zip_filename} to {extract_folder}...")
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    print(f"✅ Extraction completed! The files were saved to '{extract_folder}'.")


def filter_and_save_shapefiles(input_folder, geojson_file, output_folder):
    """Clips shapefiles using a GeoJSON mask and saves them."""
    os.makedirs(output_folder, exist_ok=True)
    gdf_geojson = gpd.read_file(geojson_file)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.shp'):
            shapefile_path = os.path.join(input_folder, filename)
            gdf_dbf = gpd.read_file(shapefile_path)
            gdf_filtered = gpd.overlay(gdf_dbf, gdf_geojson, how='intersection')
            output_filename = filename.replace('.shp', '_nuremberg.geojson')
            output_filepath = os.path.join(output_folder, output_filename)
            gdf_filtered.to_file(output_filepath, driver='GeoJSON')
            print(f"✅ Clipping completed! {output_filename} was saved to '{output_folder}'.")


def initialize_geo_pipeline():
    """Initializes the geo data pipeline."""
    os.makedirs(GEOJSONS_OUTPUT_FOLDER, exist_ok=True)
    
    if __download_file(ZIP_URL_GEOJSON_NUREMBERG, ZIP_PATH_GEOJSON_NUREMBERG):
        __process_shapefile(ZIP_PATH_GEOJSON_NUREMBERG, GEOJSONS_OUTPUT_FOLDER)
    
    download_and_extract(ZIP_URL_GEOJSONS_OSM, GEOJSONS_OUTPUT_FOLDER, GEJSONS_OSM_EXTRACTED_FILEPATH)
    filter_and_save_shapefiles(GEJSONS_OSM_EXTRACTED_FILEPATH, GEOJSON_NUREMBERG_FILEPATH, GEOJSONS_OUTPUT_FOLDER)