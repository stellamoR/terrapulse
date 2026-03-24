from nuremberg_bezirke_stats_pipeline import initialize_stats_pipeline
from nuremberg_stats import get_historical_value, get_forecasted_values, get_forecasted_categories_and_years, get_historical_categories_and_years

from nuremberg_geo_stats_pipeline import initialize_geo_pipeline
from nuremberg_geo import get_nuremberg_outer_contour, get_district_str, get_available_geojsons, find_nearest_points, find_points_within_radius, find_types


if __name__ == "__main__":

    # -----------------------------
    # Initialize pipelines
    # -----------------------------
    forecast_years = 5
    initialize_stats_pipeline(forecast_years=forecast_years)
    initialize_geo_pipeline()

    # -----------------------------
    # Bezirke data example
    # -----------------------------
    lat, lon = 49.487117, 11.127343
    district = get_district_str(lat, lon)

    # Historical data
    hist_categories, hist_years = get_historical_categories_and_years()
    categories = hist_categories[:2]
    historical_data = get_historical_value(2023, district, categories)
    print("Historical data")
    print(historical_data)

    # Forecasted data
    future_categories, future_years = get_forecasted_categories_and_years()
    categories = future_categories[:2]
    future_data = get_forecasted_values(2023, district, categories)
    print("Future data")
    print(future_data)

    # -----------------------------
    # OSM/GeoJSON data example
    # -----------------------------
    outer_contour = get_nuremberg_outer_contour()
    print("Outer contour of Nuremberg")
    print(outer_contour)

    available_geojsons = get_available_geojsons()
    interested_filepath = available_geojsons[0]

    # Nearest points
    n = 10
    nearest_points = find_nearest_points(interested_filepath, lat, lon, n)
    print("Nearest points")
    print(nearest_points)

    # Geometry types
    geometry_types = find_types(interested_filepath)
    print("Geometry types")
    print(geometry_types)

    # Points within radius
    radius = 100
    points_within_radius = find_points_within_radius(interested_filepath, lat, lon, radius)
    print("Points within radius")
    print(points_within_radius)