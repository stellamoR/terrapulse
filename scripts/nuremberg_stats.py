import pandas as pd

from nuremberg_bezirke_stats_pipeline import HISTORY_STATS_CSV, FUTURE_STATS_CSV


# ==========================
# QUERY METHODS (GET HISTORICAL & FORECASTED VALUES)
# ==========================
def get_historical_value(year: int, district_name: str, categories: list = None) -> dict:
    """Fetches the historical values for a given year and district."""
    data_df = pd.read_csv(HISTORY_STATS_CSV)
    return __query_df(data_df, year, district_name, categories)


def get_forecasted_values(year: int, district_name: str, categories: list = None) -> dict:
    """Fetches the forecasted values for a given year and district."""
    data_df = pd.read_csv(FUTURE_STATS_CSV)
    return __query_df(data_df, year, district_name, categories)


def __query_df(data_df: pd.DataFrame, year: int, district_name: str, categories: list = None):
    """Queries the dataframe for the specified district and year."""
    district_data = data_df[data_df["name"] == district_name]

    if district_data.empty:
        raise ValueError(f"❌ District '{district_name}' not found!")

    if categories is None:
        result = {}
        for _, row in district_data.iterrows():
            for col in row.index[2:]:
                if str(year) in col:
                    result[col] = row[col]
        return result

    result = {}
    for category in categories:
        category_column = f"{category}_{year}"
        if category_column not in district_data.columns:
            print(f"⚠️ Category '{category}' for district '{district_name}' not found.")
            continue
        result[category] = district_data.iloc[0][category_column]

    return result


def __get_available_categories_and_years(data_csv_path: str) -> tuple[list[str], list[int]]:
    """Retrieves available categories and years from the CSV file."""
    data_df = pd.read_csv(data_csv_path)
    
    # Get the columns after the first two columns (id and name)
    column_names = data_df.columns[2:]
    
    # Separate categories and years
    categories = set()
    years = set()

    for col in column_names:
        # Extract the category and year from the column name
        parts = col.split('_')
        
        # Check if the column name has the expected structure (e.g., "category_year")
        if len(parts) >= 2:
            category = '_'.join(parts[:-1])  # Everything except the last part is the category
            year = parts[-1]  # The last part is the year
            
            categories.add(category)
            years.add(year)

    # Sort the years numerically
    years = sorted(list(years), key=int)
    
    # Return the categories and years as a tuple
    return sorted(list(categories)), years


# ==========================
# GET HISTORICAL CATEGORIES AND YEARS
# ==========================
def get_historical_categories_and_years() -> tuple[list[str], list[int]]:
    """Retrieves available categories and years for historical data."""
    return __get_available_categories_and_years(HISTORY_STATS_CSV)


# ==========================
# GET FORECASTED CATEGORIES AND YEARS
# ==========================
def get_forecasted_categories_and_years() -> tuple[list[str], list[int]]:
    """Retrieves available categories and years for forecasted data."""
    return __get_available_categories_and_years(FUTURE_STATS_CSV)