import os
import re
import json
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
import warnings

# ==========================
# CONFIGURATION
# ==========================
BASE_URL = r"https://online-service2.nuernberg.de/geoinf/ia_bezirksatlas/"
OUTPUT_FOLDER = os.path.join("data", "nuremberg_stats")
BEZIRKE_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "bezirke_stats")
TEMP_DIR = os.path.join(OUTPUT_FOLDER, "temp")
PLOTS_FOLDER = os.path.join(BEZIRKE_OUTPUT_FOLDER, "plots")
HISTORY_STATS_CSV = os.path.join(BEZIRKE_OUTPUT_FOLDER, "processed_nuremberg_data.csv")
FUTURE_STATS_CSV = os.path.join(BEZIRKE_OUTPUT_FOLDER, "forecasted_values.csv")

FILES = [
    "data.js",
    "_nbg_Bezirk_bew_mp.shp1.js",
    "contextualLayer1.js",
    "contextualLayer2.js",
    "contextualLayer3.js",
    "contextualLayer4.js",
    "contextualLayer5.js",
]

for i in range(10):
    FILES.append(f"Statistische_Bezirke-t{i}.js")


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


def __extract_json_from_js(js_path: str) -> dict:
    """Extracts JSON data from a JS file."""
    with open(js_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
    if not match:
        print(f"❌ No JSON found in {js_path}")
        return None

    data = json.loads(match.group(1))
    return data


def __create_dataset(data_json_path: str, indicator_folder: str, output_csv_path: str) -> pd.DataFrame:
    """Creates a dataset from JS files and saves it as CSV."""
    data_json = __extract_json_from_js(data_json_path)
    if not data_json:
        return None

    bezirke = data_json["geographies"][0]["features"]
    bezirke_sorted = sorted(bezirke, key=lambda b: int(b["id"].split()[0]))
    df = pd.DataFrame({
        "id": [b["id"] for b in bezirke_sorted],
        "name": [b["name"] for b in bezirke_sorted]
    })

    new_cols = {}
    for i in range(10):
        file_path = os.path.join(indicator_folder, f"Statistische_Bezirke-t{i}.js")
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}, skipping...")
            continue
        data = __extract_json_from_js(file_path)
        for indicator in data.get("indicators", []):
            col_name = f"{indicator['name'].lower().replace(' ','_')}_{indicator.get('date','')}"
            values = indicator.get("values", [])

            cleaned_values = []
            for v in values:
                if v is None:
                    cleaned_values.append(pd.NA)
                else:
                    try:
                        v_str = str(v).replace(" ", "")
                        cleaned_values.append(float(v_str))
                    except ValueError:
                        cleaned_values.append(pd.NA)

            if len(cleaned_values) != len(df):
                print(f"⚠️ Length mismatch: {col_name} ({len(cleaned_values)} vs {len(df)})")
                continue

            new_cols[col_name] = cleaned_values

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"✅ Dataset saved: {output_csv_path}")
    return df


def __forecast_with_holt(actual_values: pd.Series, forecast_years: int = 5, smoothing_level: float = 0.5, smoothing_trend: float = 0.2) -> pd.Series:
    """Generates forecasts using the Holt-Winters method."""
    actual_values = pd.to_numeric(actual_values, errors='coerce')
    actual_values = actual_values.dropna()

    if actual_values.empty:
        raise ValueError("Data contains no valid numeric values after conversion!")

    actual_values_series = pd.Series(actual_values.values, index=range(2009, 2009 + len(actual_values)))
    model = Holt(actual_values_series)
    fitted_model = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, optimized=False)
    forecast = fitted_model.forecast(forecast_years)
    return forecast


def __group_columns_by_type(df: pd.DataFrame) -> dict:
    """Groups DataFrame columns by type."""
    column_groups = {}
    for col in df.columns[2:]:
        col_type = "_".join(col.split("_")[:-1])
        if col_type not in column_groups:
            column_groups[col_type] = []
        column_groups[col_type].append(col)
    return column_groups


def __plot_and_save_predictions(df: pd.DataFrame, forecast_years: int = 5, smoothing_level: float = 0.5, smoothing_trend: float = 0.2, plot_predictions=False) -> None:
    """Generates forecasts and saves them as CSV or plots."""
    if plot_predictions == True:
        os.makedirs(PLOTS_FOLDER, exist_ok=True)

    column_groups = __group_columns_by_type(df)
    forecast_data = []

    for col_type, cols in column_groups.items():
        if len(cols) == 0:
            continue

        for idx, row in df.iterrows():
            actual_values = row[cols].dropna()
            if len(actual_values) == 0:
                continue
            
            warnings.simplefilter("ignore", category=RuntimeWarning)
            future_predictions = __forecast_with_holt(actual_values, forecast_years=forecast_years, smoothing_level=smoothing_level, smoothing_trend=smoothing_trend)

            forecast_row = {"id": row["id"], "name": row["name"]}

            for i, prediction in enumerate(future_predictions):
                year = 2023 + i
                forecast_row[f"{col_type}_{year}"] = prediction

            forecast_data.append(forecast_row)

            if plot_predictions == True:
                plt.figure(figsize=(10, 6))
                actual_years = list(range(2009, 2009 + len(actual_values)))
                future_years = list(range(actual_years[-1] + 1, actual_years[-1] + 1 + forecast_years))

                plt.plot(actual_years, actual_values, label="Actual", color="blue", marker='o')
                plt.plot(future_years, future_predictions, label="Forecast", linestyle="--", color="red", marker='x')
                plt.xlabel("Year")
                plt.ylabel(col_type)
                plt.title(f"District {row['name']}\n{col_type}")
                plt.legend()
                plt.xticks(list(range(2009, 2029)), rotation=45)

                safe_col_name = col_type.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")\
                                        .replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_")\
                                        .replace("|", "_")
                plot_filename = f"{safe_col_name}_prediction_{row['name']}.png"
                plot_path = os.path.join(PLOTS_FOLDER, plot_filename)

                plt.savefig(plot_path)
                plt.close()

    grouped_data = {}
    for forecast_row in forecast_data:
        id_name_key = (forecast_row["id"], forecast_row["name"])
        if id_name_key not in grouped_data:
            grouped_data[id_name_key] = {"id": forecast_row["id"], "name": forecast_row["name"]}
        for key, value in forecast_row.items():
            if key not in ["id", "name"]:
                grouped_data[id_name_key][key] = value

    final_forecast_data = list(grouped_data.values())
    forecast_df = pd.DataFrame(final_forecast_data)
    forecast_df = forecast_df[["id", "name"] + sorted([col for col in forecast_df.columns if col not in ["id", "name"]])]

    forecast_df.to_csv(FUTURE_STATS_CSV, index=False)


def initialize_stats_pipeline(forecast_years=5) -> None:
    """Initializes the Nuremberg statistics data pipeline."""
    os.makedirs(BEZIRKE_OUTPUT_FOLDER, exist_ok=True)

    for file in FILES:
        url = BASE_URL + file
        output_path = os.path.join(BEZIRKE_OUTPUT_FOLDER, file)
        if __download_file(url, output_path) and file.endswith(".js"):
            data_json = __extract_json_from_js(output_path)
            if data_json is not None:
                print(f"✅ JSON from {file} can be extracted.")

    history_output_csv_path = HISTORY_STATS_CSV
    df = __create_dataset(
        data_json_path=os.path.join(BEZIRKE_OUTPUT_FOLDER, "data.js"),
        indicator_folder=BEZIRKE_OUTPUT_FOLDER,
        output_csv_path=history_output_csv_path
    )

    __plot_and_save_predictions(df, forecast_years=forecast_years, plot_predictions=False)

    print("✅ Pipeline initialization complete.")