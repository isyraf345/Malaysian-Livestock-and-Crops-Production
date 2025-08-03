import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import math

warnings.filterwarnings('ignore')

### --- Load Data Once --- ###
file_path = 'C:/Users/USER/Documents/Data Science degree/sem 4/Data Analytics/UKM Data Challenge 4.0/combined_excel.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None)

commodities = {
    "beef": "Local Production of  Beef (Cattle) - Quantity (Tonne)",
    "goat": "Local Production of Mutton (Goat) - Quantity (Tonne)",
    "sheep": "Local Production of Mutton (Sheep) - Quantity (Tonne)",
    "swine": "Local Production of Pork - Quantity (Tonne)",
    "chicken": "Local Production of Chicken Meat - Quantity (Tonne)",
    "duck": "Local Production of Duck Meat - Quantity (Tonne)",
    "paddy": "Paddy Production (Tonne)",
    "palm oil": "Production of Fresh Fruit Bunches (Tonne)",
    "rubber": "Production (Tonne '000 )",
    "egg": "Local Production ('000 Tonnes)",
    "milk": "Production of Fresh Milk - Quantity (Million Litres)"
}

### --- KMeans Clustering Function --- ###
def show_kmeans_plot():
    feature_list = []
    for commodity, column_name in commodities.items():
        try:
            temp_df = sheets[commodity][['Year', column_name]].dropna()
            temp_df = temp_df.sort_values('Year')
            last_year = 2019 if commodity != 'rubber' else 2020
            temp_df = temp_df[temp_df['Year'] <= last_year]
            y = temp_df[column_name].values
            growth_rates = (y[1:] - y[:-1]) / y[:-1] * 100
            avg_growth_rate = np.mean(growth_rates)
            volatility = np.std(y)
            feature_list.append({
                'Commodity': commodity,
                'Avg_Production_Growth': avg_growth_rate,
                'Production_Volatility': volatility
            })
        except:
            pass

    features_df = pd.DataFrame(feature_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df[['Avg_Production_Growth', 'Production_Volatility']])

    kmeans_final = KMeans(n_clusters=3, random_state=42)
    features_df['Cluster'] = kmeans_final.fit_predict(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_df['Avg_Production_Growth'], features_df['Production_Volatility'], 
                           c=features_df['Cluster'], cmap='viridis', s=100)
    for i, row in features_df.iterrows():
        plt.annotate(row['Commodity'], (row['Avg_Production_Growth'], row['Production_Volatility']),
                     textcoords="offset points", xytext=(5, 5), ha='left')
    plt.xlabel('Average Production Growth Rate (%)')
    plt.ylabel('Production Volatility')
    plt.title('Commodity Clustering (KMeans)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

### --- ARIMA Forecasting Function --- ###
def get_unit(commodity):
    if commodity == "milk":
        return "Million Litres"
    elif commodity in ["egg", "rubber"]:
        return "Thousand Tonnes"
    else:
        return "Tonnes"

def forecast_arima(selected):
    if 'egg' in selected:
        commodity = 'egg'
        egg_type = 'CHICKEN EGG' if 'chicken' in selected else 'DUCK EGG'
    else:
        commodity = selected
        egg_type = None


    prod_col = commodities[commodity]

    temp_df = sheets[commodity]
    if commodity == 'egg':
        temp_df = temp_df[temp_df['Type'].str.upper() == egg_type.upper()]

    temp_df = temp_df[['Year', prod_col]].dropna()
    temp_df['Year'] = pd.to_numeric(temp_df['Year'], errors='coerce')
    temp_df = temp_df.dropna().sort_values('Year')
    temp_df = temp_df.rename(columns={prod_col: 'y'})

    if len(temp_df) < 3:
        return "Not enough data to forecast."

    y = temp_df['y'].values
    years = temp_df['Year'].values

    model = ARIMA(y, order=(1, 1, 1))
    model_fit = model.fit()

    predictions = model_fit.predict(start=0, end=len(y)-1)
    forecast_years = np.arange(2020, 2031)
    forecast = model_fit.forecast(steps=len(forecast_years))

    mae = mean_absolute_error(y, predictions)
    rmse = math.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    plt.figure(figsize=(12, 6))
    plt.plot(years, y, marker='o', label='Historical', color='blue')
    plt.plot(forecast_years, forecast, linestyle='--', label='Forecast', color='red')

    metric_text = f"R²={r2:.3f}\nMAE={mae:.2f}\nRMSE={rmse:.2f}"
    plt.text(0.02, 0.98, metric_text, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('Year')
    label = f"Production ({get_unit(commodity)})"
    if egg_type:
        label += f" - {egg_type.title()}"
    plt.ylabel(label)
    title = f'Local Production Forecast for {commodity.title()}'
    if egg_type:
        title += f' ({egg_type.title()})'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return plt.gcf()

### --- Gradio UI --- ###
# Power BI Tab

# Your Power BI report URL
power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiYmNlOGFlNjMtOGJkNi00MTFiLWFhNzAtNDhmOGQwNDc3MzJiIiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D"
# Function to generate iframe HTML
import random
import time

# Function to generate iframe HTML with cache-busting
def embed_power_bi():
    # Add a dynamic timestamp to the URL to avoid caching
    timestamp = int(time.time())  # Get current timestamp
    power_bi_url_with_cache_busting = f"https://app.powerbi.com/view?r=eyJrIjoiYmNlOGFlNjMtOGJkNi00MTFiLWFhNzAtNDhmOGQwNDc3MzJiIiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D&t={timestamp}"  # Append timestamp
    
    # Return the HTML iframe with the updated URL
    return f'<iframe title="Malaysian Livestock and Crops Production" width="600" height="373.5" src="{power_bi_url_with_cache_busting}" frameborder="0" allowFullScreen="true"></iframe>'

# Create the Power BI iframe component directly
power_bi_component = gr.HTML(value=embed_power_bi())

# Create a Blocks for Power BI Dashboard
with gr.Blocks() as power_bi_tab:
    gr.Markdown("## Power BI Dashboard")
    gr.HTML(embed_power_bi())

# (Your ML model Interface, unchanged)
with gr.Blocks() as ml_model_interface:
    gr.Markdown("## Machine Learning Visualizations")
    with gr.Tab("KMeans Clustering"):
        btn_kmeans = gr.Button("Show KMeans Clustering")
        output_kmeans = gr.Plot()
        btn_kmeans.click(fn=show_kmeans_plot, inputs=None, outputs=output_kmeans)

    with gr.Tab("ARIMA Forecasting"):
        dropdown = gr.Dropdown(
            choices=[
                "beef", "goat", "sheep", "swine", "chicken", "duck",
                "paddy", "palm oil", "rubber", "chicken egg", "duck egg", "milk"
            ],
            label="Select Commodity for Forecasting"
        )
        btn_forecast = gr.Button("Forecast")
        output_forecast = gr.Plot()
        btn_forecast.click(fn=forecast_arima, inputs=dropdown, outputs=output_forecast)


# Tabs setup
tabs = gr.TabbedInterface(
    [power_bi_tab, ml_model_interface],
    ["Power BI Dashboard", "ML Model"]
)

# Launch
tabs.launch()
