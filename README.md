# AI-Driven Fashion Demand Forecasting and Inventory Optimization

An end-to-end machine learning and deep learning system for **fashion demand forecasting and inventory optimization** using the **Visuelle 2.0** dataset.

The project compares statistical, machine learning, and deep learning forecasting approaches and converts demand forecasts into practical inventory decisions such as safety stock, reorder point, and recommended order quantity.

---

## Project Overview

Fashion retailers need to accurately forecast product demand to avoid:

- Overstocking
- Stockouts
- Excess inventory
- Missed sales opportunities
- Inefficient replenishment

This project develops an end-to-end forecasting and inventory optimization pipeline.

The system evaluates five forecasting approaches:

1. Naive Forecast
2. Simple Exponential Smoothing (SES)
3. SARIMA
4. XGBoost
5. LSTM

The best-performing model is then used as the basis for inventory planning.

---

## Key Results

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **XGBoost** | **0.010118** | **0.016374** | **50.93%** |
| LSTM | 0.013145 | 0.018761 | 54.37% |
| Naive | 0.013372 | 0.024471 | 78.35% |
| SES | 0.018457 | 0.025633 | 56.97% |
| SARIMA | 0.021230 | 0.033497 | 76.13% |

### Best Model

**XGBoost achieved the best overall performance**, obtaining the lowest MAE, RMSE, and MAPE among the five evaluated models.

This suggests that combining recent demand history with product-related features was more effective for this dataset than the tested purely sequential deep-learning and statistical approaches.

---

## Dataset

The project uses the **Visuelle 2.0** fashion retail dataset.

Visuelle 2.0 contains fashion product and retail information from Nuna Lie across six seasons:

- SS17
- AW17
- SS18
- AW18
- SS19
- AW19

The dataset includes information such as:

- Product identifiers
- Retail/store information
- Season
- Product category
- Color
- Fabric
- Release date
- Sales
- Restocking information
- Price and discount information
- Product images
- Weather information
- Google Trends information

The raw dataset is not included in this repository.

---

## Problem Formulation

For each product-store record, historical weekly sales are used to predict future demand.

The forecasting pipeline uses historical demand observations to generate a one-step-ahead demand forecast.

The normalized sales values provided by Visuelle 2.0 are converted back into approximate sales units using the dataset normalization scalar:

```text
actual_sales = normalized_sales × 53
```

## Forecasting Models
1. Naive Forecast

The Naive model uses the most recent observed demand as the forecast for the next period.

It provides a simple baseline against which more sophisticated models can be evaluated.

2. Simple Exponential Smoothing

Simple Exponential Smoothing assigns greater importance to recent observations while maintaining a smoothed estimate of demand.

It provides a stronger statistical baseline for relatively short demand sequences.

3. SARIMA

SARIMA was included as an additional classical time-series model.

Because individual product-store records contain relatively short historical sequences, a lightweight SARIMA configuration was used:

SARIMA(1,0,0) × (1,0,0,2)

Fallback models are used when the primary configuration cannot be fitted.

4. XGBoost

XGBoost is used as the primary machine learning forecasting model.

The implementation uses:

Product/store information
Season
Category
Color
Fabric
Release date features
Recent demand statistics
Mean demand
Standard deviation
Minimum demand
Maximum demand
Total demand
Recent 3-period mean
Recent 5-period mean
Demand trend

Categorical variables are encoded using one-hot encoding.

The model uses an XGBRegressor with gradient-boosted decision trees.

5. LSTM

An LSTM neural network is used to model temporal demand patterns.

The LSTM receives historical sales observations as sequential input and predicts the next demand value.

PyTorch is used for the implementation, with Apple Silicon MPS acceleration when available.

## Evaluation Metrics
Mean Absolute Error

MAE measures the average absolute difference between predicted and actual demand.

MAE = mean(|actual - predicted|)

Lower values indicate better performance.

Root Mean Squared Error

RMSE gives greater weight to larger forecasting errors.

RMSE = sqrt(mean((actual - predicted)^2))

Lower values indicate better performance.

Mean Absolute Percentage Error

MAPE measures forecasting error relative to actual demand.

MAPE = mean(|actual - predicted| / |actual|) × 100

Lower values indicate better performance.

Because fashion demand contains many zero or low-demand observations, MAPE should be interpreted carefully.

Inventory Optimization

The forecasting model is connected to an inventory optimization layer.

The inventory system calculates:

Demand Statistics

Historical demand is used to calculate:

Mean Demand
Demand Standard Deviation
Safety Stock

Safety stock is calculated using:

Safety Stock = Z × Demand Std × √Lead Time

A 95% service level uses:

Z = 1.645
Reorder Point

The reorder point is calculated as:

Reorder Point =
Forecast Demand × Lead Time + Safety Stock
Recommended Order Quantity

The recommended order quantity is:

Order Quantity =
max(Reorder Point - Current Stock, 0)

The system then produces either:

ORDER

or:

NO ORDER
Inventory Decision Pipeline
Product
   |
   v
Forecast Demand
   |
   v
Estimated Available Stock
   |
   v
Demand Variability
   |
   v
Safety Stock
   |
   v
Reorder Point
   |
   v
Recommended Order Quantity
   |
   v
ORDER / NO ORDER

The dashboard uses estimated available stock rather than claiming to represent live inventory.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring forecasting and inventory decisions.

The dashboard provides:

Model selection
Product/store records
Forecast demand
Actual demand
Estimated available stock
Safety stock
Reorder point
Recommended order quantity
ORDER / NO ORDER recommendations
Historical demand analysis
Forecast vs actual comparison
Historical restocking analysis
Model comparison

## Project Structure
fashion-demand-optimization/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── processed/
│   └── raw/
│
├── models/
│   ├── baseline/
│   │   └── baseline_model.py
│   ├── lstm/
│   │   └── lstm_model.py
│   ├── sarima/
│   │   └── sarima_model.py
│   └── xgboost/
│       └── xgboost_model.py
│
├── notebooks/
│
├── results/
│   ├── baseline_predictions.csv
│   ├── lstm_metrics.csv
│   ├── lstm_predictions.csv
│   ├── model_comparison.csv
│   ├── naive_metrics.csv
│   ├── naive_predictions.csv
│   ├── sarima_predictions.csv
│   ├── ses_metrics.csv
│   ├── ses_predictions.csv
│   ├── xgboost_metrics.csv
│   └── xgboost_predictions.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── forecasting.py
│   ├── inventory.py
│   ├── normalization.py
│   ├── preprocessing.py
│   ├── run_all.py
│   └── run_comparison.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

## Installation

1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fashion-demand-optimization
2. Create and activate the Python environment

For example:

conda create -n aiml python=3.11
conda activate aiml
3. Install dependencies
pip install -r requirements.txt
Dataset Setup

Download the Visuelle 2.0 dataset and place the extracted files inside:

data/raw/visuelle2/

The required dataset files include the sales, restocking, and normalization files used by the pipeline.

The dataset is intentionally excluded from Git using .gitignore.

Running the Forecasting Pipeline

From the project root:

conda activate aiml
python src/run_all.py

The pipeline generates forecasting results inside:

results/
Running Model Comparison

To run the model comparison workflow:

python src/run_comparison.py

The final comparison is saved to:

results/model_comparison.csv
Running the Dashboard

Start Streamlit from the project root:
```text
streamlit run dashboard/app.py
```

The dashboard can then be opened in the browser using the local Streamlit URL.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
PyTorch
Statsmodels
Streamlit
Plotly
Limitations

Several limitations should be considered when interpreting the results.

## Short Historical Sequences

The product-level time series contain a relatively small number of weekly observations, which limits the ability of complex time-series models to learn long seasonal patterns.

MAPE and Zero Demand

Fashion demand contains many zero-demand observations. MAPE can therefore become unstable or misleading for individual records.

Estimated Inventory

The dashboard calculates estimated available stock from the available dataset information. It should not be interpreted as a real-time warehouse inventory system.

Identifier Features

Product and store identifiers may contain useful information for learning historical demand patterns but can also introduce dataset-specific patterns that may not generalize to completely new products.

Dataset-Specific Performance

The reported model performance is specific to the Visuelle 2.0 dataset and experimental setup. It should not automatically be generalized to other retailers or fashion markets.

## Future Improvements

Potential extensions include:

Hyperparameter optimization
More advanced temporal cross-validation
Transformer-based forecasting
Temporal Fusion Transformer
LightGBM/CatBoost comparison
Hierarchical forecasting
Probabilistic demand forecasting
Intermittent-demand models
Real-time inventory integration
Lead-time prediction
Multi-period inventory optimization
Cost-based replenishment optimization
Stockout and overstock cost modeling
Automated model retraining
Cloud deployment

## Conclusion

This project demonstrates an end-to-end approach to combining demand forecasting with inventory optimization for fashion retail.

Five forecasting approaches were evaluated:

Naive
SES
SARIMA
XGBoost
LSTM

Among the tested models, XGBoost achieved the strongest overall forecasting performance, producing the lowest MAE, RMSE, and MAPE.

The forecasting output is then connected to an inventory decision layer that calculates safety stock, reorder points, and recommended replenishment quantities.

The resulting Streamlit dashboard provides an interactive interface for converting machine-learning forecasts into practical inventory decisions.

## Author

Rudra Shekhar

AI/ML Project — Fashion Demand Forecasting and Inventory Optimization


### One correction before you paste it

I intentionally used:

```text
<YOUR_GITHUB_REPOSITORY_URL>

rather than inventing your GitHub URL.

Everything else above is based on the actual implementation and results we've established, not a generic project description.
