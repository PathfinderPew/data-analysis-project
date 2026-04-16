# Python Data Analysis — Predicting Irrigation Need

Exploratory data analysis on a large-scale agricultural dataset (630,000 records)
using Python. The goal was to identify patterns in soil, weather, and crop data
that influence irrigation requirements.

## Key Findings

- **Class imbalance:** 59% of fields have Low irrigation need, 38% Medium, and only 3% High
- **Soil moisture is the strongest indicator:** High-need fields averaged 17.67 moisture vs 43.31 for Low-need fields
- **Rainfall is inversely related to irrigation need:** High-need fields received ~989mm avg rainfall vs ~1,500mm for Low-need fields
- **Region had minimal impact:** Irrigation need distribution was nearly identical across all 5 regions (Central, East, North, South, West), suggesting soil composition and weather conditions are stronger predictors than geography

## What This Project Does

- Loads and inspects a 630,000-row structured agricultural dataset
- Validates data quality (zero missing values confirmed across all 21 columns)
- Engineers new features including irrigation score mapping and temperature groupings
- Computes group-level statistics to identify trends across soil type, region, and rainfall
- Visualizes results with a 4-chart figure (distribution, box plots, bar charts, correlation heatmap)

## Tech Stack

- Python 3.13
- pandas — data loading, cleaning, groupby analysis
- numpy — numerical transformations
- matplotlib — chart generation
- seaborn — correlation heatmap

## Dataset

Kaggle Playground Series S6E4 — Predicting Irrigation Need  
630,000 rows × 21 columns including soil properties, weather metrics, crop data, and region

## Project Structure
data-analysis-project/
├── data/
│   ├── train.csv                # raw dataset from Kaggle
│   └── analysis_results.png    # generated charts
├── analysis.py                 # main EDA script
├── requirements.txt            # dependencies
└── README.md

## How to Run

```bash
git clone https://github.com/PathfinderPew/data-analysis-project.git
cd data-analysis-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python analysis.py
```

Results are printed to the terminal and the chart is saved to `data/analysis_results.png`.