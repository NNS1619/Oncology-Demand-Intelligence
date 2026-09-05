# Project Methodology

This project asks:

When should a pharmaceutical forecaster rely on historical demand versus patient and market information?

The project uses three analytical notebooks.

## Notebook 1: Synthetic Oncology Market

Notebook 1 creates the synthetic market. Demand is not generated directly from sales noise. It is generated through a patient-flow chain:

population → incidence → diagnosis → metastatic population → biomarker and line of therapy segmentation → eligibility → access → treatment allocation → starts → persistence → active patients → patient-months → latent demand → observed sales.

## Notebook 2: SQL Analytical Mart

Notebook 2 transforms the raw synthetic tables into a modeling dataset. The key concept is the Forecast Information Set. Every feature must be available at the forecast origin, or it cannot be used in the ordinary forecasting experiment.

## Notebook 3: Forecasting and Scenario Intelligence

Notebook 3 compares:

- naive forecast
- seasonal naive forecast
- historical XGBoost
- patient-informed driver model
- hybrid XGBoost

The evaluation uses rolling-origin validation across 1, 3, 6, and 12-month horizons.

The notebook also includes scenario analysis and Monte Carlo uncertainty. Scenario analysis is used to test planning assumptions such as access drops, competitor pressure, epidemiology upside, persistence downside, and supply constraints.
