# Patient-Informed Oncology Demand Forecasting and Scenario Intelligence

This project is a synthetic pharmaceutical analytics POC that evaluates when historical demand is sufficient and when patient-flow, epidemiology, market access, persistence, supply, and competitive signals add value for oncology demand planning.

The project uses a controlled synthetic oncology data-generating process, point-in-time feature governance, rolling-origin validation across 1/3/6/12-month horizons, forecast-value-add testing, Monte Carlo uncertainty, scenario analysis, and a bounded RAG/LLM explanation layer.

## Core Question

When should a pharmaceutical forecaster rely on historical demand versus patient and market information?

## Project Structure

- `notebooks/`: analytical notebooks
- `data/raw_synthetic/`: synthetic market-generation outputs
- `data/processed/`: SQL analytical mart and modeling datasets
- `data/outputs/`: forecasting, scenario, and review outputs
- `app/`: Streamlit decision-support app
- `rag/`: embedding-based evidence retrieval and explanation layer
- `docs/`: methodology and project documentation

## Important Limitation

This is a synthetic case study, not a clinically validated NSCLC model. Public epidemiology sources are used only as proxy anchors for selected assumptions. The project demonstrates forecasting methodology, pharmaceutical decision logic, feature governance, uncertainty analysis, scenario planning, and grounded explanation architecture.
