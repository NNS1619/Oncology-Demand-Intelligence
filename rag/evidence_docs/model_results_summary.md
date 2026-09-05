# Model Results Summary

The forecasting experiment compared historical, patient-informed, and hybrid approaches across 1, 3, 6, and 12-month horizons.

## Main Finding

Recent observed demand was the strongest overall benchmark. The naive model performed best across all horizons.
 In this synthetic oncology market, demand is persistent because active treated patients continue over time. Recent sales therefore act as a strong proxy for current treated-patient stock.

## Forecast Value Add

Historical XGBoost and hybrid XGBoost did not beat naive overall. The patient-driver model also did not beat naive overall.

However, non-naive methods added value in selected market regimes:

- Patient-driver helped in limited-supply windows.
- Hybrid XGBoost helped in persistence-change windows.

## Business Interpretation

For routine short-term demand planning, recent observed demand may be sufficient.

For event-driven planning, patient and market signals are valuable because they help explain what changes under access, competition, persistence, epidemiology, and supply scenarios.
