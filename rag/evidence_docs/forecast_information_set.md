# Forecast Information Set

The Forecast Information Set defines what information is allowed at a forecast origin.

For a forecast made at month tau for target month tau plus h, the model may only use information that would genuinely have been available at tau.

## Allowed Information Categories

1. Observed historical information  
Examples: observed sales through the forecast origin, lagged sales, rolling averages, therapy age.

2. Published forward information  
Examples: an access decision announced before the forecast origin but effective in a future month.

3. Planning assumptions  
Examples: a user-defined scenario where access falls or a competitor launches earlier.

## Forbidden Information

The model may not use future information that was not known at the forecast origin.

Examples:

- unannounced persistence changes
- future observed sales
- future market events that were not announced
- evaluation labels such as target market regime

## Why This Matters

Without point-in-time governance, the model may accidentally learn from the future. That creates leakage and makes backtest results too optimistic.

The Forecast Information Set is the backbone of the project because the project is about information value in forecasting.
