#!/usr/bin/env python3
"""Reproduce Appendix 4 endpoint and scale-sensitivity checks.
Run from a directory containing education_series_verified.csv and backtest_metrics.csv.
Requires pandas and numpy. Outputs appendix4_endpoint_sensitivity.csv and
appendix4_normalized_backtest.csv. These are supplementary checks, not significance tests.
"""
import numpy as np
import pandas as pd

series = pd.read_csv("education_series_verified.csv")
metrics = pd.read_csv("backtest_metrics.csv")
endpoint = []
for indicator, label in [("math", "Mathematics"), ("reading", "Reading"), ("science", "Science")]:
    s = series[(series.entity == "Netherlands") & (series.source == "PISA") & (series.indicator == indicator)].sort_values("year")
    slope_all = np.polyfit(s.year.to_numpy(), s.value.to_numpy(), 1)[0]
    s_no2025 = s[s.year < 2025]
    slope_no2025 = np.polyfit(s_no2025.year.to_numpy(), s_no2025.value.to_numpy(), 1)[0]
    endpoint.append({"domain": label, "n_all": len(s), "slope_including_2025": slope_all,
                     "n_excluding_2025": len(s_no2025), "slope_excluding_2025": slope_no2025,
                     "slope_change": slope_no2025 - slope_all})
pd.DataFrame(endpoint).to_csv("appendix4_endpoint_sensitivity.csv", index=False)

normalized = []
for horizon in sorted(metrics.horizon_steps.unique()):
    for method in ["linear", "naive", "timesfm"]:
        rows = metrics[(metrics.horizon_steps == horizon) & (metrics.method == method) & (metrics.entity != "ALL")]
        ratios, weights, series_names = [], [], []
        for _, row in rows.iterrows():
            s = series[(series.entity == row.entity) & (series.source == row.source) &
                       (series.population == row.population) & (series.indicator == row.indicator)].sort_values("year")
            if len(s) < 2:
                continue
            sd = s.value.std(ddof=1)
            if sd and np.isfinite(sd):
                ratios.append(row.MAE / sd)
                weights.append(row.n)
                series_names.append(f"{row.source}-{row.indicator}")
        if ratios:
            normalized.append({"horizon_steps": int(horizon), "method": method,
                               "series_cells": len(ratios), "series_cells_used": ";".join(series_names),
                               "mean_MAE_over_series_SD": float(np.mean(ratios)),
                               "forecast_count_weighted_mean": float(np.average(ratios, weights=weights))})
pd.DataFrame(normalized).to_csv("appendix4_normalized_backtest.csv", index=False)
print("Wrote appendix4_endpoint_sensitivity.csv and appendix4_normalized_backtest.csv")
