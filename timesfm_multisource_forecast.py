#!/usr/bin/env python3
"""
Multi-source education forecasting: TimesFM 2.5 vs. linear-trend and naive baselines.

WHY SEPARATE SERIES
-------------------
PISA (age 15), TIMSS (grade 4 / grade 8), PIRLS (grade 4) and national or
Eurostat indicators use different tests, scales and populations. They must NOT
be spliced into one line. This script forecasts every source separately and
then compares their TRENDS side by side (corroboration, not merging).

INPUT
-----
Put one or more CSV files in ./data/ with these columns (long format):

    entity,source,population,indicator,year,value[,se][,note]
    Netherlands,PISA,age 15,math,2003,538
    Netherlands,TIMSS,grade 8,math,1995,<value from IEA>
    Netherlands,Eurostat,age 18-24,early_leavers_pct,2004,<value>

* Use the SAME indicator names across sources where they measure the same
  domain (math, reading, science) so the comparison plots can pair them.
* Optional `se` = published standard error (TIMSS, PIRLS). If every point of a
  series has one, trend_summary.csv also reports an SE-weighted slope.
* Rows with an empty value are dropped (with a warning).
* Only enter real published values. Never interpolate before this script.
* Run  python timesfm_multisource_forecast.py --make-template  to create
  data/TEMPLATE_education_series.csv (ignored by the loader).

Where to get the data (check current dataset names on each site):
    PISA, EU/OECD averages : OECD PISA Data Explorer
    TIMSS, PIRLS           : IEA Data Explorer / TIMSS & PIRLS study centre
    Dutch national series  : CBS StatLine (open data), Inspectie van het Onderwijs
    EU annual indicators   : Eurostat education tables

OUTPUT (./timesfm_output/)
--------------------------
    forecast_<series>.csv / .png       full forecast with q10-q90 band
    forecast_milestones.csv            about 5, 10 and 20 years ahead
    trend_summary.csv                  OLS slope per year, standard error, n
    backtest_errors.csv / backtest_metrics.csv
    comparison_<entity>_<indicator>.png  change since first observation, by source

Requirements: pip install timesfm pandas numpy matplotlib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
DATA_DIR = Path("data")
OUT = Path("timesfm_output")
HORIZON_YEARS = [5, 10, 20]   # converted to steps using each series' own spacing
MIN_OBS_FORECAST = 4          # fewer real points than this: series is skipped
MIN_TRAIN = 4                 # smallest training set in the backtest
BACKTEST_H = (1, 2)           # backtest horizons in steps (cycles or years)
KEY = ["entity", "source", "population", "indicator"]
REQUIRED = KEY + ["year", "value"]


# ----------------------------------------------------------------------
# TEMPLATE
# ----------------------------------------------------------------------
def make_template():
    DATA_DIR.mkdir(exist_ok=True)
    rows = [
        ("Netherlands", "PISA", "age 15", "math", 2003, ""),
        ("Netherlands", "TIMSS", "grade 8", "math", 1995, ""),
        ("Netherlands", "PIRLS", "grade 4", "reading", 2001, ""),
        ("Netherlands", "Eurostat", "age 18-24", "early_leavers_pct", 2004, ""),
        ("EU", "PISA", "age 15", "math", 2003, ""),
    ]
    df = pd.DataFrame(rows, columns=REQUIRED)
    df["note"] = "EXAMPLE ROW: replace with real published values"
    path = DATA_DIR / "TEMPLATE_education_series.csv"
    df.to_csv(path, index=False)
    print(f"Template written to {path}")


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def load_all():
    files = [f for f in sorted(DATA_DIR.glob("*.csv"))
             if not f.name.upper().startswith("TEMPLATE")]
    if not files:
        sys.exit(f"No CSV files in {DATA_DIR}/. Run with --make-template first. "
                 "This script never creates demo data.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing = set(REQUIRED) - set(df.columns)
        if missing:
            sys.exit(f"{f}: missing columns {sorted(missing)}")
        df["file"] = f.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    data["year"] = data["year"].astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    empty = data["value"].isna().sum()
    if empty:
        print(f"Warning: dropped {empty} rows with empty/non-numeric value")
        data = data.dropna(subset=["value"])

    dup = data.duplicated(KEY + ["year"], keep=False)
    if dup.any():
        sys.exit("Duplicate rows for the same series and year:\n"
                 + data[dup].sort_values(KEY + ["year"]).to_string(index=False))
    return data.sort_values(KEY + ["year"]).reset_index(drop=True)


def series_id(key):
    return "_".join(str(k) for k in key).lower().replace(" ", "").replace("/", "-")


def spacing(years):
    return max(1, int(np.floor(np.median(np.diff(years)) + 0.5)))


def to_steps(h, sp):
    """Years ahead -> number of assessment cycles (rounds half up)."""
    return max(1, int(np.floor(h / sp + 0.5)))


# ----------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------
def naive_forecast(y, h):
    return np.repeat(y[-1], h)


def linear_forecast(years, y, future_years):
    slope, intercept = np.polyfit(years, y, 1)
    return intercept + slope * np.asarray(future_years, dtype=float)


def ols_trend(years, y):
    """Slope per year with its standard error (n-2 degrees of freedom)."""
    n = len(y)
    x = years.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    se = np.sqrt(np.sum(resid ** 2) / (n - 2) / np.sum((x - x.mean()) ** 2)) if n > 2 else np.nan
    return slope, se


def wls_trend(years, y, se):
    """Slope weighted by published standard errors. It treats the SEs as known,
    so its SE is optimistic; the OLS SE reflects the actual scatter."""
    if se is None or len(y) < 3 or not np.all(np.isfinite(se)):
        return np.nan, np.nan
    coef, cov = np.polyfit(years.astype(float), y, 1, w=1.0 / se, cov="unscaled")
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def load_timesfm(max_horizon):
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def timesfm_forecast(model, y, h):
    """Point forecast, q10, q90 (each length h)."""
    point, quant = model.forecast(horizon=h, inputs=[y.astype(np.float32)])
    point, quant = np.asarray(point)[0], np.asarray(quant)[0]
    # quant[:, 0] = mean, quant[:, 1..9] = quantiles 0.1 ... 0.9
    return point, quant[:, 1], quant[:, 9]


# ----------------------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------------------
def backtest(model, key, years, y):
    rows = []
    for h in BACKTEST_H:
        for n in range(MIN_TRAIN, len(y) - h + 1):
            train_y, train_years = y[:n], years[:n]
            t = n + h - 1
            tf_point, _, _ = timesfm_forecast(model, train_y, h)
            preds = {
                "timesfm": tf_point[-1],
                "linear": linear_forecast(train_years, train_y, [years[t]])[0],
                "naive": naive_forecast(train_y, h)[-1],
            }
            for method, pred in preds.items():
                rows.append(dict(zip(KEY, key)) | dict(
                    horizon_steps=h, train_end_year=int(train_years[-1]),
                    target_year=int(years[t]), method=method, actual=y[t],
                    predicted=float(pred), error=float(pred - y[t])))
    return rows


def summarise(errors, by):
    return errors.groupby(by)["error"].agg(
        n="count",
        MAE=lambda e: float(np.mean(np.abs(e))),
        RMSE=lambda e: float(np.sqrt(np.mean(np.square(e)))),
    ).reset_index()


# ----------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------
def plot_forecast(key, years, y, fc, path):
    label = " | ".join(str(k) for k in key)
    plt.figure(figsize=(11, 6))
    plt.plot(years, y, "o-", color="black", label="Observed")
    plt.plot(fc.year_label, fc.timesfm, "x--", color="tab:blue", label="TimesFM")
    plt.fill_between(fc.year_label, fc.q10, fc.q90, color="tab:blue",
                     alpha=0.2, label="TimesFM q10-q90")
    plt.plot(fc.year_label, fc.linear, ":", color="tab:red", label="Linear trend")
    plt.plot(fc.year_label, fc.naive, "-.", color="tab:gray", label="Naive")
    plt.axvline(years[-1], color="grey", linestyle=":")
    plt.title(label)
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_comparison(data):
    """Change since each source's first observation (scales differ by source)."""
    for (entity, indicator), sub in data.groupby(["entity", "indicator"]):
        groups = list(sub.groupby(["source", "population"]))
        if len(groups) < 2:
            continue
        plt.figure(figsize=(11, 6))
        for (source, pop), g in groups:
            g = g.sort_values("year")
            plt.plot(g.year, g.value - g.value.iloc[0], "o-", label=f"{source} ({pop})")
        plt.axhline(0, color="grey", linewidth=0.8)
        plt.title(f"{entity}: {indicator}, change since first observation\n"
                  "(different tests and scales: compare direction, not level)")
        plt.xlabel("Year")
        plt.ylabel("Points change vs. own first observation")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"comparison_{series_id((entity, indicator))}.png", dpi=200)
        plt.close()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", action="store_true")
    args = ap.parse_args()
    if args.make_template:
        make_template()
        return

    OUT.mkdir(exist_ok=True)
    data = load_all()

    # Inventory: this table belongs in the paper's data section.
    inv = (data.groupby(KEY)
           .agg(n_obs=("year", "size"), first_year=("year", "min"),
                last_year=("year", "max"), last_value=("value", "last"))
           .reset_index())
    inv.to_csv(OUT / "data_inventory.csv", index=False)
    print("Data inventory (check last values against the published tables):")
    print(inv.to_string(index=False))

    usable = []
    for key, sub in data.groupby(KEY):
        if len(sub) < MIN_OBS_FORECAST:
            print(f"Skipping {key}: only {len(sub)} observations")
            continue
        se = sub["se"].to_numpy(dtype=float) if "se" in sub else None
        usable.append((key, sub["year"].to_numpy(), sub["value"].to_numpy(), se))
    if not usable:
        sys.exit("No series with enough observations.")

    max_steps = max(max(to_steps(h, spacing(yrs)) for h in HORIZON_YEARS)
                    for _, yrs, _, _ in usable)
    model = load_timesfm(max_horizon=max(max_steps, max(BACKTEST_H)))

    milestones, trends, errors = [], [], []
    for key, years, y, se in usable:
        sp = spacing(years)
        step_map = {h: to_steps(h, sp) for h in HORIZON_YEARS}
        n_steps = max(step_map.values())
        steps = np.arange(1, n_steps + 1)
        year_label = years[-1] + sp * steps
        print(f"\n{' | '.join(map(str, key))}: {len(y)} obs, "
              f"{years[0]}-{years[-1]}, spacing {sp} y")

        point, q10, q90 = timesfm_forecast(model, y, n_steps)
        fc = pd.DataFrame({
            "step": steps, "year_label": year_label,
            "timesfm": point, "q10": q10, "q90": q90,
            "linear": linear_forecast(years, y, year_label),
            "naive": naive_forecast(y, n_steps),
        })
        for k, v in zip(KEY, key):
            fc.insert(0, k, v)
        sid = series_id(key)
        fc.to_csv(OUT / f"forecast_{sid}.csv", index=False)
        plot_forecast(key, years, y, fc, OUT / f"forecast_{sid}.png")

        for h, s in step_map.items():
            row = fc[fc.step == s].copy()
            row["requested_years_ahead"] = h
            row["actual_years_ahead"] = int(year_label[s - 1] - years[-1])
            milestones.append(row)

        slope, slope_se = ols_trend(years, y)
        w_slope, w_se = wls_trend(years, y, se)
        trends.append(dict(zip(KEY, key)) | dict(
            n_obs=len(y), first_year=int(years[0]), last_year=int(years[-1]),
            slope_per_year=slope, slope_se=slope_se,
            slope_wls=w_slope, slope_wls_se=w_se,
            total_change=float(y[-1] - y[0])))

        if len(y) >= MIN_TRAIN + 1:
            errors.extend(backtest(model, key, years, y))
        else:
            print("  too few points for a backtest")

    if milestones:
        ms = pd.concat(milestones, ignore_index=True)
        ms.to_csv(OUT / "forecast_milestones.csv", index=False)
        print("\nMilestone forecasts:")
        print(ms.round(1).to_string(index=False))

    tr = pd.DataFrame(trends)
    tr.to_csv(OUT / "trend_summary.csv", index=False)
    print("\nTrend summary (slope per year with standard error):")
    print(tr.round(2).to_string(index=False))

    if errors:
        err = pd.DataFrame(errors)
        err.to_csv(OUT / "backtest_errors.csv", index=False)
        metrics = pd.concat([
            summarise(err, KEY + ["horizon_steps", "method"]),
            summarise(err, ["horizon_steps", "method"]).assign(
                entity="ALL", source="ALL", population="ALL", indicator="ALL"),
        ], ignore_index=True)
        metrics.to_csv(OUT / "backtest_metrics.csv", index=False)
        print("\nBacktest, pooled over all series:")
        print(metrics[metrics.entity == "ALL"].round(2).to_string(index=False))

    plot_comparison(data)
    print(f"\nDone. Files written to {OUT.resolve()}")
    print("Reminder: series are short; report backtests as indicative and the "
          "forecast as a scenario, not a prediction.")


if __name__ == "__main__":
    main()
