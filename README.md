# AM-PM · Asset Allocation Manager

A strategic and tactical (top-down) capital allocation engine for **Colombian collective investment funds (FICs), ETFs and indices**, segmented into Conservative, Moderate and Aggressive risk profiles.

The system doesn't pick individual stocks or bonds. It allocates capital across collective vehicles, using the official daily data that the Financial Superintendence of Colombia (Superintendencia Financiera) publishes on the Open Data Portal (datos.gov.co).

> Code identifiers, CLI flags, console output and the HTML report are in Spanish.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10 or later. Dependencies: `pandas`, `numpy`, `scipy` and `sodapy`.

## Usage

```bash
# Full run: universe, 3 profiles x 4 methods, rebalancing and backtest
python "Asset Allocation.py"

# A single profile, 5 years of history, COP 5 billion in assets
python "Asset Allocation.py" --perfil AGRESIVO --lookback 5 --patrimonio 5e9

# Quick run without backtest, with a datos.gov.co token (avoids throttling)
python "Asset Allocation.py" --sin-backtest --app-token $SODA_APP_TOKEN

# Reproducible offline mode (synthetic data) for testing
python "Asset Allocation.py" --offline

# Open the HTML report in the browser when the run finishes
python "Asset Allocation.py" --abrir

# Console tables only, no HTML report
python "Asset Allocation.py" --sin-html

# Tests
python tests/test_am_pm.py        # or: python -m pytest tests/ -v
```

`python "Asset Allocation.py" --help` lists every parameter (bands, costs, windows, concentration caps, shrinkage, output options).

---

## Architecture

| Module | Responsibility |
|---|---|
| [config.py](am_pm/config.py) | Parameters, taxonomy and constants in immutable dataclasses |
| [ingestion.py](am_pm/ingestion.py) | SODA client (`sodapy`), pagination, cache and synthetic fallback |
| [metrics.py](am_pm/metrics.py) | Return, volatility, Sharpe, Sortino, drawdown, VaR, covariance |
| [universe.py](am_pm/universe.py) | Curated universe and hybrid classification into 5 asset classes |
| [profiles.py](am_pm/profiles.py) | Mandates per profile and their translation into linear constraints |
| [optimizers.py](am_pm/optimizers.py) | Markowitz, Risk Parity, HRP and 1/N over the same feasible set |
| [allocation.py](am_pm/allocation.py) | Model portfolios, cardinality and ex-ante statistics |
| [rebalancing.py](am_pm/rebalancing.py) | Tactical bands, weight drift and order plan |
| [backtest.py](am_pm/backtest.py) | Walk-forward with transaction costs |
| [charts.py](am_pm/charts.py) | Interactive SVG charts and their tooltip layer, with no external libraries |
| [reporting.py](am_pm/reporting.py) | Terminal tables and a self-contained HTML report |
| [pipeline.py](am_pm/pipeline.py) | End-to-end orchestration and executive summary |
| [utils.py](am_pm/utils.py) | Logging, disk cache and series helpers |

---

## Methodology

### 1. Two-stage extraction

Dataset `qhpu-8ixx` has about 3 million rows, so downloading all of it isn't practical. Instead, the pipeline:

1. screens a single daily snapshot of the whole market (~200 funds);
2. keeps the largest share class (by AUM) of each fund;
3. only then downloads the historical series of the selected funds, paginating with `$offset` and caching to disk.

Private equity and real estate funds are excluded: without daily valuation or liquidity, they can't be allocated in a liquid mandate.

### 2. Hybrid asset class classification

The real obstacle in the Colombian market is that **a fund's commercial name doesn't reveal its investment policy**. Names like "Fiducuenta", "Sumar", "Fidugob" or "Valor Plus" hold most of the AUM and say nothing about it. A keyword classifier leaves about 60% of assets unresolved. That's why there are three layers:

1. **Rules** with explicit precedence. "Fondo Bursátil Global X **TES** Colombia" is fixed income even though it says *bursátil* (exchange-traded). "Global X **Colombia** Select" is local equity even though it says *global*: there, "Global X" is the ETF issuer, not the underlying.
2. **A risk model** for inconclusive names: realized annualized volatility, plus betas against a local equity proxy (a COLCAP ETF) and an international/FX exposure proxy.
3. **Reconciliation**: if the rule contradicts the observed risk, the data wins and the discrepancy is logged. This is how the system detects that "Acción Sociedad Fiduciaria" is not an equity fund (0.3% volatility), and that a "USD cash" fund carries equity-like risk for a peso-based investor (11% volatility from the exchange rate).

Each fund ends up with its asset class, the method that determined it and a note, all shown in the curated universe table printed to the console, so the investment committee can audit them. Name vs. risk discrepancies are listed separately in their own review table.

The five asset classes are `RF_CORTO` (short-term fixed income), `RF_MEDIANO_LARGO` (medium/long-term fixed income), `MIXTO` (balanced), `RV_LOCAL` (local equity) and `RV_INTERNACIONAL` (international equity).

### 3. Dynamic risk-free rate

`r_f` isn't an assumption. It is the AUM-weighted, compounded annualized return of the `RF_CORTO` class over the recent window: the Colombian client's real opportunity cost, the money market fund where their money already sits. It is re-estimated in every backtest window.

### 4. Mandate constraints

Each mandate includes:
- bands per asset class;
- an aggregate equity floor and ceiling: 0–10% Conservative, 20–40% Moderate, 50–85% Aggressive;
- a cap per fund and a cap per fund manager (fiduciary risk).

All constraints are linear, so feasibility is **checked with linear programming** before optimizing.

If the universe can't satisfy the mandate, constraints are relaxed in a cumulative cascade: concentration first, strategic minimums next, risk maximums last. Every relaxation is reported, because a mandate silently breached would be worse than an error.

### 5. Optimization

All four methods solve over **the same feasible set**, so differences come from the criterion and not from different degrees of freedom.

- **Markowitz (maximum Sharpe)** via the Schaible transform. The ratio `(μ−r_f)'w / √(w'Σw)` is a fractional problem that SLSQP handles poorly when target volatility is around 0.5% (the Conservative profile case). With `y = κw` it becomes a convex quadratic program with a global optimum.
- **Risk Parity**: minimizes the dispersion of risk contributions.
- **HRP**: hierarchical clustering and recursive bisection, without inverting Σ.
- **1/N**: naive benchmark.

HRP and 1/N are brought into the mandate by **Euclidean projection** onto the feasible set: the closest admissible portfolio to the proposed one.

Σ is estimated with shrinkage toward constant correlation (Ledoit-Wolf-style intensity), and μ is shrunk toward the cross-sectional mean, to avoid chasing whichever fund did best last quarter.

### 6. Tactical rebalancing

The buy-and-hold drift of the model portfolio is simulated, and current weights are compared with the targets. A deviation larger than ±5 percentage points in an asset class triggers an alert, with a suggested action and an order plan in COP, including turnover and estimated cost.

### 7. Walk-forward backtest

At each rebalance:
- `r_f`, μ and Σ are re-estimated **only** from the previous window;
- the eligible universe for that date is rebuilt;
- the portfolio is re-optimized.

Between rebalances, weights drift with the market, and turnover is charged at the configured rate. Results are compared with two passive benchmarks: cash (equal-weighted `RF_CORTO`) and the full equal-weighted universe.

---

## Output

The system **doesn't write CSV files or standalone images**. Results come through two channels:

### 1. Terminal tables

All tabular detail is printed during the run, formatted for an investment desk (aligned columns, percentages and COP amounts already formatted):

| Table | Content |
|---|---|
| Curated universe | Profile of each fund: asset class, classification method, AUM, return, volatility, Sharpe, Sortino, drawdown |
| Name vs. risk discrepancies | Funds whose commercial name contradicts their observed risk, with their betas |
| Investment policy | Bands per asset class and concentration caps for the three mandates |
| Asset class composition | Aggregate weights by profile and method, with total equity |
| Model portfolios | Positions for each profile × method: fund, manager, weight and metrics |
| Rebalancing bands | Target, current weight, deviation, status and suggested action per class |
| Rebalancing orders | Subscriptions and redemptions in COP with estimated cost |
| Backtest | Realized performance per strategy and rebalancing log |
| Comparison table | Ex-ante metrics next to realized walk-forward metrics |
| Executive summary | Committee wrap-up: universe, r_f, portfolios, alerts and backtest |

Use `--max-filas N` to limit the rows per table on long runs.

### 2. Interactive HTML report

A **single self-contained document** (`outputs/informe_am_pm.html`) that you scroll through. Charts are generated as native SVG by [charts.py](am_pm/charts.py): no matplotlib, no CDN, no network. It includes a fixed table of contents, cards with the run's key figures and three sections:

| Section | Content | On hover |
|---|---|---|
| Curated universe | Risk/return map as small multiples, one panel per asset class | Fund name, manager, return, volatility, Sharpe, Sortino, drawdown, AUM and **how it was classified** |
| Strategic composition | Stacked bars by profile and method | Exact segment weight, with its profile and method |
| Walk-forward backtest | Cumulative wealth curves net of costs | A crosshair that snaps to the date, with a tooltip showing **all six series** for that day, sorted |

Interaction details:

- **Nearest-point scatter.** Short-term fixed income funds overlap in a tight cluster, so the cursor only has to be *near* a point, not on top of it.
- **Keyboard.** Points and segments are focusable with Tab. On the curves, arrow keys move across dates (Shift for jumps of 20, Home/End for the ends, Esc to exit).
- **Names treated as untrusted data.** They come from a public API, so they are inserted with `textContent`, never by concatenating HTML.
- **Color-blind-safe palette** (CVD separation ΔE ≥ 8 between adjacent pairs) with direct labeling: a series' identity never depends on color alone, and the tooltip is never the only way to read a value.

Open it with a double click, or with `--abrir` at the end of the run. `--html PATH` sets its location and `--sin-html` skips it.

---

## Caveats

- The backtest re-optimizes over the universe **as it exists today**. Funds that were liquidated or merged aren't in the current dataset, so there is survivorship bias. The results are useful for comparing methods with each other, not as a return forecast.
- Transaction cost is a flat parameter (`--costo-bps`). It doesn't model early-withdrawal penalties or lock-up agreements, which are significant in several Colombian FICs.
- The quantitative classification infers the asset class from price behavior, not from the prospectus. It is an input for the committee, not a replacement for reviewing each fund's regulations.
- `--offline` generates synthetic data: it is useful for testing the code, never for making decisions.
