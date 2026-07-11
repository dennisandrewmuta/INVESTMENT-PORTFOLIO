# Smart Investment Portfolio System using Big Data Analytics

A machine-learning system that learns from 26 years of S&P 500 daily data
(~2.7 million rows, 472 stocks) to rank stocks and build a portfolio.

## The idea in one paragraph

Instead of asking the impossible question *"will this stock go up?"* (the whole
market drifts upward, so any model just learns to say "yes"), this system asks
the useful question: *"will this stock **beat the other stocks** over the next
month?"* That cross-sectional ranking is exactly what you need to choose which
stocks to hold. The model's per-stock accuracy is near 50% (markets are
efficient), but its **ranking** has real value: the stocks it ranks highest go
on to outperform the ones it ranks lowest.

## Project structure

| File | What it does |
|------|--------------|
| `config.py` | All settings in one place (horizon, dates, features, model) |
| `data_pipeline.py` | Loads the CSV, builds technical features, creates labels |
| `train_model.py` | Trains the model, evaluates it, runs the backtest, saves results |
| `predict.py` | Loads the trained model and builds a live portfolio (terminal) |
| `portfolio_engine.py` | Risk profiles, money allocation, and Monte Carlo forecasting |
| `database.py` | SQLite layer — saves portfolios so they can be compared later |
| `app.py` | Streamlit dashboard — the live prototype for your panel demo |
| `requirements.txt` | Python packages needed |

## What the dashboard does

The user enters three things in the sidebar — **amount to invest**, **risk
appetite** (Low / Medium / High), and a **forecast horizon in days** — and gets
six tabs:

- **Recommendations** — the chosen stocks, each with a weight, a dollar
  allocation, and the number of whole shares to buy.
- **Risk Analysis** — annual volatility, historical return, Sharpe ratio, worst
  drawdown, and a check of whether the basket's realised risk matches the chosen
  appetite.
- **Prediction** — a Monte Carlo forecast (5,000 simulations) of where the money
  could end up over the chosen horizon, with expected / best / worst case and the
  chance of profit.
- **Conclusion** — an auto-written summary tying it all together.
- **Stock Lookup** and **Model Backtest** — supporting evidence.

### Design point for your defense

The system deliberately separates two jobs: the **ML model selects stocks** (over
its fixed 21-day training horizon), while a **Monte Carlo simulation forecasts the
money** over any horizon the user picks. The risk profile is applied *after*
selection — it changes how many stocks are held and how they're weighted
(inverse-volatility for Low risk, conviction-weighted for High). This keeps each
component doing one well-defined thing, which is far easier to defend than one
model claiming to do everything.

## The database (SQLite)

`database.py` adds a small SQLite layer (`portfolio.db`, created automatically).
It uses a proper two-table relational schema — `portfolios` (one row per saved
run) and `holdings` (one row per stock, linked by a foreign key with
`ON DELETE CASCADE`). On the **Compare Saved** tab you can save a Low, a Medium,
and a High portfolio and line them up side by side.

Why SQLite (a good answer if the panel asks "why not just files / why not
PostgreSQL?"): the current pipeline is read-only batch analytics on a static
dataset, so files are fine for that part. The database earns its place the moment
you want to *persist and compare* results across runs. SQLite needs zero setup,
is a single file, and ships with Python — right-sized for a single-user
prototype. PostgreSQL/MySQL would only be needed for a multi-user deployed product.

## Presentation

`deck/Smart_Investment_Portfolio_System.pptx` is a 16-slide deck in three
sections — System Design, Prototype (with real screenshots of the interface), and
Results & Findings. Replace the `Presented by:` line on the title slide with your
name.

## The final website (Flask)

`website/` is the production-style version of the system: a Flask backend that
serves the trained model through a JSON API, plus a custom HTML/CSS/JS frontend.
It reuses the exact same `config`, `portfolio_engine`, and `database` modules as
the prototype, so the two never disagree.

```
website/
├── server.py              # Flask app + API routes
├── web_service.py         # loads the model/data once, wraps the engine
├── templates/index.html   # the single-page site
└── static/
    ├── css/style.css      # fintech theme (navy + emerald, mono numerals)
    ├── js/app.js          # fetches the API and draws the charts
    └── vendor/chart.umd.js  # Chart.js, bundled locally (works offline)
```

To run it (from the project root, after `train_model.py` has created the model):

```bash
cd website
python server.py
```

Then open http://localhost:5000. Charts work without internet (Chart.js is
bundled); the web fonts load from Google Fonts when you're online and fall back to
system fonts otherwise.

**Prototype vs final site.** The Streamlit app (`app.py`) is the fast prototype
for exploring and demoing. The Flask site (`website/`) is the deployable product:
a real API plus a frontend you fully control, and the natural place to later add
authentication, a live price feed, or a Flutter/React client — all against the
same model file and database.

## Setup in VS Code

1. Put `SP500_Historical_Data.csv` in the same folder as these files.
2. Open the folder in VS Code, open a terminal, and create a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running it

```bash
python train_model.py     # trains, evaluates, saves model + metrics.txt + results.png
python predict.py          # builds recommended_portfolio.csv from the latest data
streamlit run app.py       # launches the live dashboard for your panel demo
```

The dashboard (`app.py`) is your **prototype layer** for the demo. It has three
tabs: the recommended portfolio (with adjustable size and weighting), a stock
lookup, and the decile backtest. It loads the model `train_model.py` already
saved, so **run `train_model.py` once before launching it.** A browser tab opens
at http://localhost:8501.

> Prototype-to-production note for your defense: Streamlit is ideal for this
> prototype and for internal/analyst tools, and it deploys via Streamlit
> Community Cloud or Docker. For a polished, high-traffic public product you'd
> move to a Flask/FastAPI backend serving the same model file, with a Flutter or
> React frontend — the model and pipeline stay exactly the same.

The first run reads the 137 MB CSV once and caches it as `sp500_cache.pkl`, so
later runs start fast. Training on the full dataset takes about a minute.

**Tip while developing:** in `config.py` set
`SAMPLE_TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "XOM"]` to run on 5 stocks
in seconds, then set it back to `None` for the full run.

## How to talk about the results in your defense

These are the points that separate a strong project from a weak one:

1. **"My accuracy is ~50% and that is correct."** Stock direction is near a coin
   flip because markets are efficient. A classmate reporting 95% accuracy almost
   certainly leaked future information into their features.

2. **"I avoided lookahead bias with a strict time-based split."** Training data
   ends before the test data begins. Features only ever use past prices; only the
   label uses the future. Explain this clearly — examiners probe for it.

3. **"The metric that matters is the decile spread."** Sort predictions into 10
   buckets. The top bucket (the model's favourites) earned a much higher average
   forward return than the bottom bucket. That gap is the model's stock-selection
   edge — and it is what a portfolio actually exploits.

4. **"Why this counts as Big Data Analytics."** ~2.7M rows across 472 entities
   and 26 years; an out-of-core-friendly gradient-boosting model; engineered
   features; a leakage-safe evaluation protocol; and a downstream decision system
   (portfolio construction) built on the predictions.

## Honest limitation

This is an academic system, **not financial advice**. The edge is statistical
and only shows up across a whole basket over many periods — never bet on a single
prediction. Real deployment would also need transaction costs, slippage,
survivorship-bias checks, and risk controls.

## Ideas to extend it (good for extra marks)

- Add **risk-based weighting** (e.g. inverse-volatility) in `predict.py`.
- Compare models: swap `HistGradientBoostingClassifier` for `RandomForest` or
  add `xgboost`, and report which ranks best by decile spread.
- Add a **walk-forward backtest**: retrain every year and chain the test periods.
- Plot a cumulative **equity curve** of the top-decile portfolio vs the market.
