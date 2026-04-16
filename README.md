# What Do Steam Players Actually Love?

A data analysis of the top 1,000 Steam games by reception, playtime, and value — plus an interactive Streamlit dashboard for exploring the results.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Questions explored

1. What does the review-score landscape look like across Steam's most-played games?
2. Does price predict how well a game is received?
3. Does playtime track reception — do beloved games keep players longer?
4. Which genres are Steam players kindest to?
5. Which games deliver the most hours per dollar?
6. Can we cluster the top 1,000 into recognisable **archetypes** (lifestyle giants, cult indie darlings, controversial megahits, etc.)?

## Project structure

```
├── steam_reception_analysis.ipynb   # Full analysis notebook (EDA + clustering)
├── app.py                           # Streamlit dashboard
├── data_cache/                      # Cached SteamSpy API responses (~1.3 MB)
├── pyproject.toml                   # Dependencies (managed by uv)
└── uv.lock                          # Lockfile for reproducibility
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Install dependencies
uv sync

# Run the notebook (populates data_cache/ on first run)
uv run jupyter lab

# Launch the dashboard
uv run streamlit run app.py
```

## The dashboard

Four tabs, all filterable by review count, price, genre, and archetype:

- **Overview** — review-score distribution, price and playtime scatter plots
- **Genres** — mean review score by genre with adjustable minimum-games threshold
- **Archetypes** — PCA cluster map, profile table, and a "find similar games" recommender
- **Game Explorer** — full sortable/searchable table of every game in the filtered view

## Data source

[SteamSpy API](https://steamspy.com/api.php) (free, public). The notebook fetches the top 1,000 games by estimated owners, then enriches each with per-game details. Cached responses are included in the repo so the dashboard works without hitting the API.

## Tech stack

pandas, NumPy, Matplotlib, seaborn, scikit-learn (KMeans + PCA), Plotly, Streamlit, requests
