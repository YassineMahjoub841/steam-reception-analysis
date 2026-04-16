"""
Steam Top 1,000 Explorer — interactive dashboard for the reception analysis.

Companion to `steam_reception_analysis.ipynb`. Loads the cached SteamSpy data,
re-runs the K-means archetype clustering, and exposes a filterable dataset
with a PCA archetype map and a "find similar games" tool.

Run:
    uv run streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Steam Top 1,000 Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

CACHE_PATH = Path("data_cache/steamspy_appdetails_top1000.json")
K = 5  # number of archetype clusters


# -----------------------------------------------------------------------------
# Data loading and cleaning — mirrors the notebook pipeline
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading Steam data...")
def load_games() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        st.error(
            f"Data cache not found at `{CACHE_PATH}`. "
            "Run the notebook `steam_reception_analysis.ipynb` first to populate it."
        )
        st.stop()

    raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    df = pd.DataFrame.from_dict(raw, orient="index").reset_index(drop=True)

    # Price fields
    for col in ("price", "initialprice"):
        df[f"{col}_usd"] = pd.to_numeric(df[col], errors="coerce") / 100
    df["discount_pct"] = pd.to_numeric(df["discount"], errors="coerce")

    # Owners range -> midpoint
    def parse_owners(s):
        try:
            lo, hi = s.replace(",", "").split(" .. ")
            return (int(lo) + int(hi)) / 2
        except Exception:
            return np.nan

    df["owners_est"] = df["owners"].apply(parse_owners)

    # Review fields
    df["positive"] = pd.to_numeric(df["positive"], errors="coerce")
    df["negative"] = pd.to_numeric(df["negative"], errors="coerce")
    df["reviews_total"] = df["positive"] + df["negative"]
    df["review_score"] = df["positive"] / df["reviews_total"].replace(0, np.nan)

    # Playtime in hours
    df["avg_playtime_hrs"] = pd.to_numeric(df["average_forever"], errors="coerce") / 60
    df["median_playtime_hrs"] = pd.to_numeric(df["median_forever"], errors="coerce") / 60

    # Genres as list
    df["genres_list"] = (
        df["genre"].fillna("").str.split(",")
        .apply(lambda xs: [g.strip() for g in xs if g.strip()])
    )

    df["appid"] = pd.to_numeric(df["appid"], errors="coerce").astype("Int64")

    # Filter to games with enough reviews for a stable score
    df = df[df["reviews_total"] >= 500].copy()
    df = df.drop_duplicates(subset="appid").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Fitting archetypes...")
def fit_clusters(df: pd.DataFrame):
    """Run the same clustering pipeline as the notebook. Returns the clustered
    subset (with cluster, pc1, pc2 columns) and the scaled feature matrix used
    for nearest-neighbor search."""
    clust = df[
        (df["reviews_total"] >= 1_000)
        & (df["median_playtime_hrs"] > 0)
        & (df["owners_est"] > 0)
    ].copy().reset_index(drop=True)

    clust["log_reviews"] = np.log10(clust["reviews_total"])
    clust["log_owners"] = np.log10(clust["owners_est"])
    clust["log_playtime"] = np.log10(clust["median_playtime_hrs"])
    clust["log_price"] = np.log10(clust["price_usd"] + 1)

    feature_cols = [
        "review_score", "log_reviews", "log_owners", "log_playtime", "log_price",
    ]
    scaler = StandardScaler()
    X = scaler.fit_transform(clust[feature_cols].values)

    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    clust["cluster"] = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    clust["pc1"] = coords[:, 0]
    clust["pc2"] = coords[:, 1]

    return clust, X


def build_profile(clust: pd.DataFrame) -> pd.DataFrame:
    return (
        clust.groupby("cluster")
        .agg(
            n_games=("name", "count"),
            review_score=("review_score", "mean"),
            reviews_total=("reviews_total", "median"),
            owners_est=("owners_est", "median"),
            playtime_hrs=("median_playtime_hrs", "median"),
            price_usd=("price_usd", "median"),
        )
    )


def label_clusters(profile: pd.DataFrame) -> dict:
    """Assign human-readable archetype labels to cluster IDs using feature ranks.

    The labels are assigned in a fixed priority order so the result is stable
    regardless of which integer K-means used for which group."""
    labels: dict = {}
    remaining = set(profile.index)

    def pick_from(sub: pd.DataFrame, score_fn):
        return score_fn(sub).idxmax()

    # 1. Lifestyle giants: highest rank-sum of owners + playtime
    sub = profile.loc[list(remaining)]
    cid = pick_from(sub, lambda d: d["owners_est"].rank() + d["playtime_hrs"].rank())
    labels[cid] = "Lifestyle giants"
    remaining.discard(cid)

    # 2. Controversial megahits: lowest review score among remaining
    cid = profile.loc[list(remaining), "review_score"].idxmin()
    labels[cid] = "Controversial megahits"
    remaining.discard(cid)

    # 3. Long-tail hobby: highest playtime among remaining
    cid = profile.loc[list(remaining), "playtime_hrs"].idxmax()
    labels[cid] = "Long-tail hobby"
    remaining.discard(cid)

    # 4. Cult indie darlings: high review score, low price
    sub = profile.loc[list(remaining)]
    cid = pick_from(sub, lambda d: d["review_score"].rank() - d["price_usd"].rank())
    labels[cid] = "Cult indie darlings"
    remaining.discard(cid)

    # 5. The last one: polished premium
    for cid in remaining:
        labels[cid] = "Polished premium"

    return labels


# -----------------------------------------------------------------------------
# Load everything once
# -----------------------------------------------------------------------------
games = load_games()
clustered, X_all = fit_clusters(games)
profile = build_profile(clustered)
archetype_map = label_clusters(profile)
profile["archetype"] = profile.index.map(archetype_map)
clustered["archetype"] = clustered["cluster"].map(archetype_map)

# Merge archetype info back onto the full games dataframe (NaN for games that
# did not qualify for clustering, which is fine)
games = games.merge(
    clustered[["appid", "cluster", "archetype", "pc1", "pc2"]],
    on="appid", how="left",
)

all_genres = sorted({g for gs in games["genres_list"] for g in gs})
all_archetypes = sorted(archetype_map.values())


# -----------------------------------------------------------------------------
# Sidebar — filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    min_reviews = st.select_slider(
        "Minimum reviews",
        options=[500, 1_000, 5_000, 10_000, 50_000, 100_000],
        value=1_000,
        help="Filter out games with fewer reviews than this — their percentages "
             "are noisy.",
    )

    max_price = int(np.ceil(games["price_usd"].max()))
    price_range = st.slider(
        "Price range (USD)",
        min_value=0, max_value=max_price, value=(0, max_price),
    )

    selected_archetypes = st.multiselect(
        "Archetypes",
        options=all_archetypes,
        default=all_archetypes,
        help="Clusters produced by K-means (k=5) on log-scaled standardized features.",
    )

    selected_genres = st.multiselect(
        "Genres (any match)",
        options=all_genres,
        default=[],
        help="Leave empty to include all genres.",
    )

    st.markdown("---")
    st.caption(
        f"**Data:** SteamSpy top 1,000 games by estimated owners.  \n"
        f"**Clustering:** KMeans(k={K}) on log-scaled standardized features.  \n"
        f"**Projection:** PCA to 2D for visualization."
    )


# -----------------------------------------------------------------------------
# Apply filters
# -----------------------------------------------------------------------------
mask = (
    (games["reviews_total"] >= min_reviews)
    & (games["price_usd"] >= price_range[0])
    & (games["price_usd"] <= price_range[1])
)

# Only apply the archetype filter if the user narrowed the selection
if set(selected_archetypes) != set(all_archetypes):
    mask &= games["archetype"].isin(selected_archetypes)

if selected_genres:
    mask &= games["genres_list"].apply(
        lambda gs: any(g in selected_genres for g in gs)
    )

filtered = games[mask].copy()


# -----------------------------------------------------------------------------
# Header + KPIs
# -----------------------------------------------------------------------------
st.title("Steam Top 1,000 Explorer")
st.caption(
    "Interactive companion to the `steam_reception_analysis` notebook. "
    "Filter the dataset, explore the game archetypes, and find games similar "
    "to one you already love."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Games in view", f"{len(filtered):,}")
k2.metric(
    "Median review score",
    f"{filtered['review_score'].median():.1%}" if len(filtered) else "—",
)
k3.metric(
    "Median playtime",
    f"{filtered['median_playtime_hrs'].median():.1f} h" if len(filtered) else "—",
)
k4.metric(
    "Median price",
    f"${filtered['price_usd'].median():.2f}" if len(filtered) else "—",
)

if len(filtered) == 0:
    st.warning("No games match the current filters. Try relaxing them.")
    st.stop()


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_genres, tab_archetypes, tab_explorer = st.tabs(
    ["Overview", "Genres", "Archetypes", "Game Explorer"]
)


# ---- Tab: Overview ----------------------------------------------------------
with tab_overview:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Review-score distribution")
        fig = px.histogram(
            filtered, x="review_score", nbins=40,
            color_discrete_sequence=["#2e86ab"],
        )
        fig.update_layout(
            xaxis_title="Positive review share",
            yaxis_title="Number of games",
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Price vs review score")
        paid = filtered[filtered["price_usd"] > 0]
        if len(paid):
            fig = px.scatter(
                paid, x="price_usd", y="review_score",
                size="owners_est", color="archetype",
                hover_name="name",
                hover_data={
                    "price_usd": ":.2f", "review_score": ":.2f",
                    "median_playtime_hrs": ":.1f",
                    "owners_est": False, "archetype": False,
                },
                opacity=0.7,
            )
            fig.update_layout(
                xaxis_title="Price (USD)",
                yaxis_title="Review score",
                margin=dict(l=10, r=10, t=10, b=10),
                height=380,
                legend=dict(orientation="h", y=-0.25, title=""),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No paid games in the current filter.")

    st.subheader("Playtime vs review score (log-scale x)")
    pt = filtered[filtered["median_playtime_hrs"] > 0]
    if len(pt):
        fig = px.scatter(
            pt, x="median_playtime_hrs", y="review_score",
            color="archetype", hover_name="name", log_x=True,
            hover_data={
                "median_playtime_hrs": ":.1f", "review_score": ":.2f",
                "price_usd": ":.2f", "archetype": False,
            },
            opacity=0.65,
        )
        fig.update_layout(
            xaxis_title="Median playtime (hours, log scale)",
            yaxis_title="Review score",
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
            legend=dict(orientation="h", y=-0.2, title=""),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---- Tab: Genres ------------------------------------------------------------
with tab_genres:
    st.subheader("Review score by genre  (filtered view)")
    genre_df = filtered.explode("genres_list").rename(
        columns={"genres_list": "genre_item"}
    )
    genre_df = genre_df[
        genre_df["genre_item"].notna() & (genre_df["genre_item"] != "")
    ]

    if len(genre_df) == 0:
        st.info("No genre data for the current filter.")
    else:
        min_n = st.slider(
            "Minimum games per genre",
            min_value=1, max_value=30, value=5,
            help="Genres with fewer than this many games in the current filter are hidden.",
        )
        genre_stats = (
            genre_df.groupby("genre_item")
            .agg(
                n_games=("name", "count"),
                mean_review=("review_score", "mean"),
                median_owners=("owners_est", "median"),
            )
            .query("n_games >= @min_n")
            .sort_values("mean_review")
            .reset_index()
        )
        if len(genre_stats) == 0:
            st.info("No genres meet the minimum games threshold.")
        else:
            fig = px.bar(
                genre_stats, x="mean_review", y="genre_item",
                orientation="h", color="mean_review",
                color_continuous_scale="RdYlGn",
                hover_data={"n_games": True, "median_owners": ":,.0f"},
            )
            fig.update_layout(
                xaxis_title="Mean review score",
                yaxis_title="",
                margin=dict(l=10, r=10, t=10, b=10),
                height=max(400, 28 * len(genre_stats)),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ---- Tab: Archetypes --------------------------------------------------------
with tab_archetypes:
    st.subheader("Archetype map  (PCA projection of the 5D feature space)")
    st.caption(
        "Each point is a game, placed by PCA projection of the 5 clustering "
        "features (review score + log reviews/owners/playtime/price). Colour = archetype."
    )

    pca_df = filtered[filtered["pc1"].notna()]
    if len(pca_df):
        fig = px.scatter(
            pca_df, x="pc1", y="pc2", color="archetype",
            hover_name="name",
            hover_data={
                "review_score": ":.2f", "price_usd": ":.2f",
                "median_playtime_hrs": ":.1f",
                "pc1": False, "pc2": False, "archetype": False,
            },
            opacity=0.7,
        )
        fig.update_layout(
            xaxis_title="PC1", yaxis_title="PC2",
            margin=dict(l=10, r=10, t=10, b=10),
            height=520,
            legend=dict(orientation="h", y=-0.15, title=""),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No clustered games in the current filter.")

    st.subheader("Archetype profiles")
    st.caption(
        "Median values within each cluster — the archetype's 'personality'. "
        "Unaffected by the sidebar filters."
    )
    display_profile = profile.reset_index()[[
        "archetype", "n_games", "review_score", "price_usd",
        "playtime_hrs", "owners_est", "reviews_total",
    ]].rename(columns={
        "archetype": "Archetype",
        "n_games": "Games",
        "review_score": "Mean review",
        "price_usd": "Median price",
        "playtime_hrs": "Median playtime (h)",
        "owners_est": "Median owners",
        "reviews_total": "Median reviews",
    })
    st.dataframe(
        display_profile.style.format({
            "Mean review": "{:.1%}",
            "Median price": "${:.2f}",
            "Median playtime (h)": "{:.1f}",
            "Median owners": "{:,.0f}",
            "Median reviews": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Find games similar to...")
    st.caption(
        "Pick a game and I'll return the 5 nearest neighbors in the scaled "
        "feature space that drives the clustering. This works like a tiny "
        "content-based recommender."
    )

    candidates = sorted(clustered["name"].dropna().unique().tolist())
    choice = st.selectbox(
        "Pick a game",
        options=["(none selected)"] + candidates,
        index=0,
    )

    if choice and choice != "(none selected)":
        matches = clustered.index[clustered["name"] == choice].tolist()
        if matches:
            idx = matches[0]
            target = X_all[idx]
            dists = np.linalg.norm(X_all - target, axis=1)
            nearest = clustered.assign(distance=dists)
            nearest = (
                nearest[nearest["name"] != choice]
                .nsmallest(5, "distance")
                [[
                    "name", "archetype", "review_score", "price_usd",
                    "median_playtime_hrs", "distance",
                ]]
                .reset_index(drop=True)
                .rename(columns={
                    "name": "Game",
                    "archetype": "Archetype",
                    "review_score": "Review score",
                    "price_usd": "Price",
                    "median_playtime_hrs": "Playtime (h)",
                    "distance": "Distance",
                })
            )
            st.dataframe(
                nearest.style.format({
                    "Review score": "{:.1%}",
                    "Price": "${:.2f}",
                    "Playtime (h)": "{:.1f}",
                    "Distance": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )


# ---- Tab: Game Explorer -----------------------------------------------------
with tab_explorer:
    st.subheader("Game table")
    st.caption(
        f"{len(filtered):,} games match the current filters. "
        "Click any column header to sort."
    )
    cols = [
        "name", "archetype", "review_score", "reviews_total",
        "price_usd", "median_playtime_hrs", "owners_est", "genre",
    ]
    display_df = (
        filtered[cols]
        .sort_values("review_score", ascending=False)
        .rename(columns={
            "name": "Game",
            "archetype": "Archetype",
            "review_score": "Review",
            "reviews_total": "# Reviews",
            "price_usd": "Price",
            "median_playtime_hrs": "Playtime (h)",
            "owners_est": "Owners (est.)",
            "genre": "Genre",
        })
    )
    st.dataframe(
        display_df.style.format({
            "Review": "{:.1%}",
            "# Reviews": "{:,.0f}",
            "Price": "${:.2f}",
            "Playtime (h)": "{:.1f}",
            "Owners (est.)": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True, height=600,
    )
