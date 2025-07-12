import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# Load the precomputed DataFrame CSV
@st.cache_data
def load_data(path="data/fine_food_reviews_with_clusters_and_scores.csv"):
    return pd.read_csv(path)


df = load_data()

st.title("Cluster Explorer")

# Sidebar controls
st.sidebar.header("Controls")
# Choose embedding
embed = st.sidebar.selectbox("Embedding", ["PCA", "t-SNE", "UMAP"])
# Dynamically detect available cluster columns
cluster_cols = [col for col in df.columns if col.startswith("Cluster_")]
method = st.sidebar.selectbox("Clustering Method", cluster_cols)


# Compute embeddings based on current selection
def compute_embeddings(X_vec, method_embed):
    if method_embed == "PCA":
        return PCA(n_components=2).fit_transform(X_vec)
    elif method_embed == "t-SNE":
        return TSNE(n_components=2, random_state=42).fit_transform(X_vec)
    else:
        return umap.UMAP(n_components=2, random_state=42).fit_transform(X_vec)


# Parse TF-IDF vector strings into numeric array
def parse_vec(s):
    vals = s.strip("[]").split()
    return np.array([float(v) for v in vals])


@st.cache_data
# cache per (selection, data) combination
def get_embedding_and_df(embed_choice):
    vecs = df["doc_vector_tfidf"].apply(parse_vec)
    X = np.stack(vecs.values)
    emb = compute_embeddings(X, embed_choice)
    return emb


X_emb = get_embedding_and_df(embed)

# Select labels based on chosen clustering method
labels = df[method]

# Build DataFrame for plotting
plot_df = pd.DataFrame(
    {
        "x": X_emb[:, 0],
        "y": X_emb[:, 1],
        "cluster": labels.astype(str),
        "score": df["Score"],
        "summary": df["Summary"],
    }
)

# Render scatter
fig = px.scatter(
    plot_df,
    x="x",
    y="y",
    color="cluster",
    hover_data=["score", "summary"],
    title=f"{embed} projection colored by {method}",
)

st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Data points: ", len(df))
