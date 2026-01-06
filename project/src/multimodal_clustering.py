"""
Multi-modal clustering script 
Combines precomputed audio+lyrics latent vectors (`results/z_hybrid.npy`) with genre metadata (`data/genre/song_quadrant_genre.csv`) to perform clustering.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_ordered_ids(data_root):
    audio_dir = Path(data_root) / "audio"
    lyrics_dir = Path(data_root) / "lyrics"
    audio_files = sorted(audio_dir.glob("*/*.mp3"))
    lyrics_files = sorted(lyrics_dir.glob("*/*.txt"))
    audio_ids = [f.stem for f in audio_files]
    lyrics_ids = [f.stem for f in lyrics_files]
    common_ids = sorted(set(audio_ids) & set(lyrics_ids))
    return common_ids


def load_genre_df(genre_csv_path):
    df = pd.read_csv(genre_csv_path)
    df['Song'] = df['Song'].astype(str).str.strip()
    df['Genres'] = df['Genres'].fillna("")
    return df


def encode_genres(df_genre, song_ids, top_n=200):
    # Build list of genre lists aligned to song_ids
    genres_series = df_genre.set_index('Song').reindex(song_ids)['Genres'].fillna("")
    genre_lists = genres_series.apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
    # compute top genres
    all_genres = pd.Series([g for lst in genre_lists for g in lst])
    if all_genres.empty:
        return None, None
    top_genres = all_genres.value_counts().nlargest(top_n).index.tolist()
    # create multi-hot matrix for top genres
    genre_matrix = []
    for lst in genre_lists:
        row = [1 if g in lst else 0 for g in top_genres]
        genre_matrix.append(row)
    genre_mat = np.array(genre_matrix, dtype=int)
    return genre_mat, top_genres


def encode_quadrant(df_genre, song_ids):
    quad = df_genre.set_index('Song').reindex(song_ids)['Quadrant'].fillna('')
    quad_dummies = pd.get_dummies(quad)
    return quad_dummies.values, list(quad_dummies.columns)


def choose_k_by_silhouette(X, k_min=2, k_max=12):
    best_k = None
    best_score = -1
    for k in range(k_min, min(k_max, len(X)-1) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        if score > best_score:
            best_score = score
            best_k = k
    return best_k or 2


def run_clustering(features, try_hdbscan=True):
    # Try HDBSCAN if available
    if try_hdbscan:
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
            labels = clusterer.fit_predict(features)
            if len(set(labels)) > 1:
                return labels, 'hdbscan'
        except Exception:
            pass
    # Fallback: KMeans with silhouette-based k
    k = choose_k_by_silhouette(features, 2, 12)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels, f'kmeans_k{k}'


def visualize_and_save(X, labels, out_path, title=None):
    tsne = TSNE(n_components=2, random_state=42)
    X2 = tsne.fit_transform(X)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X2[:,0], X2[:,1], c=labels, cmap='tab20', s=10)
    plt.colorbar(scatter, fraction=0.03, pad=0.04)
    if title:
        plt.title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / 'results'
    data_root = repo_root / 'data'
    genre_csv = data_root / 'genre' / 'song_quadrant_genre.csv'

    z_path = results_dir / 'z_hybrid.npy'
    if not z_path.exists():
        print(f"z_hybrid not found at {z_path}")
        return
    z = np.load(z_path)
    print('Loaded z_hybrid with shape', z.shape)

    # Reconstruct ordered song ids (same logic as hybrid_features notebook)
    song_ids = load_ordered_ids(data_root)
    if len(song_ids) != len(z):
        print('Warning: song id count and z_hybrid length differ; using min length')
        n = min(len(song_ids), len(z))
        song_ids = song_ids[:n]
        z = z[:n]

    df_genre = load_genre_df(genre_csv)
    genre_mat, top_genres = encode_genres(df_genre, song_ids, top_n=200)
    quad_mat, quad_cols = encode_quadrant(df_genre, song_ids)

    feature_list = [z]
    if genre_mat is not None:
        feature_list.append(genre_mat)
    if quad_mat is not None and quad_mat.shape[1] > 0:
        feature_list.append(quad_mat)

    X = np.concatenate(feature_list, axis=1)
    # Save raw multimodal latent vectors for future reuse
    multimodal_path = results_dir / 'z_multimodal.npy'
    np.save(multimodal_path, X)
    print('Saved multimodal latent vectors to', multimodal_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Save scaled features and song id mapping
    np.save(results_dir / 'z_multimodal_scaled.npy', X_scaled)
    np.save(results_dir / 'song_ids.npy', np.array(song_ids))
    print('Saved scaled features to', results_dir / 'z_multimodal_scaled.npy')
    print('Saved song id mapping to', results_dir / 'song_ids.npy')

    labels, method = run_clustering(X_scaled, try_hdbscan=True)
    print('Clustering done with method:', method)

    out_df = pd.DataFrame({'Song': song_ids, 'cluster': labels})
    # attach title/quadrant from genre df if available
    meta = df_genre.set_index('Song').reindex(song_ids)[['Title','Quadrant']]
    out_df = out_df.join(meta, on='Song')
    os.makedirs(results_dir, exist_ok=True)
    out_csv = results_dir / 'multi_modal_clusters.csv'
    out_df.to_csv(out_csv, index=False)
    print('Saved clustering results to', out_csv)

    # visualize
    vis_dir = results_dir / 'latent_visualization'
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = vis_dir / f'multimodal_clusters_{method}.png'
    visualize_and_save(X_scaled, labels, str(vis_path), title=f'Multimodal Clusters ({method})')
    print('Saved visualization to', vis_path)


if __name__ == '__main__':
    main()
