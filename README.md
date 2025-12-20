
# 🎧 VAE-Driven Hybrid Language Music Clustering

**Unsupervised Representation Learning for Audio–Text Music Data**

---

## 📖 Overview

This repository implements an **unsupervised learning pipeline based on Variational Autoencoders (VAEs)** to cluster **hybrid-language music tracks (English + Bangla)** using **audio signals and song lyrics**.

The project focuses on learning **compact, semantically meaningful latent representations** from music data and analyzing how well these representations separate songs by **language, style, and acoustic–lyrical similarity** — without using explicit labels during training.

---

## ✨ Key Highlights

* 🎼 **Unsupervised learning** with Variational Autoencoders
* 🔊 **Audio feature learning** from MFCCs and spectrograms
* 📝 **Lyric embedding integration** for hybrid audio–text modeling
* 🔗 **Multi-modal fusion** of music and language representations
* 📊 **Clustering analysis** in learned latent spaces
* 📈 **Baseline comparisons** against PCA and standard Autoencoders
* 🎯 **Clear visualizations** using t-SNE and UMAP

---

## 🧠 Technical Approach

### Representation Learning

* **Variational Autoencoder (VAE)** learns a probabilistic latent space from music features
* Latent variables capture both **acoustic structure** and **linguistic cues**
* Extended variants (CNN-VAE / Beta-VAE / CVAE) support disentanglement and robustness

### Feature Modalities

* **Audio**: MFCCs, Mel-spectrograms
* **Lyrics**: TF-IDF, Word2Vec, or Transformer-based embeddings
* **Fusion**: Concatenation or joint latent modeling

### Clustering

* K-Means
* Agglomerative Clustering
* DBSCAN
  Clustering is performed **only on latent representations**, not raw features.

---

## 📊 Evaluation Strategy

Clustering quality is measured using **standard unsupervised and semi-supervised metrics**:

* Silhouette Score
* Calinski–Harabasz Index
* Davies–Bouldin Index
* Adjusted Rand Index (when labels are available)
* Normalized Mutual Information (NMI)
* Cluster Purity

These metrics help quantify **cluster compactness, separation, and semantic alignment**.

---

## 📁 Project Structure

```bash
project/
├── data/
│   ├── audio/          # WAV / MP3 files
│   └── lyrics/         # Text lyric files
│
├── src/
│   ├── vae.py          # VAE & variants
│   ├── dataset.py     # Feature extraction & loaders
│   ├── clustering.py  # Clustering algorithms
│   └── evaluation.py  # Metrics & analysis
│
├── notebooks/
│   └── exploratory.ipynb
│
├── results/
│   ├── latent_visualization/
│   └── clustering_metrics.csv
│
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/vae-music-clustering.git
cd vae-music-clustering
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare the data

* Place audio files in `data/audio/`
* Place corresponding lyric files in `data/lyrics/`

### 4️⃣ Train & cluster

```bash
python src/vae.py
python src/clustering.py
python src/evaluation.py
```

---

## 📈 Visual Outputs

* Latent space projections (t-SNE / UMAP)
* Cluster distributions
* Cross-modal similarity patterns
* Reconstruction samples from latent space

These visualizations provide **interpretability** for unsupervised learning results.

---

## 🧪 Experimental Focus

* Does a VAE learn better music representations than PCA?
* How does lyric information influence clustering?
* Can latent space separate songs by language without labels?
* Do disentangled VAEs improve cluster stability?

---

## 🔬 Intended Use

* Academic coursework (Neural Networks / Representation Learning)
* Research prototyping in **Music Information Retrieval (MIR)**
* Unsupervised multi-modal learning experiments

---

## 📜 License & Disclaimer

This project is intended for **educational and research purposes only**.
Dataset usage must comply with original dataset licenses and copyright terms.

---

## 👤 Author

**Tanjum Ibnul Mahmud**
Neural Networks — Unsupervised Learning Project

---

