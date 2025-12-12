# Hybrid Language Music Clustering (VAE-based)

Repository skeleton for the Hybrid Language Music Clustering project.

Structure:

project/
data/
audio/ # audio files or extracted features (.gitkeep)
lyrics/ # lyrics files (.gitkeep)
notebooks/
exploratory.ipynb
src/
vae.py
dataset.py
clustering.py
evaluation.py
results/
latent_visualization/ # visualization outputs (.gitkeep)
clustering_metrics.csv

Usage:

- Populate data/audio and data/lyrics with your files.
- Install dependencies: pip install -r requirements.txt
- Use notebooks/exploratory.ipynb to run experiments and visualize results.
- Implement models and utilities in src/ and save outputs in results/.
