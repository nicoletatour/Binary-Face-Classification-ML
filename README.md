# Machine Learning — Binary Face Classification - Kaggle Competition

Classify facial images into **two classes** using two complementary approaches:
1) an **end‑to‑end CNN** trained directly on images, and  
2) a **feature‑based** pipeline that extracts embeddings from a **pretrained CNN** and trains classical ML models (kNN, SVM, Logistic Regression, MLP).

**Primary metric:** F1‑score (also accuracy, precision, recall, confusion matrix, ROC).

## 1) End‑to‑end CNN 
Open `cnn.ipynb` and experiment with:
- **2–5** convolutional layers (3×3) with **ReLU** (+ optional **BatchNorm**)
- **2×2 max‑pool** after blocks
- **1–3** dense layers as the classification head, final **softmax**
- Optional **dropout** and **batch size** sweeps
- **Early stopping** on **val F1**


## 2) Feature‑based pipeline 
1. **Extract features** :
   Use a pretrained model , remove the classifier, and export features for train/val/test to:
   ```
   data/features_train.npy
   data/features_val.npy
   data/features_test.npy
   ```
2. **Train / evaluate models** from the notebooks:
   - `SVMs.ipynb` — kernels: **linear**, **RBF** (cosine ≈ normalize features + linear kernel)
   - `kNN.ipynb` — sweep **K** and distance (**euclidean**, **cosine**)
   - `LogisticRegression.ipynb` — regularization grid (e.g., `C ∈ {0.1, 1, 10, 100}`)
   - `mlp.ipynb` — shallow **MLP** with **1–4** hidden layers, units in **{512, 256, 128, 64}**, activations **ReLU/Tanh**, optimizers **Adam/SGD**
3. **Metrics & plots** (per notebook):
   - Print **F1**, accuracy, precision, recall
   - Save plots to `outputs/`:
     - `outputs/confusion_<model>.png`
     - `outputs/roc_<model>.png`

