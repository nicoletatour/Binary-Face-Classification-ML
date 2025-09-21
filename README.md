# Machine Learning — Binary Face Classification - Kaggle Competition

Classify facial images into **two classes**:
1) an **end-to-end CNN** trained on images, and  
2) a **feature-based** pipeline using features from a **pretrained CNN** with classical ML (kNN, SVM, Logistic Regression, MLP).

**Primary metric:** F1-score (also report accuracy, precision, recall, confusion matrix, ROC).  

### 1) End-to-end CNN (images → CNN → softmax)
Open `cnn.ipynb` and try:
- **2–5** conv layers (3×3) with **ReLU** (+ optional **BatchNorm**), **2×2 max-pool**,
- **1–3** dense layers in the head, final **softmax**,
- optional **dropout** and batch size sweeps,
- early stopping on **val F1**.
Save weights to `models/` and predictions to `submissions/`

### 2) Feature-based pipeline (pretrained CNN features + classical ML)
1. **Extract features** (add a small cell if needed):
   - Load a pretrained model (e.g., `torchvision.models.resnet18(weights=...)`), remove the classification head, and export features for train/val/test to:
     ```
     data/features_train.npy
     data/features_val.npy
     data/features_test.npy
     ```
2. **Train/evaluate models**:
   - `SVMs.ipynb` — kernels: **linear**, **RBF** (cosine can be approximated by L2-normalizing features and using linear).
   - `kNN.ipynb` — sweep **K** and distance (**euclidean**, **cosine**).
   - `LogisticRegression.ipynb` — regularization grid (e.g., `C ∈ {0.1, 1, 10, 100}`).
   - `mlp.ipynb` — shallow MLP: **1–4** hidden layers, units in **{512, 256, 128, 64}**, ReLU/Tanh; optimizers: Adam/SGD.
3. **Metrics & plots**: each notebook prints **F1**, accuracy, precision, recall and saves:
   - `outputs/confusion_<model>.png`
   - `outputs/roc_<model>.png`

