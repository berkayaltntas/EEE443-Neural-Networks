# 🧠 EEE443 - Neural Networks

This repository contains materials and implementations developed for the EEE443 Neural Networks course. It includes:

- 🧪 Mini projects showcasing foundational deep learning concepts implemented from scratch using Python and NumPy  
- 📘 Weekly tutorials exploring core neural network components and training mechanisms

---

## 🧪 Mini Projects


### 📦 Dataset

The dataset used in the mini project is publicly available on [my Hugging Face profile](https://huggingface.co/berkayaltntas) and can be downloaded directly from the link below:

🔗 [Download datasets.zip](https://huggingface.co/berkayaltntas/eee443-miniproject-dataset/resolve/main/datasets.zip)

### 🔹 Autoencoder for Unsupervised Feature Extraction | Python, NumPy

- Extracted compact and meaningful representations from natural images without labeled data  
- Implemented a single-layer autoencoder with **Tikhonov regularization** and **KL divergence**  
- Preprocessed inputs with grayscale conversion, mean normalization, and pixel scaling  
- Trained using mini-batch gradient descent

---

### 🔹 Neural Network-Based Language Model | Python, NumPy

- Designed a feedforward network to predict the next word from trigram inputs  
- Used shared word embeddings, sigmoid hidden layer, and softmax output  
- Trained with mini-batch SGD with momentum and early stopping  
- Evaluated using top-10 prediction accuracy on unseen trigrams

---

### 🔹 Human Activity Recognition with RNN, LSTM, and GRU | Python, NumPy

- Classified physical activities using multivariate time-series data from motion sensors  
- Implemented and compared RNN, LSTM, and GRU architectures trained with BPTT  
- Used Xavier initialization and Adam optimizer  
- GRU achieved the best performance with **86.5% test accuracy**

---

## 📘 Tutorials

Tutorials from the course exploring key neural network operations:

- `Tutorial 1 - Forward and Backward Pass of Linear Layer.ipynb`
- `Tutorial 2 - Multi-Layer Perceptron and Backpropagation.ipynb`
- `Tutorial 3 - Training with Gradient Descent.ipynb`
- `Tutorial 4 - Activation Functions.ipynb`
- `Tutorial 5 - CNNs with NumPy.ipynb`
- `Tutorial 6 - CNN in PyTorch.ipynb`

> All tutorials are located in the [`Tutorials/`](./Tutorials) folder.

---




