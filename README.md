# 🧠 Graph-Based Sentence Boundary Detection with Entropy Modeling

This repository presents a **hybrid NLP system** for **sentence boundary detection** using **graph traversal**, **Node2Vec embeddings**, **neural entropy modeling**, and **CRF post-processing**. The approach integrates **syntactic parsing**, **knowledge graph construction**, and **deep learning** for accurate detection of sentence boundaries in unstructured text — especially where punctuation and casing may be unreliable.

---

## 🚀 Features

- **SVO Triplet Extraction** using [spaCy](https://spacy.io/)
- **Knowledge Graph Construction** with NetworkX
- **Node2Vec Embeddings** for entity representation
- **Graph & Embedding Feature Engineering**
- **Entropy-Based Neural Network** for boundary probability estimation
- **CRF (Conditional Random Fields)** for sequence-level smoothing
- **Entropy Visualization** and sentence reconstruction
- **Dynamic Threshold Tuning** and **Temperature Calibration**
- **Modular, Resumable Pipeline** (caching & file checks included)

---

## 📁 Project Structure

```bash
├── Fully_Working.py           # Main pipeline for training, inference, and evaluation
├── crf_utils.py               # (Required) Utility for CRF feature formatting
├── war_and_peace.txt          # Cached source text (downloaded automatically)
├── *.pkl                      # Intermediate data, trained models, and embeddings
└── README.md                  # Project documentation
```

## 📊 Model Performance

After full training on 12,000 paragraphs:

| **Metric**           | **Value**        |
|----------------------|------------------|
| Precision            | 0.89             |
| Recall               | 0.81             |
| F1-Score             | 0.85             |
| Accuracy             | 0.93             |
| Best Threshold       | 0.70             |
| Temperature (ECE)    | 1.3 (ECE = 0.03) |

> Evaluation is performed on the test set extracted from *War and Peace*.  
> See `run_inference()` for real-time metrics logging.

---
## 🧰 Requirements

- Python 3.8+
- PyTorch
- spaCy (`en_core_web_sm`)
- scikit-learn
- NetworkX
- Gensim (for Node2Vec)
- NLTK
- matplotlib
- tqdm
- sklearn-crfsuite

### Install Dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```
---
## 📦 How It Works

### Data Preprocessing

- Downloads and cleans *War and Peace* from Project Gutenberg.
- Splits text into paragraphs and tokenizes into sentences.

### Triplet Extraction

- Applies spaCy to extract Subject-Verb-Object (SVO) triplets per sentence.

### Knowledge Graph Construction

- Entities and actions are added as nodes and edges to a directed graph.

### Feature Generation

- Extracts both structural (graph-based) and semantic (embedding-based) features for node pairs.

### Model Training

- Trains a neural model to estimate entropy for each token using:
  - Binary Cross Entropy
  - KL-Divergence
  - Entropy penalty loss
- Applies temperature scaling and ECE for probabilistic calibration.
- Trains a CRF to post-process entropy outputs for structured prediction.

### Inference

- Accepts custom user input.
- Applies the entropy model + CRF for boundary detection.
- Prints token-wise entropy, visualizes predictions, and reconstructs sentence splits.
---
## 📈 Visualization Example

The system provides an interactive entropy analysis that includes:

- Entropy line chart with annotated tokens
- Boundary distribution bar chart
- Token-wise entropy histogram
- Detected sentence boundaries overlay

> Visualizations are generated using `matplotlib` in the `create_traversal_visualization()` function.
---
## 🧠 Techniques Used

- SpaCy dependency parsing for SVO extraction  
- Node2Vec for semantic graph learning  
- Neural network entropy modeling (multi-loss)  
- Temperature scaling and ECE for calibration  
- Sequence modeling with CRF  

---

## 🛠 Future Work

- Add multilingual model support (via multilingual spaCy + training)  
- Convert pipeline to API or Streamlit web app  
- Integrate with document segmentation systems  
- Fine-tune models on noisy OCR outputs or user-generated content  

---

## 🧑‍💻 Authors

- **Zephyra Collective**   
  Feel free to reach out or contribute via pull requests.

---

## 📬 Feedback or Issues?

Open a GitHub [issue](https://github.com/your-repo/issues) or contact the author for bugs, feature requests, or deployment help.

> ⚡ *"From knowledge graphs to entropy graphs – a better way to learn sentence breaks."*

