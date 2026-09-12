# Indonesian Emotion Detection with OpenAI o4-mini Text Normalization

This repository contains the implementation and data used in the study:

**Improving Indonesian Emotion Detection with OpenAI o4-mini Text Normalization**

The study evaluates the effect of OpenAI o4-mini text normalization on Indonesian emotion classification using:

- L1 Logistic Regression with TF-IDF
- IndoBERT fine-tuning
- IndoBERTweet fine-tuning
- 70/30 stratified train-test split
- AdamW optimizer with a learning rate of `3e-6`
- 40 training epochs
- Cross-entropy loss
- PCA analysis of BERT embeddings
- SVD analysis of TF-IDF representations
- Silhouette score evaluation

## Repository structure

```text
.
├── data/
│   ├── raw/
│   └── normalized/
├── results/
├── emotion_detection_o4_normalization.ipynb
├── requirements.txt
└── README.md
```

## Data

The `raw` directory contains the original preprocessed Indonesian Twitter emotion dataset.

The `normalized` directory contains the text data produced through the OpenAI o4-mini normalization process.

The dataset contains six emotion classes:

- Sadness
- Anger
- Joy
- Love
- Fear
- Neutral

## Emotion classification

The experiment compares classification performance between raw and o4-mini normalized text.

For the conventional machine-learning baseline, TF-IDF representations are classified using L1-regularized Logistic Regression.

IndoBERT and IndoBERTweet are fine-tuned for emotion classification using a 70/30 stratified train-test split, AdamW with a learning rate of `3e-6`, cross-entropy loss, and 40 training epochs.

Classification performance is evaluated using precision, recall, and F1-score.

## Representation analysis

The learned representations are analyzed in a low-dimensional space to examine class separability.

- PCA is applied to IndoBERT and IndoBERTweet embeddings.
- SVD is applied to sparse TF-IDF representations.
- Silhouette score is used to quantify separation between emotion classes.

## Environment

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

Run the experiment with:

```bash
jupyter notebook emotion_detection_o4_normalization.ipynb
```

GPU acceleration is strongly recommended for Transformer fine-tuning.
