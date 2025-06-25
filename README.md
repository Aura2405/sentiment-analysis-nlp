# Sentiment Analysis — NLP with TensorFlow and Hugging Face

This project contains two NLP models for sentiment analysis:

- A deep learning model built with TensorFlow/Keras
- A transformer-based sentiment classifier using Hugging Face

## Contents

- `sentiment.ipynb` – Jupyter Notebook with training, testing, and evaluation of both models.
- `hf_sentiment.py` – Standalone script using Hugging Face's pipeline.
- `sentiment_model.keras` – Saved Keras model (trained on preprocessed text data).
- `word_index.pkl` – Tokenizer word index used for text vectorization.
- `requirements.txt` – Python dependencies.
- `LICENSE`, `.gitignore`, `README.md`

## Model Details

- TensorFlow/Keras model:
  - Text vectorized using a Tokenizer (word_index stored in `word_index.pkl`)
  - Embedding layer + CNN or LSTM-based classifier
  - Trained on labeled sentiment dataset

- Hugging Face model:
  - Uses distilbert-base-uncased-finetuned-sst-2-english via 🤗 Transformers
  - Classifies input text with high accuracy

## How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-nlp.git
   cd sentiment-analysis-nlp

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Notebook:
   ```bash
   jupyter notebook sentiment.ipynb

4. Or run the Hugging Face script:
   ```bash
   python hf_sentiment.py

Notes:

1. The dataset is not included. You can use any CSV or TXT file with labeled text for training.

2. The .keras and .pkl files are essential for the deep learning model to run.

3. For inference, load the tokenizer and model like this:
   ```python
   from tensorflow.keras.models import load_model
   import pickle

   model = load_model("sentiment_model.keras")
   with open("word_index.pkl", "rb") as f:
       word_index = pickle.load(f)

