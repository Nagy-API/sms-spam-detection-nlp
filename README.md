# SMS Spam Detection NLP

This project classifies English SMS-style text as `spam` or `ham` (legitimate) and returns a spam probability. The local web interface, branded ShieldMail, accepts pasted text and `.txt` files. Its interface uses email terminology, while the training and evaluation data consists of SMS messages.

## Project structure

| Path | Purpose |
| --- | --- |
| `app/main.py` | FastAPI service. It cleans and tokenizes text, removes English stop words, and lemmatizes tokens using NLTK. A unigram and bigram `CountVectorizer` feeds a multinomial Naive Bayes classifier. |
| `frontend/` | Static HTML, CSS, and JavaScript that calls the local API. |
| `models/` | Project-generated scikit-learn classifier and vectorizer, saved with joblib. |
| `notebook/sms_spam_detection.ipynb` | Data exploration, preprocessing, training experiments, and evaluation on the SMS dataset. |
| `requirements.txt` | Python dependencies for the API. |

## Install and run locally

Use Python 3.11. From the repository root, create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with `source .venv/bin/activate`. Install the dependencies and NLTK resources in the active environment. The saved artifacts were serialized with scikit-learn 1.8.0.

```sh
python -m pip install -r requirements.txt "scikit-learn==1.8.0"
python -m nltk.downloader stopwords punkt_tab averaged_perceptron_tagger_eng wordnet
```

In the first terminal, start the API from `app/` so its relative paths resolve.

```sh
cd app
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In a second terminal with the environment active, run this command from the repository root and open [the local interface](http://127.0.0.1:8080).

```sh
python -m http.server 8080 --bind 127.0.0.1 --directory frontend
```

The API exposes `GET /` for a health check and `POST /predict` for a JSON body with a `text` field. The response includes `label` and `spam_probability`. [Interactive API documentation](http://127.0.0.1:8000/docs) is available when the API is running.

## Dataset and retraining

The included model files let the API run without the dataset. For retraining or running the notebook, download `spam.csv` from the [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it at `data/spam.csv`, creating `data/` if needed. The project expects the CSV's `v1` and `v2` columns and reads it as Latin-1. The exact CSV is kept local because the Kaggle listing does not identify a redistribution license and the messages include contact details.

The original SMS Spam Collection is attributed to Tiago Almeida and Jos Hidalgo (2011) and is distributed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). UCI provides the corpus in a different file format. The project reads the Kaggle CSV representation and transforms its SMS text into unigram and bigram counts for training. [UCI dataset citation](https://doi.org/10.24432/C5CC84)

To run the notebook, install the additional packages and start Jupyter from `notebook/`.

```sh
python -m pip install jupyterlab matplotlib seaborn imbalanced-learn
cd notebook
jupyter lab sms_spam_detection.ipynb
```

The API trains and saves both artifacts on startup when either model file is missing. For a fresh training run, use a separate checkout with the CSV in place and an empty `models/` directory so you can preserve the supplied files.
