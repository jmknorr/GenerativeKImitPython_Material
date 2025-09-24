# %% packages
from transformers import pipeline

# %% Modelabfrage
# Das Model wird lokal gespeichert
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment_task("Covid cases are increasing fast!")
