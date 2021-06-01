# Banking-FB-Sentiment-Analysis-Ro-BERT-

This repository is a Natural Language Procesing framework that uses the state of the art BERT model (trained on Romanian language).
This project intends to make use of a the Romanian-BERT model for analaysing customer sentiment for the Romanian Banking sector using Facebook data.

Steps:
* Scrapp bank's Facebook pages to extract customer comments (https://pypi.org/project/facebook-scraper/)
* Make use of the allready pre-traind Romanian-BERT model from https://github.com/dumitrescustefan/Romanian-Transformers using Huggingface.
* Furher train the model using labeled comments in order to build a binary clasiffier for Positive/Negative comments. The labeled comments where taken from https://github.com/katakonst/sentiment-analysis-tensorflow/tree/master/datasets and they where scrapped from e-MAG.
* Used the newly trianed Sentiment Analysis model to inference and quantify customer satisfaction on the extracted banking sector comments.
