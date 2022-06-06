import pandas as pd
from snorkel.labeling import labeling_function, LabelingFunction, PandasLFApplier
from snorkel.labeling.model import LabelModel
from numpy import loadtxt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline

positive_keywords = loadtxt("data/positive_words.txt", dtype="str")
negative_keywords = loadtxt("data/negative_words.txt", dtype="str")
sid_obj = SentimentIntensityAnalyzer()
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@labeling_function()
def positive_search_text(df_row):
    if any(keyword in str(df_row['reviewText']).lower() for keyword in positive_keywords):
        return 1
    return -1

@labeling_function()
def negative_search_text(df_row):
    if any(keyword in str(df_row['reviewText']).lower() for keyword in negative_keywords):
        return 0
    return -1

@labeling_function()
def positive_search_summary(df_row):
    if any(keyword in str(df_row['summary']).lower() for keyword in positive_keywords):
        return 1
    return -1

@labeling_function()
def negative_search_summary(df_row):
    if any(keyword in str(df_row['summary']).lower() for keyword in negative_keywords):
        return 0
    return -1

@labeling_function()
def vader_sentiment(df_row):
    sentiment_dict = sid_obj.polarity_scores(df_row['reviewText'])
    if sentiment_dict['compound'] >= 0.05:
        return 1
    elif sentiment_dict['compound'] <= -0.05:
        return 0
    return -1

@labeling_function()
def text_blob(df_row):
    classifier = TextBlob(df_row['reviewText'])
    if classifier.sentiment.polarity >= 0.05:
        return 1
    elif classifier.sentiment.polarity <= -0.05:
        return 0
    return -1

@labeling_function()
def transformer(df_row):
    classifier = sentiment_pipeline(df_row['reviewText'])
    if classifier[0]['label'] == 'LABEL_0':
        return 0
    elif classifier[0]['label'] == 'LABEL_2':
        return 1
    return -1


book_reviews = pd.read_csv("data/book_reviews_labels.csv")
lfs = [positive_search_text, negative_search_text, positive_search_summary, negative_search_summary, vader_sentiment, text_blob]

applier = PandasLFApplier(lfs=lfs)
L_predictions = applier.apply(book_reviews)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_predictions)
y_preds = label_model.predict(L_predictions, tie_break_policy='random')

book_reviews['bunch_weak'] = y_preds
book_reviews.to_csv('data/book_reviews_labels.csv', index=False)
