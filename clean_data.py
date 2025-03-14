
import re
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, \
    strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short

FILTERS = [lambda x: x.lower(), strip_punctuation, strip_multiple_whitespaces,
           strip_numeric, remove_stopwords, strip_short]


def process(text):
    return preprocess_string(text, filters=FILTERS)

print("hello world")
print("hello world")
print("hello world")
print("hello world")
print("hello world")
print("hello world")
print("hello world")
print("hello world")


print("hello world")
print("hello world")
print("hello world")
print("hello world")
print("hello world")


def label(score):
    if score >= 3.5:
        return 1
    else:
        return 0


# CREATE THE TRAINING DATASET WITH ONLY BOOK REVIEWS
'''
training_df = None
counter = 0
for chunk in pd.read_csv("data/reviews.csv", chunksize=100000):
    counter += 1
    if counter % 10 == 0:
        print(counter)
    category = chunk["category"]
    indices = [i + 100000 * (counter - 1) for i in range(len(category)) if category[i + 100000 * (counter - 1)] == "Books"]

    if len(indices) > 0:
        trainingIndices = sample(indices, int(len(indices) / 5.))
        if training_df is None:
            training_df = chunk.loc[trainingIndices]
        else:
            training_df = pd.concat([training_df, chunk.loc[trainingIndices]])

training_df = training_df.drop(columns=['reviewerID', 'asin', 'reviewTime', 'unixReviewTime', 'verified', 'reviewYear'])
training_df.to_csv('data/book_reviews.csv', index=False)
'''

# CREATE THE TESTING DATASET WITHOUT BOOK REVIEWS
'''
testing_df = None
counter = 0
for chunk in pd.read_csv("data/amazon_v2.1/reviews.csv", chunksize=100000):
    counter += 1
    if counter % 10 == 0:
        print(counter)

    category = chunk["category"]
    indices = [i + 100000 * (counter - 1) for i in range(len(category))
               if category[i + 100000 * (counter - 1)] != "Books" and category[i + 100000 * (counter - 1)] != "Kindle_Store"
               and category[i + 100000 * (counter - 1)] != "Movies_and_TV"]

    if len(indices) > 0:
        trainingIndices = sample(indices, int(len(indices) / 30.))
        if testing_df is None:
            testing_df = chunk.loc[trainingIndices]
        else:
            testing_df = pd.concat([testing_df, chunk.loc[trainingIndices]])

testing_df = testing_df.drop(columns=['reviewerID', 'asin', 'reviewTime', 'unixReviewTime', 'verified', 'reviewYear'])
testing_df.to_csv('data/OOD_reviews.csv', index=False)
'''

# APPLY THE TEXT PROCESSING TO THE DATASETS
'''
book_reviews = pd.read_csv("data/book_reviews.csv")
book_reviews['processedText'] = book_reviews['reviewText'].apply(process)
book_reviews.to_csv('data/book_reviews.csv', index=False)

OOD_reviews = pd.read_csv("data/OOD_reviews.csv")
OOD_reviews['processedText'] = OOD_reviews['reviewText'].apply(process)
OOD_reviews.to_csv('data/OOD_reviews.csv', index=False)
'''

# LABEL THE DATASETS
'''
book_reviews = pd.read_csv("data/book_reviews.csv")
book_reviews['true_label'] = book_reviews['overall'].apply(label)
book_reviews.to_csv('data/book_reviews.csv', index=False)

OOD_reviews = pd.read_csv("data/OOD_reviews.csv")
OOD_reviews['true_label'] = OOD_reviews['overall'].apply(label)
OOD_reviews.to_csv('data/OOD_reviews.csv', index=False)
'''
