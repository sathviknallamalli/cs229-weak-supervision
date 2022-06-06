from ntpath import join
import numpy as np
import pandas as pd
import csv
import ast
from random import sample
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open('glove.6B.200d.txt','r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def getVectors(text, model, outputString):
    msgVectors = []
    allSentences = text['processed_words'].tolist()
    ratings = text['rating'].tolist()
    category = text['category'].tolist()
    counter = 0
    for i in allSentences:
        words = i.split(' ')
        allSentenceVectors = [0] * 200
        for j in words:
            if j in model:
                allSentenceVectors += model[j]
        avgvector = np.multiply(allSentenceVectors, 1/len(words))
        astring  = '"'
        astring += ''.join(str(e) + ':' for e in avgvector)
        astring += '"'
        msgVectors.append(astring)
        counter += 1
    prefix = "msg"
    indexlist = []
    for i in range(len(msgVectors)):
        indexlist.append(prefix + str(i+1))
    print("about to add to csv")
    df = pd.DataFrame(data={"msgID": indexlist , "category": category, "rating": ratings, "vectors": msgVectors})
    df.to_csv("data/" + outputString + ".csv", quoting = csv.QUOTE_NONE, escapechar = ' ', index=False)

def load_csv(add_intercept=False):
    df = pd.read_csv('./OOD_reviews_new.csv', usecols= ['true_label', 'category','processedText',])
    df.rename(columns={'true_label': 'rating', 'processedText': 'processed_words'}, inplace=True)
    
    for index, row in df.iterrows():
        words = row[1].split(' ')
        runstring = ''
        for i in words:
            i = i.replace('[', '')
            i = i.replace(']', '')
            i = i.replace("'", '')
            i = i.replace(',', '')
            i = i.replace('.', '')
            runstring += i + ' '
        df.at[index,'processed_words'] = runstring
    df.to_csv('data/three_col_OOD_reviews.csv', quoting = csv.QUOTE_NONE, escapechar = ' ', index=False) 

    return df

def calc_tf_idf(messages):

    # *** START CODE HERE ***

    anotherlist = messages['y'].tolist()

    countvectorizer = CountVectorizer(analyzer= 'word', max_features=30, stop_words='english')
    tfidfvectorizer = TfidfVectorizer(analyzer='word',  max_features=30,stop_words= 'english')

    count_wm = countvectorizer.fit_transform(anotherlist)
    tfidf_wm = tfidfvectorizer.fit_transform(anotherlist)

    count_tokens = countvectorizer.get_feature_names()
    tfidf_tokens = tfidfvectorizer.get_feature_names()

    pref = "msg"
    indexlist = []
    for i in range(1000):
        indexlist.append(pref + str(i+1))

    df_countvect = pd.DataFrame(data = count_wm.toarray(),index = indexlist,columns = count_tokens)
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = indexlist,columns = tfidf_tokens)

    df_countvect.to_csv('data/countvect.csv',  index=False)
    df_tfidfvect.to_csv('data/tfidf.csv', quoting = csv.QUOTE_NONE, escapechar = ' ', index=False)

    print("Count Vectorizer\n")
    print(df_countvect)
    print("\nTD-IDF Vectorizer\n")
    print(df_tfidfvect)

    # *** END CODE HERE ***

def main():
    #processed_text = load_csv()
    #df = pd.read_csv('./data/three_col_OOD_reviews.csv')
    #glove_model = load_glove_model('glove.6B.200d.txt')
    #getVectors(df, glove_model, "oodVectors")

    df = pd.read_csv('./OOD_reviews_new.csv')
    print(df.shape[0])


if __name__ == "__main__":
    main()