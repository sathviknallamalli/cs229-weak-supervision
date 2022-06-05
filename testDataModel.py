import pickle
import pandas as pd
import numpy as np


pickled_model = pickle.load(open('finalized_model_true.sav', 'rb'))
pickled_model2 = pickle.load(open('finalized_model_weak.sav', 'rb'))

pickled_model3 = pickle.load(open('model2_10k.sav', 'rb'))
pickled_model4 = pickle.load(open('model2_100k.sav', 'rb'))

oodData = pd.read_csv('oodVectors-200-new.csv')
vectorsPre = oodData['vectors']

oodVectorsPost = pd.DataFrame(index=np.arange(105261), columns=np.arange(200))
for i in range(105261):
    cur = vectorsPre[i].split(":")
    cur.pop()
    curVect = [float(item) for item in cur]
    oodVectorsPost.iloc[i, :] = curVect

""" bookDataVectors = pd.read_csv('10k20%vectors.csv')
vectorsPre = bookDataVectors['vectors']
bookDataVectorsLabels = pd.read_csv('labels_20%of100kofBooks.csv')
trueLabels = bookDataVectorsLabels['true_rating']

vectorsPost = pd.DataFrame(index=np.arange(2000), columns=np.arange(200))
for i in range(2000):
    cur = vectorsPre[i].split(":")
    #cur.pop()
    curVect = [float(item) for item in cur]
    vectorsPost.iloc[i, :] = curVect
print('vectorsPost')

preds = pickled_model3.predict_proba(vectorsPost.values.tolist())
perfect_weak = open('20%Of10kofBook_testedOn_100kmodel2.txt', 'w')
for i in range(len(preds)):
    perfect_weak.write(str(preds[i]))
perfect_weak.close() """

predictions = pickled_model3.predict_proba(oodVectorsPost.values.tolist())
preds_true = open('model2predOOD_10k.txt', 'w')
for i in range(len(predictions)):
    preds_true.write(str(predictions[i]))

preds_true.close()

predictions = pickled_model4.predict_proba(oodVectorsPost.values.tolist())
preds_weak = open('model2predOOD_100k.txt', 'w')
for i in range(len(predictions)):
    preds_weak.write(str(predictions[i]))

preds_weak.close()
