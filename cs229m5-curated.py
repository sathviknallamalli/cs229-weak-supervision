import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBClassifier
from collections import Counter

bookDataVectors = pd.read_csv('bookVectors-200.csv')
bookDataLabels = pd.read_csv('book_reviews_labels.csv')
oodDataVectors = pd.read_csv('oodVectors-200-new.csv')
weakLabels = bookDataLabels['bunch_weak']
trueLabels = bookDataLabels['true_label']
vectorsPre = bookDataVectors['vectors']
oodVectorsPre = oodDataVectors['vectors']
oodRatings = oodDataVectors['rating']

labels = []
for i in range(100000):
    labels.append([trueLabels[i], weakLabels[i]])

vectorsPost = pd.DataFrame(index=np.arange(100000), columns=np.arange(200))
for i in range(100000):
    cur = vectorsPre[i].split(":")
    cur.pop()
    curVect = [float(item) for item in cur]
    vectorsPost.iloc[i, :] = curVect
print('vectorsPost')


oodVectorsPost = pd.DataFrame(index=np.arange(105261), columns=np.arange(200))
for i in range(105261):
    cur = oodVectorsPre[i].split(":")
    cur.pop()
    curVect = [float(item) for item in cur]
    oodVectorsPost.iloc[i, :] = curVect
print('oodVectorsPost')

x_train,x_test,y_train,y_test=train_test_split(vectorsPost, labels, test_size = 0.20, random_state=5)
print('split')

y_train_true = []
y_test_true = []
for i in range(len(y_train)):
    y_train_true.append(y_train[i][0])
for i in range(len(y_test)):
    y_test_true.append(y_test[i][0])

x_test.to_csv('100k20%vectors', sep='\t')

print("THINGS R WRITTEN")

#model1 training
counter1 = Counter(oodRatings)
scale1 = counter1[0] / counter1[1]
model1 = XGBClassifier(scale_pos_weight=scale1, verbosity=2)
model1.fit(oodVectorsPost.values.tolist(), oodRatings.values.tolist())
print('model 1 fit')

model1file = 'model1_100k.sav'
pickle.dump(model1, open(model1file, 'wb'))

#model1 testing
model1_pred = model1.predict(x_train.values.tolist())
print('model 1 test')

#model2 training
counter2 = Counter(model1_pred)
scale2 = counter1[0] / counter1[1]
model2 = XGBClassifier(scale_pos_weight=scale2, verbosity=2)
model2.fit(x_train.values.tolist(), model1_pred)
print('model 2 fit')

#model2 testing
model2_pred = model2.predict(oodVectorsPost.values.tolist())
model2_pred_proba = model2.predict_proba(oodVectorsPost.values.tolist())
model2file = open('model2pred_100k.txt', 'w')
for i in range(len(model2_pred_proba)):
    model2file.write(str(model2_pred_proba[i]))
print('model 2 test')

filenameTrue = 'model2_100k.sav'
pickle.dump(model2, open(filenameTrue, 'wb'))
