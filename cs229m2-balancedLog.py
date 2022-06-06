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

bookDataVectors = pd.read_csv('bookVectors-200.csv')
bookDataLabels = pd.read_csv('book_reviews_weaklabel.csv')
oodDataVectors = pd.read_csv('oodVectors-200-new.csv')
weakLabels = bookDataLabels['weak_labels']
trueLabels = bookDataLabels['true_label']
vectorsPre = bookDataVectors['vectors']
oodVectorsPre = oodDataVectors['vectors']
oodRatings = oodDataVectors['rating']

labels = []
for i in range(len(trueLabels)):
    labels.append([trueLabels[i], weakLabels[i]])

vectorsPost = pd.DataFrame(index=np.arange(1102493), columns=np.arange(200))
for i in range(1102493):
    cur = vectorsPre[i].split(":")
    cur.pop()
    curVect = [float(item) for item in cur]
    vectorsPost.iloc[i, :] = curVect
print('vectorsPost')

oodLabels = oodRatings
""" for i in range(len(oodRatings)):
    curRating = float(oodRatings[i])
    if (curRating > 3):
        oodLabels.append(1)
    else:
        oodLabels.append(0) """

oodVectorsPost = pd.DataFrame(index=np.arange(105261), columns=np.arange(200))
for i in range(105261):
    cur = oodVectorsPre[i].split(":")
    cur.pop()
    curVect = [float(item) for item in cur]
    oodVectorsPost.iloc[i, :] = curVect
print('oodVectorsPost')

x_train,x_test,y_train,y_test=train_test_split(vectorsPost, labels, test_size = 0.20, random_state=5)
print('split')

y_test_true = []
y_train_true = []
y_train_weak = []

for i in range(len(y_test)):
    y_test_true.append(y_test[i][0])

for i in range(len(y_train)):
    y_train_true.append(y_train[i][0])
    y_train_weak.append(y_train[i][1])

print('rearranged')
model_true = LogisticRegression(class_weight='balanced')
model_weak = LogisticRegression(class_weight='balanced')

model_true.fit(x_train, y_train_true)
model_weak.fit(x_train, y_train_weak)

print('fit')

y_pred_true = model_true.predict(x_test)
y_pred_weak = model_weak.predict(x_test)

y_pred_ood_true = model_true.predict(oodVectorsPost)
y_pred_ood_weak = model_weak.predict(oodVectorsPost)

y_pred_true_prob = model_true.predict_proba(x_test)
y_pred_weak_prob = model_weak.predict_proba(x_test)

y_pred_ood_true_prob = model_true.predict_proba(oodVectorsPost)
y_pred_ood_weak_prob = model_weak.predict_proba(oodVectorsPost)

true_file = open('true_pred_proba2.txt', 'w')
weak_file = open('weak_pred_proba2.txt', 'w')
trueOOD_file = open('trueOOD_pred_proba2.txt', 'w')
weakOOD_file = open('weakOOD_pred_proba2.txt', 'w')

for i in range(len(y_pred_true_prob)):
    true_file.write(str(y_pred_true_prob[i]))
    weak_file.write(str(y_pred_weak_prob[i]))
for i in range(len(y_pred_ood_true_prob)):
    trueOOD_file.write(str(y_pred_ood_true_prob[i]))
    weakOOD_file.write(str(y_pred_ood_weak_prob[i]))

true_file.close()
weak_file.close()
trueOOD_file.close()
weakOOD_file.close()

filenameTrue = 'finalized_model_true2.sav'
pickle.dump(model_true, open(filenameTrue, 'wb'))
filenameWeak = 'finalized_model_weak2.sav'
pickle.dump(model_weak, open(filenameWeak, 'wb'))

print('pred')
acc_true = model_true.score(x_test, y_test_true)
acc_weak = model_weak.score(x_test, y_test_true)
print(acc_true, acc_weak)

acc_true_ood = model_true.score(oodVectorsPost, oodLabels)
acc_weak_ood = model_weak.score(oodVectorsPost, oodLabels)
print(acc_true_ood, acc_weak_ood)