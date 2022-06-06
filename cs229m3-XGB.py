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
from matplotlib import pyplot
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
for i in range(10000):
    labels.append([trueLabels[i], weakLabels[i]])
#1102493
vectorsPost = pd.DataFrame(index=np.arange(10000), columns=np.arange(200))
for i in range(10000):
    cur = vectorsPre[i].split(":")
    cur.pop()
    curVect = [float(item) for item in cur]
    vectorsPost.iloc[i, :] = curVect
print('vectorsPost')

x_train,x_test,y_train,y_test=train_test_split(vectorsPost, labels, test_size = 0.20, random_state=5)
print('split')

x_train_list = x_train.values.tolist()

y_test_true = []
y_train_true = []
y_train_weak = []

for i in range(len(y_test)):
    y_test_true.append(y_test[i][0])

for i in range(len(y_train)):
    y_train_true.append(y_train[i][0])
    y_train_weak.append(y_train[i][1])

counterTrue = Counter(y_train_true)
scaleTrue = counterTrue[0] / counterTrue[1]
counterWeak = Counter(y_train_weak)
scaleWeak = counterWeak[0] / counterWeak[1]
print('counter stuff')

model_true = XGBClassifier(scale_pos_weight=scaleTrue, verbosity=2)
model_weak = XGBClassifier(scale_pos_weight=scaleWeak, verbosity=2)

print('fitting')
evalset1 = [(x_train_list, y_train_true)]
model_true.fit(x_train_list, y_train_true, eval_metric='logloss', eval_set=evalset1)
results1 = model_true.evals_result()
evalset = [(x_train_list, y_train_weak)]
model_weak.fit(x_train_list, y_train_weak, eval_metric='logloss', eval_set=evalset)
results = model_weak.evals_result()
pyplot.plot(results['validation_0']['logloss'], label='weak')
pyplot.plot(results1['validation_0']['logloss'], label='true')
pyplot.legend()
pyplot.show()
print('plot done')
print('fit done')

oodLabels = oodRatings

oodVectorsPost = pd.DataFrame(index=np.arange(105261), columns=np.arange(200))
for i in range(105261):
    cur = oodVectorsPre[i].split(":")
    cur.pop()
    curVect = [float(item) for item in cur]
    oodVectorsPost.iloc[i, :] = curVect
print('oodVectorsPost')




print('rearranged')






print('fit')

x_test_list = x_test.values.tolist()
oodVectorsPost_list = oodVectorsPost.values.tolist()

y_pred_true = model_true.predict(x_test_list)
y_pred_weak = model_weak.predict(x_test_list)

y_pred_ood_true = model_true.predict(oodVectorsPost_list)
y_pred_ood_weak = model_weak.predict(oodVectorsPost_list)

y_pred_true_prob = model_true.predict_proba(x_test_list)
y_pred_weak_prob = model_weak.predict_proba(x_test_list)

y_pred_ood_true_prob = model_true.predict_proba(oodVectorsPost_list)
y_pred_ood_weak_prob = model_weak.predict_proba(oodVectorsPost_list)

true_file = open('10ktrue_pred_probaXGB.txt', 'w')
weak_file = open('10kweak_pred_probaXGB.txt', 'w')
trueOOD_file = open('10ktrueOOD_pred_probaXGB.txt', 'w')
weakOOD_file = open('10kweakOOD_pred_probaXGB.txt', 'w')

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

filenameTrue = '10kfinalized_model_trueXGB.sav'
pickle.dump(model_true, open(filenameTrue, 'wb'))
filenameWeak = '10kfinalized_model_weakXGB.sav'
pickle.dump(model_weak, open(filenameWeak, 'wb'))

ylabels = open('10klabels.txt', 'w')
for i in range(len(y_test_true)):
    ylabels.write(str(y_test_true[i]) + ',' + str(y_pred_true[i]))

print('pred')
acc_true = model_true.score(x_test_list, y_test_true)
acc_weak = model_weak.score(x_test_list, y_test_true)
print(acc_true, acc_weak)

acc_true_ood = model_true.score(oodVectorsPost_list, oodLabels)
acc_weak_ood = model_weak.score(oodVectorsPost_list, oodLabels)
print(acc_true_ood, acc_weak_ood)