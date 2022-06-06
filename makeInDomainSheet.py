import pandas as pd
import csv
from sklearn.metrics import f1_score

filename = "20%Of10kofBook_testedOn_10kmodel2.csv"
outputfile = "20%Of10kofBook_testedOn_10kmodel2Sheet.csv"
trueratingfile = '10k20%vectors.csv'
df = pd.read_csv(filename)
vectors = df['predictionVector'].tolist()
pred_rating = df['predictionVector'].tolist()

for i in range(len(vectors)):
    astring = '"'
    astring += str(vectors[i])
    astring += '"'
    vectors[i] = astring

    nums = vectors[i].split(' ')
    num1 = nums[0]
    num2 = nums[1]
    num1 = num1.replace('[', '')
    num2 = num2.replace(']', '')
    num1 = num1.replace('"', '')
    num2 = num2.replace('"', '')

    if(float(num1) > 0.5):
        pred_rating[i] = '0'
    else:
        pred_rating[i] = '1'

df2 = pd.read_csv(trueratingfile)
overallrating = df2['true_rating'].tolist()

""" for i in range(len(overallrating)):
    if overallrating[i] == 1:
        overallrating[i] = '0'
    elif overallrating[i] == 2:
        overallrating[i] = '0'
    elif overallrating[i] == 3:
        overallrating[i] = '0'
    elif overallrating[i] == 4:
        overallrating[i] = '1'
    elif overallrating[i] == 5:
        overallrating[i] = '1' """

dftotal = pd.DataFrame(data={"predictionVector": vectors,  "trueRating": overallrating, "predictedRating": pred_rating})
dftotal.to_csv(outputfile, quoting = csv.QUOTE_NONE, escapechar = ' ', index=False)