import pandas as pd
import csv
from sklearn.metrics import f1_score
from sklearn import metrics

def makesheetOOD(filename, outputname):
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

    df2 = pd.read_csv('ood_categories.csv')
    categories = df2['category'].tolist()

    df = pd.read_csv('oodVectors-200-new.csv')
    overallrating = df['rating'].tolist()

    dftotal = pd.DataFrame(data={"predictionVector": vectors , "category": categories,  "trueRating": overallrating, "predictedRating": pred_rating})
    dftotal.to_csv(outputname, quoting = csv.QUOTE_NONE, escapechar = ' ', index=False)


def statCalcOOD(filename):
    df = pd.read_csv(filename)
    categories = df['category'].tolist()

    unique_categories = ['Video_Games', 'Luxury_Beauty', 'Patio_Lawn_and_Garden', 'Prime_Pantry', 'Musical_Instruments', 'Digital_Music', 'Software', 'Automotive', 'Industrial_and_Scientific', 'Cell_Phones_and_Accessories', 'All_Beauty', 'Grocery_and_Gourmet_Food', 'Electronics', 'Gift_Cards', 'Office_Products', 'CDs_and_Vinyl', 'Tools_and_Home_Improvement', 'Home_and_Kitchen', 'Sports_and_Outdoors', 'Arts_Crafts_and_Sewing', 'Clothing_Shoes_and_Jewelry', 'Magazine_Subscriptions', 'Toys_and_Games', 'Pet_Supplies']
    f1_scorevals = {}
    roc_vals = {}

    df = df.reset_index()  # make sure indexes pair with number of rows
    for i in unique_categories:
        predictedRating = []
        trueRating = []
        for index, row in df.iterrows():
            if(row['category'] == i):
                predictedRating.append(row['predictedRating'])
                trueRating.append(row['trueRating'])
        f1_scorevals[i] = f1_score(trueRating, predictedRating)
        try:
            roc_vals[i] = metrics.roc_auc_score(trueRating, predictedRating)
        except ValueError:
            pass

    print(f1_scorevals)
    print(roc_vals)

def statCalcInDomain(filename):
    df = pd.read_csv(filename)
    true = df['trueRating'].tolist()
    predict = df['predictedRating'].tolist()

    f1_scorevals = f1_score(true, predict)
    print(f1_scorevals)

    rocVal = metrics.roc_auc_score(true, predict)
    print("rocval: ", rocVal)