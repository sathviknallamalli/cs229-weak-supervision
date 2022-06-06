import statistics

test_dict ={'Video_Games': 0.6821995568898385,
'Luxury_Beauty': 0.6531862745098038,
'Patio_Lawn_and_Garden': 0.6597242926120495,
'Prime_Pantry': 0.5869760003629435,
'Musical_Instruments': 0.6705586080586081,
'Digital_Music': 0.6246105919003115,
'Software': 0.4666666666666667,
'Automotive': 0.588865562966703,
'Industrial_and_Scientific': 0.5548309178743961,
'Cell_Phones_and_Accessories': 0.628245145891829,
'All_Beauty': 0.8076923076923077,
'Grocery_and_Gourmet_Food': 0.6117641304614325,
'Electronics': 0.641843423833338,
'Office_Products': 0.6731062391953481,
'CDs_and_Vinyl': 0.6555478889881032,
'Tools_and_Home_Improvement': 0.65454694043745,
'Home_and_Kitchen': 0.6614875621861517,
'Sports_and_Outdoors': 0.63630206119847,
'Arts_Crafts_and_Sewing': 0.6911290054237513,
'Clothing_Shoes_and_Jewelry': 0.6573324261190392,
'Toys_and_Games': 0.6492465399772496,
'Pet_Supplies': 0.6145690458713825}

res = 0

for val in test_dict.values():
    res += val

res = res / len(test_dict)
print("The computed mean : " + str(res)) 
print("Standard Deviation of sample is % s " % (statistics.stdev(test_dict.values())))
