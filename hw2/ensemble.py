# Reference:　https://docs.python.org/zh-tw/3/library/collections.html
import pandas as pd
from collections import Counter

path_6 = './prediction0.csv'
path_5 = './prediction1.csv'
path_4 = './prediction2.csv'
path_3 = './prediction3.csv'
path_2 = './prediction4.csv'
path_1 = './prediction5.csv'
path_7 = './prediction6.csv'
path_8 = './prediction.csv'

def append_in(path):
    arr = []
    with open(path) as f1:
        for lines in f1:
            if (lines != 'Id,Class\n'):
                arr.append(lines.split(',')[-1].split('\n')[0])
    return arr

arr_1 = append_in(path_1)
arr_2 = append_in(path_2)
arr_3 = append_in(path_3)
arr_4 = append_in(path_4)
arr_5 = append_in(path_5)
arr_6 = append_in(path_6)
arr_7 = append_in(path_7)
arr_8 = append_in(path_8)

pred = []
for i in range(len(arr_1)):
    temp = []
    temp.append(arr_1[i])
    temp.append(arr_2[i])
    temp.append(arr_3[i])
    temp.append(arr_4[i])
    temp.append(arr_5[i])
    temp.append(arr_6[i])
    temp.append(arr_7[i])
    temp.append(arr_8[i])
    classification = Counter(temp)
    pred.append(classification.most_common(1)[0][0])

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(0, len(pred))]
df["Class"] = pred
df.to_csv("prediction_ensemble8.csv", index = False)
