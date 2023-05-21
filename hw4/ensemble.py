# Reference:　https://docs.python.org/zh-tw/3/library/collections.html
import pandas as pd
from collections import Counter

path_1 = 'output1.csv'
path_2 = 'output2.csv'
path_3 = 'output3.csv'
path_4 = 'output4.csv'

def append_in(path):
    arr = []
    arr0 = []
    with open(path) as f1:
        for lines in f1:
            if (lines != 'Id,Category\n'):
                arr.append(lines.split(',')[-1].split('\n')[0])
                arr0.append(lines.split(',')[0].split('\n')[0])
    return arr, arr0

arr_1, arr0_1 = append_in(path_1)
arr_2, arr0_2 = append_in(path_2)
arr_3, arr0_3 = append_in(path_3)
arr_4, arr0_4 = append_in(path_4)

pred = []
for i in range(len(arr_1)):
    temp = []
    temp.append(arr_2[i])
    temp.append(arr_1[i])
    temp.append(arr_3[i])
    temp.append(arr_4[i])
    classification = Counter(temp)
    pred.append(classification.most_common(1)[0][0])

df = pd.DataFrame()
df["Id"] = arr0_1
print(df)
df["Category"] = pred
df.to_csv("prediction-ensemble4.csv", index = False)