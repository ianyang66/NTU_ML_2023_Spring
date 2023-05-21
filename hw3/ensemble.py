# Reference:　https://docs.python.org/zh-tw/3/library/collections.html
import pandas as pd
from collections import Counter

path_1 = 'submission831.csv'
path_2 = 'submission83.csv'
path_3 = 'submission822.csv'
path_4 = 'submission80-1.csv'
path_5 = 'submission80-2.csv'
path_6 = 'submission80-3.csv'
path_7 = 'submission814.csv'
path_8 = 'submission807.csv'
path_9 = 'submission805.csv'
path_10 = 'submission798.csv'
path_11 = 'submission789.csv'
path_12 = 'submission769.csv'
path_13 = 'submission755.csv'

def append_in(path):
    arr = []
    with open(path) as f1:
        for lines in f1:
            if (lines != 'Id,Category\n'):
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
arr_9 = append_in(path_9)
arr_10 = append_in(path_10)
arr_11 = append_in(path_11)
arr_12 = append_in(path_12)
arr_13 = append_in(path_13)

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
    temp.append(arr_9[i])
    temp.append(arr_10[i])
    temp.append(arr_11[i])
    temp.append(arr_12[i])
    temp.append(arr_13[i])
    classification = Counter(temp)
    pred.append(classification.most_common(1)[0][0])

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(0, len(pred))]
df["Category"] = pred
df.to_csv("submission_ensemble13.csv", index = False)