# Machine Learning hw3, 2023 Spring

1. Train & Inference
    分別執行 `80-1.ipynb`, `80-2.ipynb`, `80-3.ipynb`, `83.ipynb`, `755.ipynb`, `769.ipynb`, `789.ipynb`, `798.ipynb`, `805.ipynb`, `807.ipynb`, `814.ipynb`, `822.ipynb`, `831.ipynb` 這13個jupyter notebook

   

2. Ensemble
    Run
    ```python ensemble.py```

Note1. Train 之前 Dataset要放在像這樣的路徑

![image-20230327135849860](C:\Users\wealt\AppData\Roaming\Typora\typora-user-images\image-20230327135849860.png)



Note2. `get_norm_v1.py` and `get_norm_v2.py`是我嘗試過為train圖片算平均normalize的方法，在部分Train的程式碼中transform有採用 `get_norm_v2.py`所得之結果做normalization，`get_norm_v1.py`則因訓練起來效果太差而棄用，研判是因為此方法是直接對圖片轉RGB後計算平均與標準差，而 `get_norm_v2.py` 則是圖片 transform 後計算平均與標準差。

Note3. question.ipynb 就是回答gradescore問題的code