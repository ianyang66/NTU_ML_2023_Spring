# Machine Learning hw7, 2023 Spring

1. Train & Inference 12份result的csv
    分別照順序執行 `ml-hw7-v2.ipynb`, `ml-hw7-v3.ipynb`, `ml-hw7-v4.ipynb`, `ml-hw7-v6.ipynb`, `ml-hw7-v7.ipynb`, `ml-hw7-v11.ipynb`, `ml-hw7-v18.ipynb`, `ml-hw7-v19.ipynb`, `ml-hw7-pertbasemrc.ipynb`, `ml-hw7-tok-v4.ipynb`, `ml-hw7-tok-v5.ipynb`, `ml-hw7-epo2+1-v6.ipynb` 這12個jupyter notebook，會產出12份result.csv和12個model
    
    
    
2. Inference other 5 precision file from other tokenizer

    執行下面command去使用我擴充字典的tokenizer推論v2-v7的結果。

    ```
    bash infer_new5.sh
    ```

    

3. Ensemble
    執行`ensemble.ipynb`將前面產出的所有csv做ensemble產出最後的結果 `result-ensemble-2-17-post.csv`

ps.整個Reproduce流程約96小時

