# Machine Learning hw2, 2023 Spring

因為要ensemble，本來的notebook太亂，所以參考ADL作法，改為Script。

本次使用兩種前處理方式：
1. 原始sample code
2. 拔掉concat，自己寫一個collate function 

執行run.sh，即可從訓練所有模型到產生ensemble後的prediction file完整執行 (hint:要訓練蠻久的，因為我的GPU是Laptop版的，所以batch size調很小，一個rnn系列的model要訓練5~7個小時以上:cry:，總共有七個rnn系列加一個nn的模型去ensemble，所以大約要45小時)

執行以下指令即可訓練
```bash run.sh```