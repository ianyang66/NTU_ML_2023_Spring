python train_v2.py \
--train_ratio 0.9 \
--seed 1124 \
--batch_size 100 \
--num_epoch 300 \
--learning_rate 1e-3 \
--hidden_dim 35 \
--hidden_layers 15 \
--dropout 0.2 \
--fc_dropout 0.4 \
--model gru \
--extra_name 3

python test_v2.py \
--hidden_dim 35 \
--hidden_layers 15 \
--dropout 0.2 \
--fc_dropout 0.4 \
--model gru \
--extra_name 3