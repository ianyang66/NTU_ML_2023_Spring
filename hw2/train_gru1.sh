python train_v2.py \
--train_ratio 0.95 \
--seed 19991124 \
--batch_size 50 \
--num_epoch 300 \
--learning_rate 1e-3 \
--hidden_dim 35 \
--hidden_layers 15 \
--dropout 0.25 \
--fc_dropout 0.4 \
--model gru \
--extra_name 1

python test_v2.py \
--hidden_dim 35 \
--hidden_layers 15 \
--dropout 0.25 \
--fc_dropout 0.4 \
--model gru \
--extra_name 1