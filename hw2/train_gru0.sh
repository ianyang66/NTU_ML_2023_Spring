python train_v2.py \
--train_ratio 0.9 \
--seed 19990910 \
--batch_size 50 \
--num_epoch 300 \
--learning_rate 1e-3 \
--hidden_dim 35 \
--hidden_layers 12 \
--dropout 0.2 \
--fc_dropout 0.4 \
--model gru \
--extra_name 0

python test_v2.py \
--hidden_dim 35 \
--hidden_layers 12 \
--dropout 0.2 \
--fc_dropout 0.4 \
--model gru \
--extra_name 0
