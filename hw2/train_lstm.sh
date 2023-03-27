python train_v2.py \
--train_ratio 0.9 \
--seed 19991124 \
--batch_size 128 \
--num_epoch 450 \
--learning_rate 1e-3 \
--hidden_dim 35 \
--hidden_layers 12 \
--dropout 0.25 \
--fc_dropout 0.4 \
--model lstm \
--extra_name 6

python test_v2.py \
--hidden_dim 35 \
--hidden_layers 12 \
--dropout 0.25 \
--fc_dropout 0.4 \
--model lstm \
--extra_name 6