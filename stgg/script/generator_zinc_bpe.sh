python train_smiles_lstm_generator.py \
--dataset_name zinc \
--num_layers 3 \
--num_samples 10000 \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--string_type bpe_zinc \
--group bpe