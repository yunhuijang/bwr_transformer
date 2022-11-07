python train_smiles_lstm_generator.py \
--dataset_name qm9 \
--num_layers 3 \
--num_samples 10000 \
--max_epochs 300 \
--check_sample_every_n_epoch 20 \
--string_type smiles \
--group char_rnn