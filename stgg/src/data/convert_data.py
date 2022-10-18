from deepsmiles import Converter
import os
from pathlib import Path
import pickle
import selfies as sf
from deepsmiles import Converter
from joblib import Parallel, delayed
import re
import pandas as pd
import numpy as np
import json


# data_name = 'zinc'
# # python
# DATA_DIR = "../resource/data"
# GDSS_DATA_DIR = f"../../GDSS/data/"

# #debugging
# DATA_DIR = "stgg/resource/data"
# GDSS_DATA_DIR = f"GDSS/data/"

# raw_dir = f"{DATA_DIR}/{data_name}"
# tokens_set = set()

# filtered_df = pd.DataFrame()
# valid_len = 0
# for split in ['test', 'train']:
#     smiles_list_path = os.path.join(raw_dir, f"{split}.txt")
#     stgg_smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
#     # stgg_smiles_list =[string.split(',')[1] for string in stgg_smiles_list][1:]
#     gdss_df = pd.read_csv(GDSS_DATA_DIR + f'{data_name}250k.csv', header=0)
#     gdss_smiles_list = gdss_df['smiles'].tolist()
#     if split == 'test':
#         valid_len = len(stgg_smiles_list)
#         print(valid_len)
#     filtered_df = pd.concat([filtered_df, gdss_df.loc[gdss_df['smiles'].isin(stgg_smiles_list)]])
#     filtered_df.reset_index()

# filtered_df = filtered_df.reset_index(drop=True).drop(['Unnamed: 0'], axis=1)
# filtered_df.to_csv(GDSS_DATA_DIR + f'{data_name}_new.csv', index=True)
# with open(f'{GDSS_DATA_DIR}valid_idx_{data_name}_new.json', 'w') as f:
#     json.dump({'valid_idxs': [str(num).zfill(6) for num in np.arange(0,valid_len)]}, f)
# 
save_interval = 10
check_sample_every_n_epoch = 10
for epoch in np.arange(0,100,1):
    if epoch % check_sample_every_n_epoch == check_sample_every_n_epoch-1:
        print(epoch)