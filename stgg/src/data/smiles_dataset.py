from rdkit import Chem
import enum
from deepsmiles import Converter
import selfies as sf
from joblib import Parallel, delayed
import sentencepiece as spm
from collections import defaultdict
import pandas as pd
import json
from math import log
import copy

PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    "#",
    "(",
    ")",
    "-",
    "/",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "=",
    "Br",
    "C",
    "Cl",
    "F",
    "I",
    "N",
    "O",
    "P",
    "S",
    "[C@@H]",
    "[C@@]",
    "[C@H]",
    "[C@]",
    "[CH-]",
    "[CH2-]",
    "[H]",
    "[N+]",
    "[N-]",
    "[NH+]",
    "[NH-]",
    "[NH2+]",
    "[NH3+]",
    "[O+]",
    "[O-]",
    "[OH+]",
    "[P+]",
    "[P@@H]",
    "[P@@]",
    "[P@]",
    "[PH+]",
    "[PH2]",
    "[PH]",
    "[S+]",
    "[S-]",
    "[S@@+]",
    "[S@@]",
    "[S@]",
    "[SH+]",
    "[n+]",
    "[n-]",
    "[nH+]",
    "[nH]",
    "[o+]",
    "[s+]",
    "\\",
    "c",
    "n",
    "o",
    "s",
]
TOKENS_SELFIES = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
# in QM9
'[#Branch1]', '[#Branch2]', '[#C]', '[#N]', 
'[/C]', '[/Cl]', '[/NH1+1]', '[/N]', '[/O-1]', '[/O]', '[/S-1]', 
'[/S]', '[=Branch1]', '[=Branch2]', '[=C]', '[=N+1]', '[=NH1+1]', 
'[=NH2+1]', '[=N]', '[=O]', '[=Ring1]', '[=Ring2]', '[=S@@]', 
'[=S]', '[Br]', '[Branch1]', '[Branch2]', '[C@@H1]', '[C@@]', 
'[C@H1]', '[C@]', '[C]', '[Cl]', '[F]', '[I]', '[N+1]', '[N-1]', 
'[NH1+1]', '[NH1-1]', '[NH1]', '[NH2+1]', '[NH3+1]', '[N]', '[O-1]', 
'[O]', '[P@@]', '[P]', '[Ring1]', '[Ring2]', '[S-1]', '[S@@]', '[S@]', 
'[S]', '[\\C]', '[\\N]', '[\\O-1]', '[\\O]', '[\\S]',
# in ZINC
'[PH1+1]', '[S@@+1]', '[=S@]', '[\\S@]', '[S+1]', '[=SH1+1]', '[\\N+1]', '[/O+1]', 
'[/C@@]', '[-\\Ring1]', '[/NH1-1]', '[\\NH1]', '[-/Ring2]', '[/NH1]', '[\\NH2+1]', 
'[/C@]', '[=O+1]', '[=P]', '[PH1]', '[\\S-1]', '[\\I]', '[/C@H1]', '[=PH2]', 
'[/C@@H1]', '[/F]', '[/N+1]', '[\\Br]', '[\\N-1]', '[\\Cl]', '[\\NH1+1]', '[\\F]', 
'[=P@@]', '[P@@H1]', '[P+1]', '[=S+1]', '[\\C@H1]', '[=N-1]', '[CH2-1]', '[-/Ring1]', 
'[CH1-1]', '[/NH2+1]', '[\\C@@H1]', '[/S@]', '[=OH1+1]', '[=P@]', '[/Br]', '[/N-1]', 
'[P@]', '[#N+1]']
TOKENS_DEEPSMILES = TOKENS + ['%20', '%22', '%28', '%12', 
'%15', '%13', '%10', '%16', '%11', 
'%14', '%21', '%18', '%17', '%19']
MAX_LEN = 250

# SPM tokens
sp_qm9 = spm.SentencePieceProcessor(model_file='../resource/tokenizer/qm9/qm9.model')
sp_zinc = spm.SentencePieceProcessor(model_file='../resource/tokenizer/zinc/zinc.model')
TOKENS_SPM_QM9 = [sp_qm9.IdToPiece(ids) for ids in range(sp_qm9.GetPieceSize())]
TOKENS_SPM_QM9.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN])
TOKENS_SPM_ZINC = [sp_zinc.IdToPiece(ids) for ids in range(sp_zinc.GetPieceSize())]
TOKENS_SPM_ZINC.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN])

# SPM_BPE tokens & merges
# zinc: token size 100 / qm9: token size 50 (but 36)
with open(f"../resource/tokenizer/qm9/merges_50.txt", 'r') as merges_file:
    merges = merges_file.read()
    merges_dict = eval(merges)
    MERGES_QM9 = {eval(key): value for key, value in merges_dict.items()}

with open(f"../resource/tokenizer/zinc/merges_100.txt", 'r') as merges_file:
    merges = merges_file.read()
    merges_dict = eval(merges)
    MERGES_ZINC = {eval(key): value for key, value in merges_dict.items()}

with open(f"../resource/tokenizer/qm9/tokens_50.txt", 'r') as tokens_file:
    tokens = tokens_file.readlines()
    TOKENS_BPE_QM9 = [token.rstrip('\n') for token in tokens]
    TOKENS_BPE_QM9.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN])

with open(f"../resource/tokenizer/zinc/tokens_100.txt", 'r') as tokens_file:
    tokens = tokens_file.readlines()
    TOKENS_BPE_ZINC = [token.rstrip('\n') for token in tokens]
    TOKENS_BPE_ZINC.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN])

# SPM_UNI tokens & models
# zinc: token size 100 / qm9: token size 100 (but 36)
with open(f"../resource/tokenizer/qm9/model_100.txt", 'r') as model_file:
    model = model_file.read()
    model_dict = eval(model)
    MODEL_QM9 = model_dict

with open(f"../resource/tokenizer/zinc/model_100.txt", 'r') as model_file:
    model = model_file.read()
    model_dict = eval(model)
    MODEL_ZINC = model_dict

with open(f"../resource/tokenizer/qm9/tokens_uni_100.txt", 'r') as tokens_file:
    tokens = tokens_file.readlines()
    TOKENS_UNI_QM9 = [token.rstrip('\n') for token in tokens]
    TOKENS_UNI_QM9.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN])

with open(f"../resource/tokenizer/zinc/tokens_uni_100.txt", 'r') as tokens_file:
    tokens = tokens_file.readlines()
    TOKENS_UNI_ZINC = [token.rstrip('\n') for token in tokens]
    TOKENS_UNI_ZINC.extend([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN])


@enum.unique
class TokenType(enum.IntEnum):
    ATOM = 1
    BOND = 2
    BRANCH_START = 3
    BRANCH_END = 4
    RING_NUM = 5
    SPECIAL = 6


ORGANIC_ATOMS = "B C N O P S F Cl Br I * b c n o s p".split()

def token_to_id(tokens):
    return {token: tokens.index(token) for token in tokens}

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # If we have found a better segmentation ending at end_idx, we update
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score

def train_tokenizer_with_unigram(dataset='qm9', vocab_size=100):
    data = pd.read_csv(f'stgg/resource/data/{dataset}/train_val.txt')
    sp = spm.SentencePieceProcessor(model_file=f'stgg/resource/tokenizer/{dataset}/{dataset}.model')
    data.columns =['smiles']
    data['tokens'] = data['smiles'].apply(lambda x: sp.encode_as_pieces(x))
    data['ids'] = data['smiles'].apply(lambda x: sp.encode_as_ids(x))

    word_freqs = dict(data['tokens'].explode().value_counts())

    char_freqs = defaultdict(int)
    subwords_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word)):
            char_freqs[word[i]] += freq
            # Loop through the subwords of length at least 2
            for j in range(i + 2, len(word) + 1):
                subwords_freqs[word[i:j]] += freq

    sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)

    token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
    token_freqs = {token: freq for token, freq in token_freqs}

    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

    percent_to_remove = 0.1
    while len(model) > vocab_size:
        scores = compute_scores(model, word_freqs)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        # Remove percent_to_remove tokens with the lowest scores.
        for i in range(int(len(model) * percent_to_remove)):
            _ = token_freqs.pop(sorted_scores[i][0])

        total_sum = sum([freq for token, freq in token_freqs.items()])
        model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

    tokens = list(model.keys())

    with open(f"stgg/resource/tokenizer/{dataset}/model_{vocab_size}.txt", 'w') as model_file:
        model_file.write(json.dumps(model))
    with open(f"stgg/resource/tokenizer/{dataset}/tokens_uni_{vocab_size}.txt", 'w') as vocab_file:
        for line in tokens:
            vocab_file.write(f"{line}\n")

    return model, tokens

def compute_loss(model, word_freqs):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss

def compute_scores(model, word_freqs):
    scores = {}
    model_loss = compute_loss(model, word_freqs)
    for token, score in model.items():
        # We always keep tokens of length 1
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token, word_freqs) - model_loss
    return scores

def tokenize_spm_uni(smiles, dataset='qm9'):
    if dataset == 'qm9':
        pre_tokenized_text = sp_qm9.encode_as_pieces(smiles)
        model = MODEL_QM9
        TOKEN2ID = token_to_id(TOKENS_UNI_QM9)
    elif dataset == 'zinc':
        pre_tokenized_text = sp_zinc.encode_as_pieces(smiles)
        model = MODEL_ZINC
        TOKEN2ID = token_to_id(TOKENS_UNI_ZINC)

    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    tokens = [TOKEN2ID[BOS_TOKEN]]
    token_strs = sum(encoded_words, [])
    tokens.extend([TOKEN2ID[token] for token in token_strs])
    tokens.append(TOKEN2ID[EOS_TOKEN])
    return tokens

def compute_pair_freqs(splits, word_freqs):
    # used in train_tokenizer_with_bpe
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a, b, splits, word_freqs):
    # used in train_tokenizer_with_bpe
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

def train_tokenizer_with_bpe(dataset='qm9', vocab_size=50):
    # generate merges and tokens file for BPE
    data = pd.read_csv(f'../resource/data/{dataset}/train_val.txt')
    data.columns =['smiles']
    sp = spm.SentencePieceProcessor(model_file=f'stgg/resource/tokenizer/{dataset}/{dataset}.model')
    data['tokens'] = data['smiles'].apply(lambda x: sp.encode_as_pieces(x))
    data['ids'] = data['smiles'].apply(lambda x: sp.encode_as_ids(x))

    word_freqs = dict(data['tokens'].explode().value_counts())
    key_list = [list(key) for key in word_freqs.keys()]
    vocab = list({char for char_list in key_list for char in char_list})
    splits = {word: [c for c in word] for word in word_freqs.keys()}

    pair_freqs = compute_pair_freqs(splits, word_freqs)
    merges = defaultdict()
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        if best_pair == '':
            print(f"Too large vocab size: break at {len(vocab)} before vocab size {vocab_size}")
            break
        splits = merge_pair(*best_pair, splits, word_freqs)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])

    merges_str = {str(key):value for key, value in merges.items()}
    with open(f"../resource/tokenizer/{dataset}/merges_{vocab_size}.txt", 'w') as merges_file:
        merges_file.write(json.dumps(merges_str))
    with open(f"../resource/tokenizer/{dataset}/tokens_bpe_{vocab_size}.txt", 'w') as vocab_file:
        for line in vocab:
            vocab_file.write(f"{line}\n")
    return merges, vocab

def tokenize_spm_bpe(smiles, dataset='qm9'):
    if dataset == 'qm9':
        pre_tokenized_text = sp_qm9.encode_as_pieces(smiles)
        merges = MERGES_QM9
        TOKEN2ID = token_to_id(TOKENS_BPE_QM9)
    elif dataset == 'zinc':
        pre_tokenized_text = sp_zinc.encode_as_pieces(smiles)
        merges = MERGES_ZINC
        TOKEN2ID = token_to_id(TOKENS_BPE_ZINC)

    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split
    tokens = [TOKEN2ID[BOS_TOKEN]]
    token_strs = sum(splits, [])
    tokens.extend([TOKEN2ID[token] for token in token_strs])
    tokens.append(TOKEN2ID[EOS_TOKEN])
    return tokens

def tokenize_spm(smiles, dataset='qm9'):
    if dataset == 'qm9':
        TOKEN2ID = token_to_id(TOKENS_SPM_QM9)
        tokens = [TOKEN2ID[BOS_TOKEN]]
        tokens.extend(sp_qm9.encode_as_ids(smiles))
    else:
        TOKEN2ID = token_to_id(TOKENS_SPM_ZINC)
        tokens = [TOKEN2ID[BOS_TOKEN]]
        tokens.extend(sp_zinc.encode_as_ids(smiles))
    tokens.append(TOKEN2ID[EOS_TOKEN])
    return tokens

def tokenize_selfies(selfies):
    selfies_split = selfies.split("]")[:-1]
    tokens = ["[bos]"]
    tokens.extend([selfies + "]" for selfies in selfies_split])
    tokens.append("[eos]")
    TOKEN2ID = token_to_id(TOKENS_SELFIES)
    return [TOKEN2ID[token] for token in tokens]

def tokenize_deepsmiles(deep_smiles):
    deep_smiles = iter(deep_smiles)
    tokens = ["[bos]"]
    peek = None
    while True:
        char = peek if peek else next(deep_smiles, "")
        peek = None
        if not char:
            break

        if char == "[":
            token = char
            for char in deep_smiles:
                token += char
                if char == "]":
                    break

        elif char in ORGANIC_ATOMS:
            peek = next(deep_smiles, "")
            if char + peek in ORGANIC_ATOMS:
                token = char + peek
                peek = None
            else:
                token = char

        elif char == "%":
            peek = next(deep_smiles, "")
            if peek == '(':
                token = char + peek + next(deep_smiles, "") + next(deep_smiles, "") + next(deep_smiles, "") + next(deep_smiles, "")
            else:
                token = char + peek + next(deep_smiles, "")

        elif char in "-=#$:.)%/\\" or char.isdigit():
            token = char
        else:
            raise ValueError(f"Undefined tokenization for chararacter {char}")

        tokens.append(token)

    tokens.append("[eos]")
    TOKEN2ID = token_to_id(TOKENS_DEEPSMILES)
    return [TOKEN2ID[token] for token in tokens]

def tokenize(smiles):
    smiles = iter(smiles)
    tokens = ["[bos]"]
    peek = None
    while True:
        char = peek if peek else next(smiles, "")
        peek = None
        if not char:
            break

        if char == "[":
            token = char
            for char in smiles:
                token += char
                if char == "]":
                    break

        elif char in ORGANIC_ATOMS:
            peek = next(smiles, "")
            if char + peek in ORGANIC_ATOMS:
                token = char + peek
                peek = None
            else:
                token = char

        elif char == "%":
            token = char + next(smiles, "") + next(smiles, "")
            print(token)
            print(TOKEN2ID[token])
            assert False

        elif char in "-=#$:.()%/\\" or char.isdigit():
            token = char
        else:
            raise ValueError(f"Undefined tokenization for chararacter {char}")

        tokens.append(token)

    tokens.append("[eos]")
    TOKEN2ID = token_to_id(TOKENS)
    return [TOKEN2ID[token] for token in tokens]


def untokenize(sequence, string_type):
    if string_type == 'spm':
        ID2TOKEN = id_to_token(TOKENS_SPM_QM9)
    elif string_type == 'spm_zinc':
        ID2TOKEN = id_to_token(TOKENS_SPM_ZINC)
        # return [sp_qm9.IdToPiece(id) for id in sequence]
    elif string_type == 'selfies':
        ID2TOKEN = id_to_token(TOKENS_SELFIES)
    elif string_type == 'deep_smiles':
        ID2TOKEN = id_to_token(TOKENS_DEEPSMILES)
    elif string_type == 'smiles':
        ID2TOKEN = id_to_token(TOKENS)
    elif string_type == 'bpe':
        ID2TOKEN = id_to_token(TOKENS_BPE_QM9)
    elif string_type == 'bpe_zinc':
        ID2TOKEN = id_to_token(TOKENS_BPE_ZINC)
    elif string_type == 'uni':
        ID2TOKEN = id_to_token(TOKENS_UNI_QM9)
    elif string_type == 'uni_zinc':
        ID2TOKEN = id_to_token(TOKENS_UNI_ZINC)
    else:
        raise ValueError(f"Undefined string type {string_type}")


    tokens = [ID2TOKEN[id_] for id_ in sequence]
    if tokens[0] != "[bos]":
        return ""
    elif "[eos]" not in tokens:
        return ""

    tokens = tokens[1 : tokens.index("[eos]")]
    return "".join(tokens)


DATA_DIR = "../resource/data"

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc"
    simple = True

    def __init__(self, split, string_type='smiles'):
        self.string_type = string_type

        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        if string_type in ['smiles', 'spm', 'spm_zinc', 'bpe', 'bpe_zinc', 'uni', 'uni_zinc']:
            string_list = smiles_list
        elif string_type == 'selfies':
            string_list = Parallel(n_jobs=8)(delayed(sf.encoder)(smiles) for smiles in smiles_list)
        elif string_type == 'deep_smiles':
            converter = Converter(rings=True, branches=True)
            string_list = Parallel(n_jobs=8)(delayed(converter.encode)(smiles) for smiles in smiles_list)
        else :
            raise ValueError(f"Undefined string type {string_type}")
        self.smiles_list = string_list


    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        if self.string_type == 'smiles':
            mol = Chem.MolFromSmiles(smiles)
            
            if self.simple:
                Chem.Kekulize(mol)

            smiles = Chem.MolToSmiles(mol)
            return torch.LongTensor(tokenize(smiles))

        elif self.string_type == 'selfies':
            return torch.LongTensor(tokenize_selfies(smiles))
        elif self.string_type == 'deep_smiles':
            return torch.LongTensor(tokenize_deepsmiles(smiles))
        elif self.string_type == 'spm':
            return torch.LongTensor(tokenize_spm(smiles, 'qm9'))
        elif self.string_type == 'spm_zinc':
            return torch.LongTensor(tokenize_spm(smiles, 'zinc'))
        elif self.string_type == 'bpe':
            return torch.LongTensor(tokenize_spm_bpe(smiles, 'qm9'))
        elif self.string_type == 'bpe_zinc':
            return torch.LongTensor(tokenize_spm_bpe(smiles, 'zinc'))
        elif self.string_type == 'uni':
            return torch.LongTensor(tokenize_spm_uni(smiles, 'qm9'))
        elif self.string_type == 'uni_zinc':
            return torch.LongTensor(tokenize_spm_uni(smiles, 'zinc'))
            
        else:
            raise ValueError(f"Undefined string type {self.string_type}")
    
    def get_tokens(self):
        if self.string_type == 'selfies':
            symbols = sf.get_alphabet_from_selfies(self.smiles_list)
            return symbols

        
class QM9Dataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/qm9"
    simple = True


class SimpleMosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"
    simple = True


class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"
    simple = False