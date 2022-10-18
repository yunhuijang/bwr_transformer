from rdkit import Chem
import enum
from deepsmiles import Converter
import selfies as sf
from joblib import Parallel, delayed

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
# in QM5
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

def tokenize_selfies(selfies):
    selfies_split = selfies.split("]")[:-1]
    tokens = []
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
            # TODO: 똑바로 했는지 확인
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
    if string_type == 'selfies':
        ID2TOKEN = id_to_token(TOKENS_SELFIES)
    elif string_type == 'deep_smiles':
        ID2TOKEN = id_to_token(TOKENS_DEEPSMILES)
    elif string_type == 'smiles':
        ID2TOKEN = id_to_token(TOKENS)
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
        if string_type == 'smiles':
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