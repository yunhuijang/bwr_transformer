from rdkit import Chem, RDLogger

import torch
import torch.nn.functional as F
import selfies as sf
from deepsmiles import Converter
import sentencepiece as spm


def train_sentence_piece(dataset='qm9'):
    # train sentence piece and generate model and vocab file
    if dataset == 'qm9':
        spm.SentencePieceTrainer.Train('--input=resource/data/qm9/train_val.txt --model_prefix=qm9 --vocab_size=127 --max_sentence_length=200')
    elif dataset == 'zinc':
        spm.SentencePieceTrainer.Train('--input=resource/data/zinc/train_val.txt --model_prefix=zinc --vocab_size=200 --max_sentence_length=400')

def compute_sequence_accuracy(logits, batched_sequence_data, ignore_index=0):
    batch_size = batched_sequence_data.size(0)
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]
    preds = torch.argmax(logits, dim=-1)

    correct = preds == targets
    correct[targets == ignore_index] = True
    elem_acc = correct[targets != 0].float().mean()
    sequence_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, sequence_acc


def compute_sequence_cross_entropy(logits, batched_sequence_data, ignore_index=0):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)

    return loss

def compute_entropy(logits, batched_sequence_data, ignore_index=0):
    logits = logits[:, :-1].reshape(-1, logits.size(-1))
    targets = batched_sequence_data[:, 1:].reshape(-1)

    logits = logits[targets != ignore_index]
    probs = torch.softmax(logits, dim=-1)
    probs = probs[~torch.isinf(logits)]
    loss = -(probs * torch.log(probs)).sum() / logits.size(0)
    return loss
    
def canonicalize_selfies(selfies):
    if selfies is None:
        return None
    try:
        smiles = sf.decoder(selfies)
        if smiles is None:
            return None
        selfies = sf.encoder(smiles)
    except:
        return None
    
    if len(selfies) == 0:
        return None
    
    return selfies

def canonicalize_deep_smiles(deep_smiles):
    if deep_smiles is None:
        return None
    try:
        converter = Converter(rings=True, branches=True)
        smiles = converter.decode(deep_smiles)
        if smiles is None:
            return None
        deep_smiles = converter.encode(smiles)
    except:
        return None
    
    if len(deep_smiles) == 0:
        return None
    
    return deep_smiles

def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        smiles = Chem.MolToSmiles(mol)
    except:
        return None   


    if len(smiles) == 0:
        return None

    return smiles

def pad_square(squares, padding_value=0):
    max_dim = max([square.size(0) for square in squares])
    batched_squares = torch.full((len(squares), max_dim, max_dim), padding_value, dtype=torch.long)
    for idx, square in enumerate(squares):
        batched_squares[idx, : square.size(0), : square.size(1)] = square

    return batched_squares
