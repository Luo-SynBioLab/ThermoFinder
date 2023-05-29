import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
from transformers import T5EncoderModel, T5Tokenizer
from Extract_feature import *
from CPCProt.tokenizer import Tokenizer
from CPCProt import CPCProtModel, CPCProtEmbedding
import tensorflow.compat.v1 as tf
from tensorflow.python.util import deprecation
import torch
import json
import numpy as np
import pandas as pd
import re
import gc
import random
import pickle
from tensorflow.compat.v1 import InteractiveSession
deprecation._PRINT_DEPRECATION_WARNINGS = False


AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]

def residues_to_one_hot(amino_acid_residues):
    to_return = []
    normalized_residues = amino_acid_residues.replace('U', 'C').replace('O', 'X')
    for char in normalized_residues:
        if char in AMINO_ACID_VOCABULARY:
            to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
            to_append[AMINO_ACID_VOCABULARY.index(char)] = 1.
            to_return.append(to_append)
        elif char == 'B':  # Asparagine or aspartic acid.
            to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
            to_append[AMINO_ACID_VOCABULARY.index('D')] = .5
            to_append[AMINO_ACID_VOCABULARY.index('N')] = .5
            to_return.append(to_append)
        elif char == 'Z':  # Glutamine or glutamic acid.
            to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
            to_append[AMINO_ACID_VOCABULARY.index('E')] = .5
            to_append[AMINO_ACID_VOCABULARY.index('Q')] = .5
            to_return.append(to_append)
        elif char == 'X':
            to_return.append(
                np.full(len(AMINO_ACID_VOCABULARY), 1. / len(AMINO_ACID_VOCABULARY)))
        elif char == _PFAM_GAP_CHARACTER:
            to_return.append(np.zeros(len(AMINO_ACID_VOCABULARY)))
        else:
            raise ValueError('Could not one-hot code character {}'.format(char))
    return np.array(to_return)


def pad_one_hot_sequence(sequence: np.ndarray,
                         target_length: int) -> np.ndarray:
    """Pads one hot sequence [seq_len, num_aas] in the seq_len dimension."""
    sequence_length = sequence.shape[0]
    pad_length = target_length - sequence_length
    if pad_length < 0:
        raise ValueError(
            'Cannot set a negative amount of padding. Sequence length was {}, target_length was {}.'
                .format(sequence_length, target_length))
    pad_values = [[0, pad_length], [0, 0]]
    return np.pad(sequence, pad_values, mode='constant')


def Load_data():
    print('Data Loading...')
    Sequence = []
    with open('Proteome_protein/70_proteins_X_50.fasta', 'r') as myfile:
        for line in myfile:
            if line[0] != '>':
                Sequence.append(line.strip('\n'))
    random.shuffle(Sequence)
    Length = len(Sequence)
    Sequence_2 = []
    with open('Proteome_protein/30_proteins_X_50.fasta', 'r') as myfile:
        for line in myfile:
            if line[0] != '>':
                Sequence_2.append(line.strip('\n'))
    random.shuffle(Sequence_2)
    Sequence.extend(Sequence_2[:Length])
    for i in range(len(Sequence)):
        Sequence[i] = Sequence[i][:1000]
    Mysequence = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        Mysequence.append(zj)
    print(len(Sequence))
    return Sequence


def CNN_features(Sequence):
    sess = tf.Session()
    graph = tf.Graph()
    with graph.as_default():
        saved_model = tf.saved_model.load(sess, ['serve'], 'trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760')
    sequence_input_tensor_name = saved_model.signature_def['confidences'].inputs['sequence'].name
    sequence_lengths_input_tensor_name = saved_model.signature_def['confidences'].inputs['sequence_length'].name
    embedding_signature = saved_model.signature_def['pooled_representation']
    embedding_signature_tensor_name = embedding_signature.outputs['output'].name
    features = []
    for i in range(len(Sequence)):
        print(i)
        seq = Sequence[i]
        with graph.as_default():
            embedding = sess.run(
                embedding_signature_tensor_name,
                {
                    sequence_input_tensor_name: [residues_to_one_hot(seq)],
                    sequence_lengths_input_tensor_name: [len(seq)],
                }
            )
        features.append(embedding[0])
    print(len(features), len(features[0]))
    return features


def Elmo_features(Sequence):
    model_dir = Path('uniref50_v2')
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    embedder = ElmoEmbedder(options, weights, cuda_device=0)
    features = []
    for i in range(len(Sequence)):
        print(i)
        seq = Sequence[i]
        embedding = embedder.embed_sentence(list(seq))  # List-of-Lists with shape [3,L,1024]
        protein_embd = (torch.tensor(embedding).sum(dim=0).mean(dim=0)).cpu().detach().numpy()
        features.append(protein_embd)
    print(len(features), len(features[0]))
    return features


def CPC_features(Sequence):
    ckpt_path = "CPC/best.ckpt"  # Replace with actual path to CPCProt weights
    model = CPCProtModel()
    model.load_state_dict(torch.load(ckpt_path))
    embedder = CPCProtEmbedding(model)
    tokenizer = Tokenizer()
    features = []
    for i in range(len(Sequence)):
        print(i)
        seq = Sequence[i]
        input = torch.tensor([tokenizer.encode(seq)])
        z_mean = embedder.get_z_mean(input)[0]   # (512)
        c_mean = embedder.get_c_mean(input)[0]   # (512)
        c_final = embedder.get_c_final(input)  # (512)
        fea = torch.cat((z_mean, c_mean, c_final), dim=0)
        fea = fea.detach().cpu().numpy()
        features.append(fea)
    print(len(features), len(features[0]))
    return features


def protTrans_features(Sequence):
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    # Automatic extracted features
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    print(len(features_normalize), len(features_normalize[0]))
    return features_normalize


def Write_features(features_ensemble):
    features_ensemble = features_ensemble.T
    Mydic= {}
    for i in range(len(features_ensemble)):
        if i < 1100:
            Mydic['CNN_'+str(i+1)] = features_ensemble[i]
        elif i < 1100+1024:
            Mydic['ELMO_'+str(i+1-1100)] = features_ensemble[i]
        elif i < 1100+1024+1536:
            Mydic['CPC_'+str(i+1-1100-1024)] = features_ensemble[i]
        else:
            Mydic['protTrans_'+str(i+1-1100-1024-1536)] = features_ensemble[i]
    with open('Features_All_proteome.pkl', 'wb') as f:
        pickle.dump(Mydic, f)
    # res = pd.DataFrame(Mydic)
    # res.to_excel('Features_All_proteome.xlsx')


if __name__ == '__main__':
    Sequence = Load_data()
    # CNN features
    f_CNN = CNN_features(Sequence)
    # ELMO features
    f_Elmo = Elmo_features(Sequence)
    # CPC features
    f_CPC = CPC_features(Sequence)
    # ProtTrans features
    f_protTrans = protTrans_features(Sequence)
    # Save features
    features_ensemble = np.concatenate((f_CNN, f_Elmo, f_CPC, f_protTrans), axis=1)
    Write_features(features_ensemble)
