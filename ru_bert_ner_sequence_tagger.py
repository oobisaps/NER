'''
    Standard Modules
'''

import os
import sys
import time
import argparse
import pickle as pkl
from collections import Counter

from functools import reduce

'''
    External Modules
'''

import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt

import torch 
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils

'''
    Internal Modules
'''

from ml_helpers import *
from pytorch_helpers import *
from constants import *


'''
    CONSTANTS
'''

SEED = 17
EPOCHS = 5
BATCH_SIZE = 32
SHUFFLE = True
IN_FEATURES = 768
MAX_SEQ_LEN = 128


parser = argparse.ArgumentParser()
parser.add_argument('-optimizer', type=str, required=True, help='Adam/SGD')
parser.add_argument('-loss_function', type=str, required=True, help='Cross_Entropy/?')

parser.add_argument('-learning_rate', type=str, required=True, help='learning_rate of optimizer')
parser.add_argument('-max_seq_length', type=str, required=True, help='128/256/512')

parser.add_argument('-epochs', type=str, required=True, help='5/10/15')
parser.add_argument('-finetunning', type=str, required=True, help='0/1')
parser.add_argument('-gradient_clipping', type=str, required=True, help='0/1')
parser.add_argument('-max_grad', type=str, required=True, help='0,1...1.0')


parser.add_argument('-title', type=str, required=True, help='title_of_model')
parser.add_argument('-version', type=str, required=True, help='V1...VN')
parser.add_argument('-test_code', type=str, required=True, help='0/1')



args = parser.parse_args()



tokenizer = BertTokenizer(vocab_file = os.path.join(ru_bert_path, vocab_path), 
                          do_lower_case = False, 
                          do_basic_tokenize = False)

bert_preprocessor_transform_obj = Data_Pipeline(
    Bert_Markers_Adder(),
    
    Pytorch_Wordpiece_Tokenizer(tokenizer),
    
    Token_To_Id_Transformer(tags2index, tokenizer),  
    
    Pad_Trunc_Sequence_Getter(max_len = 128, 
                              padding_truncating = 'post',
                              vocab = vocab,
                              tags_dict_index = tags2index,
                              values = {
                                  'tokens' : 0,
                                  'tags' : 1
                              }),
    
    make_attention_mask,
    make_long_tensor,
    # debug
)

fmt = '{:5} : {:}'

with open(os.path.join(ABS_PATH, collections3_v2_path), 'rb') as file:
    collection3_v2_dataset = pkl.load(file)


with open(os.path.join(ABS_PATH, akerke_tagged_complaints_path), 'rb') as file:
    akerke_tagged_complaints_dataset = pkl.load(file)

print('Collection3_V2\n')
for key in collection3_v2_dataset:
    print(fmt.format(key, len(collection3_v2_dataset[key])))
print()

print('Akerke_Tagged_Complaints\n')
for key in akerke_tagged_complaints_dataset:
    print(fmt.format(key, len(akerke_tagged_complaints_dataset[key])))

print()



train_data = list(collection3_v2_dataset['train']) + list(akerke_tagged_complaints_dataset['train'])
test_data = list(collection3_v2_dataset['test']) + list(akerke_tagged_complaints_dataset['test'])
valid_data = list(collection3_v2_dataset['valid']) + list(akerke_tagged_complaints_dataset['valid'])

train_data = [(train_data[i][0], train_data[i][1], i) for i in range(len(train_data))]
test_data = [(test_data[i][0], test_data[i][1], i) for i in range(len(test_data))]
valid_data = [(valid_data[i][0], valid_data[i][1], i) for i in range(len(valid_data))]


train_dataset = NER_Dataset(dataset = train_data, transform = bert_preprocessor_transform_obj)
test_dataset = NER_Dataset(dataset = test_data, transform = bert_preprocessor_transform_obj)
valid_dataset = NER_Dataset(dataset = valid_data, transform = bert_preprocessor_transform_obj)

train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = SHUFFLE)
test_loader = data_utils.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = SHUFFLE)
valid_loader = data_utils.DataLoader(dataset = valid_dataset, batch_size = BATCH_SIZE, shuffle = SHUFFLE)


print(len(tags2index))

ru_bert_ner_sequence_tagger = Bert_For_Token_Classification(config, 
                                                            num_labels = len(tags2index),
                                                            bert_layer = bert_layer)



if args.optimizer == 'Adam':
    optimizer = optim.Adam
elif args.optimizer == 'SGD':
    optimizer = optim.SGD

if args.loss_function == 'Cross_Entropy':
    loss_function = nn.CrossEntropyLoss()
else:
    pass



model_param = {
    'finetunning' : int(args.finetunning),
    'max_seq_len' : int(args.max_seq_length),
    'learning_rate' : float(args.learning_rate),
    'epochs' : int(args.epochs),
    'gradient_clipping' : int(args.gradient_clipping),
    'max_grad' : float(args.max_grad),
    'test_code' : int(args.test_code),
    'num_labels' : len(tags2index),
    'optimizer' : args.optimizer,
    'loss_function' : args.loss_function
}

environment_param = {
    'abs_path' : 'Model_Results',
    'title' : args.title,
    'version' : args.version,
    'verbose_mode' : ''
}


abs_model_helper = Model_Helper(model = ru_bert_ner_sequence_tagger, 
                                optimizer = optimizer, 
                                loss_function = loss_function, 
                                loaders = {
                                    'train' : train_loader,
                                    'test' : test_loader,
                                    'valid' : valid_loader
                                }, 
                                index2tags = index2tags, 
                                model_param = model_param, 
                                environment_param = environment_param)

abs_model_helper.train_model()






