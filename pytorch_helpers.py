'''

    Standard Modules

'''

import re 
import os

import pickle as pkl

import collections

from functools import reduce
from collections import Counter

#######################################################################################################################################


''' 

    External Modules

'''

import numpy as np
import pandas as pd

from tqdm import tqdm, trange

#######################################################################################################################################

''' 

    DeepPavlov Modules

'''

# from deeppavlov.models.preprocessors.bert_preprocessor import BertNerPreprocessor

#######################################################################################################################################

''' 

    Pytorch-Pretrained-Bert Modules

'''

#######################################################################################################################################

from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

#######################################################################################################################################

''' 

    PyTorch Modules

'''

import torch
from torch.utils.data import Dataset

#######################################################################################################################################


# NER DATASET
class NER_Dataset(Dataset):

    def __init__(self,dataset, transform = None):
        self.dataset = dataset
        self.transform = transform

    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        tokens, tags, index = self.dataset[idx]

        sample = {
            'tokens' : tokens,
            'tags' : tags,
            'sample_index' : index
        }

        if self.transform:
            return self.transform(sample)
        
        return sample
        
# HELPERS


def print_results(sample):
    
    fmt = '{:20} : {:5} : {}'
    
    tokens, tags, index = list(sample.keys())
    
    for token, tag in zip(sample[tokens], sample[tags]):
        print(fmt.format(token, tag, sample[index]))
    
    print()


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

# DATA PIPELINE


class Data_Pipeline:
    
    def __init__(self, *args):
        self.content_pipeline = list(args)
    
    
    def __call__(self, obj):
        
        return reduce(lambda x,y : y(x), [obj] + self.content_pipeline)


class Bert_Markers_Adder:
    
    def __init__(self, verbose = False):
        self.verbose = verbose    
    
    def __call__(self, sample):
        
        sample['tokens'] = ['[CLS]'] + sample['tokens'] + ['[SEP]']
        sample['tags'] = ['X'] + sample['tags'] + ['X']
        
        if self.verbose:
            print_results(sample)
        
        return sample


def add_artefacts_tags(tags, cased_index, cased_tag = 'X'):
    for cased_index in cased_index:
        tags.insert(cased_index, cased_tag)
    
    return tags

def convert_tags_to_ids(tags, tags_dict_index):
    
    return [tags_dict_index[tag] for tag in tags]

class Pytorch_Wordpiece_Tokenizer:
    
    def __init__(self, tokenizer, verbose = False, cased_token = '#'):
        
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.cased_token = cased_token
    
    def __call__(self, sample):
        
        cased_tokens = self.tokenizer.tokenize(' '.join(sample['tokens']))
        cased_index = [i for i in range(len(cased_tokens)) if self.cased_token in cased_tokens[i]]
        cased_tags = add_artefacts_tags(sample['tags'], cased_index)
        
        sample['tokens'], sample['tags'] = cased_tokens, cased_tags
        
        cased_sample = {
            'cased_tokens' : cased_tokens,
            'cased_tags' : cased_tags,
            'sample_index' : sample['sample_index']
        }
        
        del sample
        
        if self.verbose:
            print_results(cased_sample)
        
        return cased_sample
        
class Token_To_Id_Transformer:
    
    def __init__(self, tags_dict_index, tokenizer, verbose = False):
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.tags_dict_index = tags_dict_index
        
    def __call__(self, cased_sample):
        
        cased_tokens_ids = self.tokenizer.convert_tokens_to_ids(cased_sample['cased_tokens'])
        cased_tags_ids = convert_tags_to_ids(cased_sample['cased_tags'], self.tags_dict_index)
        
        cased_sample['tokens_ids'], cased_sample['tags_ids'], cased_sample['sample_index_ids'] = cased_tokens_ids, \
                                                                                                 cased_tags_ids, \
                                                                                                 cased_sample['sample_index']
        
        del cased_sample['cased_tokens'], cased_sample['cased_tags'], cased_sample['sample_index']
        
        # print(cased_sample.keys())
        
        if self.verbose:
            print_results(cased_sample)
        
        return cased_sample

class Pad_Trunc_Sequence_Getter:
    
    def __init__(self, max_len, padding_truncating, values, vocab, tags_dict_index, verbose = False):
        
        self.verbose = verbose
        self.max_len = max_len
        self.padding_truncating = padding_truncating
        self.values = values
        self.vocab = vocab
        self.tags_dict_index = tags_dict_index
    
    def __call__(self, cased_sample):
        pad_trunc_tokens_ids = pad_sequences([cased_sample['tokens_ids']], 
                                             maxlen = self.max_len, 
                                             truncating = self.padding_truncating, 
                                             padding = self.padding_truncating, 
                                             value = self.values['tokens'])
        
        pad_trunc_tags_ids = pad_sequences([cased_sample['tags_ids']], 
                                           maxlen=self.max_len, 
                                           truncating = self.padding_truncating, 
                                           padding = self.padding_truncating, 
                                           value = self.values['tags'])
        
        if len(cased_sample['tokens_ids']) > self.max_len:
            pad_trunc_tokens_ids = list(pad_trunc_tokens_ids[0][:-1]) + [self.vocab['[SEP]']]
            pad_trunc_tags_ids = list(pad_trunc_tags_ids[0][:-1]) + [self.tags_dict_index['X']]
        
        else:
            pad_trunc_tokens_ids = pad_trunc_tokens_ids[0]
            pad_trunc_tags_ids = pad_trunc_tags_ids[0]
            
        
        cased_sample['pad_trunc_tokens_ids'], cased_sample['pad_trunc_tags_ids'] = pad_trunc_tokens_ids, \
                                                                                   pad_trunc_tags_ids
        
        cased_sample['pad_trunc_sample_index_ids'] = [cased_sample['sample_index_ids']]
        
        del cased_sample['tokens_ids'], cased_sample['tags_ids'], cased_sample['sample_index_ids']
        
        if self.verbose:
            print_results(cased_sample)
        
        
        return cased_sample

def make_attention_mask(cased_sample):
    cased_sample['pad_trunc_attention_mask'] = [float(token_id>0) for token_id in cased_sample['pad_trunc_tokens_ids']]
    
    return cased_sample

def make_long_tensor(cased_sample):
    return {key : torch.LongTensor(cased_sample[key]) for key in cased_sample}

def debug(cased_sample):
    fmt = '{:35} - {}'

    print(cased_sample.keys())

    for key in cased_sample:
        print(fmt.format(key, cased_sample[key].shape))
    
    return cased_sample





 
        