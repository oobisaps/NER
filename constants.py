
'''

    Environment Pathes

'''
import os
import torch
from pytorch_helpers import load_vocab
from pytorch_pretrained_bert import BertModel, BertConfig, BertForPreTraining




ABS_PATH = '/home/alem/Alem_Sagandykov_Documents/Alem_Social/Location_Identifier/Named_Entity_Recognition/data/'
collections3_v2_path = 'Russian/collection3_v2/pickle/dataset.pkl'
akerke_tagged_complaints_path = 'Russian/Alem_Tagged_Complaints/akerke_tagged/pickle/dataset.pkl'
ru_bert_path = '/home/alem/Alem_Sagandykov_Documents/Alem_Social/HERMES/'

vocab_path = 'Production/vocab.txt'
config_path = 'Production/bert_config.json'
ru_bert_pytorch_weights_path_pth = 'pytorch_dump/deeppavlov_pretrained_rubert.pth'
ru_bert_pytorch_weights_path_bin = 'pytorch_dump/rubert_cased_L-12_H-768_A-12_pt/pytorch_model.bin'

vocab = load_vocab(os.path.join(ru_bert_path, vocab_path))

tags2index = {
    'X' : 0, 'O' : 1,
    'B-ORG' : 2, 'I-ORG' : 3,
    'B-PER' : 4, 'I-PER' : 5,
    'B-LOC' : 6, 'I-LOC' : 7
}

index2tags = dict(zip(tags2index.values(), tags2index.keys()))

config = BertConfig(os.path.join(ru_bert_path, config_path))
bert_for_pretraining = BertForPreTraining(config)
ru_bert_weights = torch.load(os.path.join(ru_bert_path, ru_bert_pytorch_weights_path_bin))
bert_for_pretraining.load_state_dict(ru_bert_weights)

bert_layer = bert_for_pretraining.bert

