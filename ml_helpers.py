'''

    Standard Modules

'''

import re 
import os
import sys
import time
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


from pytorch_pretrained_bert import BertAdam, BertConfig

#######################################################################################################################################

''' 

    PyTorch Modules

'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

#######################################################################################################################################

''' 

    Seqeval Metrics

'''

from seqeval.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score

#######################################################################################################################################

class Bert_For_Token_Classification(nn.Module):
    
    def __init__(self, config, num_labels, bert_layer):
        super(Bert_For_Token_Classification, self).__init__()
        self.num_labels = num_labels
        self.bert = bert_layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_ids, token_type_ids = None, attention_mask = None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return logits

class Model_Helper:

    def __init__(self,model, optimizer, loss_function, loaders, index2tags, **parametres):
        self.model = model
        self.optimizer = optimizer
        self.loaders = loaders
        self.loss_function = loss_function
        self.statistics_df = {
            'train' : pd.DataFrame(), 
            'test' : pd.DataFrame(),
            'valid' : pd.DataFrame()
        }
        self.index2tags = index2tags
        self.model_train_parametres = parametres['model_param']
        self.environment_parametres = parametres['environment_param']
        self.model_directory = self.init_home_directory()
        self.time_init = time.ctime().replace(' ', '_')
        self.filelogs_info = self.init_file_logs()

    def init_home_directory(self):
        model_directory = os.path.join(self.environment_parametres['abs_path'], 
                                       self.environment_parametres['title'], 
                                       self.environment_parametres['version']
        )

        if os.path.exists(model_directory): pass    
        else: os.makedirs(name = model_directory, exist_ok = True)

        return model_directory
    
    def init_file_logs(self,):

        filelogs_path = os.path.join(self.model_directory,
                                     self.environment_parametres['title'] + '_model_logs_' + self.time_init + '.txt')
        filelogs_file = open(filelogs_path, 'w') if self.environment_parametres['verbose_mode'] != sys.stdout \
                                                 else self.environment_parametres['verbose_mode']
        
        return (filelogs_file, filelogs_path)
    
    def init_fine_tunning(self):
        
        if self.model_train_parametres['finetunning']:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
            ]

        else:
            param_optimizer = list(self.model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        
        return optimizer_grouped_parameters


    def parse_batch(self, batch):

        tags, token_ids, attention_masks, index = batch['pad_trunc_tags_ids'], \
                                                  batch['pad_trunc_tokens_ids'], \
                                                  batch['pad_trunc_attention_mask'], \
                                                  batch['pad_trunc_sample_index_ids']
        
        if self.model.device == 'cpu':
            pass

        else:
            tags, token_ids, attention_masks = tags.to(self.model.device), \
                                               token_ids.to(self.model.device), \
                                               attention_masks.to(self.model.device)
        
        return (tags, token_ids, attention_masks, index)
    
    def parse_index_to_tags(self, batch_tags):

        return [[self.index2tags[index] for index in tags] for tags in batch_tags.cpu().numpy()]

    
    def get_results(self, y_true,y_pred):
        result_med = collections.namedtuple(
                                        'result_med',
                                            [
                                                'precision',
                                                'recall',
                                                'f1_score',
                                                'accuracy',
                                        ]
                                )
        
        result_med.accuracy = accuracy_score(y_true,y_pred)
        result_med.precision = precision_score(y_true,y_pred)
        result_med.recall = recall_score(y_true,y_pred)
        result_med.f1_score = f1_score(y_true,y_pred)

        return result_med

    def result_wrapper(self,acc,prec,rec,f1,title,parametrs):
    
        results = {
            'title' : title,
            'accuracy' : acc,
            'precision' : prec,
            'recall':rec,
            'f1_score' : f1,
            'parametrs' : parametrs,
        }
    
        return results


    def make_statistics(self, **kwargs):

        self.statistics_df[kwargs['loader_case']]['INDEX'] = kwargs['index_all']
        self.statistics_df[kwargs['loader_case']]['LABEL'] = kwargs['y_true']
        self.statistics_df[kwargs['loader_case']]['PREDICTED'] = kwargs['y_pred']
        self.statistics_df[kwargs['loader_case']]['PROBA'] = kwargs['probas']

        self.statistics_df[kwargs['loader_case']].to_csv(os.path.join(self.model_directory,
                                              self.environment_parametres['title'] + f"_{kwargs['loader_case']}" + '_statistics.csv'), 
                                                index = False)


    def test_model(self, loader_key):
        y_true, y_pred, probas, index_all = [], [], [], []

        self.model.eval()

        with torch.no_grad():
            for batch_index, batch, in enumerate(self.loaders[loader_key], 0):
                tags, token_ids, attention_masks, index = self.parse_batch(batch)

                outputs = self.model(input_ids = token_ids,attention_mask = attention_masks)

                proba, y_pred_batch = torch.max(outputs, 2)

                # save for check results
                probas.extend(proba)
                index_all.extend(index)

                y_true.extend(self.parse_index_to_tags(tags))
                y_pred.extend(self.parse_index_to_tags(y_pred_batch))

                if self.model_train_parametres['test_code']:
                    break

                else:
                    continue
    

        self.make_statistics(index_all = index_all, 
                             y_true = y_true, 
                             y_pred = y_pred, 
                             probas = probas,
                             loader_case = loader_key, 
                             model_logs = None)
        

        print(loader_key, file = self.filelogs_info[0])    
        print(100 * '.', file = self.filelogs_info[0])
        print('', file = self.filelogs_info[0])
        print(f"TOTAL_RESULTS_{loader_key}: ", file = self.filelogs_info[0])
        print('', file = self.filelogs_info[0])
        print(classification_report(y_true, y_pred), file = self.filelogs_info[0])
        print('', file = self.filelogs_info[0])
        print(100 * '.', file = self.filelogs_info[0])

 
    def train_model(self,):

        def init_model_state_info():
            model_state_info = {
                    'model_state_dict' : None,
                    'optimizer_state_dict' : None,
                    'max_seq_length' : self.model_train_parametres['max_seq_len'],
                    'epochs' : self.model_train_parametres['epochs'],
                    'learnin_rate' : self.model_train_parametres['learning_rate'],
                    'total_loss_train' : None,
                    'model_logs' : dict((zip(range(self.model_train_parametres['epochs']), 
                                             [None] * self.model_train_parametres['epochs'])))
            }

            return model_state_info
        
        model_state_info = init_model_state_info()
        optimizer_grouped_parameters = self.init_fine_tunning()
        self.optimizer = self.optimizer(optimizer_grouped_parameters, self.model_train_parametres['learning_rate'])

        parametrs_string = 'learning_rate : {}, momentum : {}, epochs : {}, max_seq_length : {}'.format(self.model_train_parametres['learning_rate'],
                                                                                                        0.9,
                                                                                                        self.model_train_parametres['epochs'],
                                                                                                        self.model_train_parametres['max_seq_len'],)

        cli_string = "optimizer : {}, loss_function : {}, finetunning : {}, gradient_clipping : {}, max_grad : {}, num_labels : {}".format(
            self.model_train_parametres['optimizer'],
            self.model_train_parametres['loss_function'],
            self.model_train_parametres['finetunning'],
            self.model_train_parametres['gradient_clipping'],
            self.model_train_parametres['max_grad'],
            self.model_train_parametres['num_labels']
        )

        print('tags : ', *list(self.index2tags.values()), file = self.filelogs_info[0])
        print('CLI PARAMETRES : ', file = self.filelogs_info[0])
        print(cli_string, file = self.filelogs_info[0])
        print('HYPERPARAMETRS : ', file = self.filelogs_info[0])
        print(parametrs_string, file = self.filelogs_info[0])
        print('DEVICE : ',self.model.device, file = self.filelogs_info[0])

        total_loss = 0.0
        filelogs_fmt = '{} -> Epoch {} -> Batch_index {} -> {} -> {}'
        epoch_logs_format = '{} -> Epoch_loss : {}'

        self.model.to(self.model.device)

        for epoch in trange(self.model_train_parametres['epochs'], desc = 'epochs'):
            
            running_loss = 0.0
            y_true, y_pred, probas,index_all = [], [], [], []
            total_loss = 0.0

            self.model.train()

            nb_tr_examples, nb_tr_steps = 0, 0

            for batch_index, batch in enumerate(self.loaders['train'], 0):

                tags, token_ids, attention_masks, index = self.parse_batch(batch)

                self.model.zero_grad() # zero the parameter gradients
                
                # forward + backward + optimize
                outputs = self.model(input_ids = token_ids,attention_mask = attention_masks)

                proba, y_pred_batch = torch.max(outputs, 2)

                # save for check results
                probas.extend(proba)
                index_all.extend(index)

                y_true.extend(self.parse_index_to_tags(tags))
                y_pred.extend(self.parse_index_to_tags(y_pred_batch))

                loss = self.loss_function(outputs.view(-1, self.model_train_parametres['num_labels']), tags.view(-1))

                loss.backward()

                if self.model_train_parametres['gradient_clipping']:

                    torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), 
                                                   max_norm = self.model_train_parametres['max_grad'])

                self.optimizer.step()

                print(filelogs_fmt.format(time.ctime() ,epoch, batch_index,'loss', loss.item()), file = self.filelogs_info[0])
                running_loss += loss.item()
                total_loss += loss.item()

                if batch_index % 2000 == 1999:    # print every 2000 mini-batches
                    print(' ',file = self.filelogs_info[0])
                    print(100 * '*', file = self.filelogs_info[0])
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, batch_index + 1, running_loss / 2000), file = self.filelogs_info[0])
                    running_loss = 0.0
                    print(100 * '*', file = self.filelogs_info[0])
                    print(' ',file = self.filelogs_info[0])
                

                if self.model_train_parametres['test_code']:
                    break

                else:
                    continue

            self.test_model(loader_key = 'valid')
        
            norm_coeff = total_loss if batch_index == 0 else total_loss / batch_index

            print(' ', file = self.filelogs_info[0]) 
            print(100 * '-', file = self.filelogs_info[0])
            print(epoch_logs_format.format(time.ctime(), total_loss / norm_coeff), file = self.filelogs_info[0])
            print('Metrics :',file = self.filelogs_info[0])

            print(classification_report(y_true, y_pred), file = self.filelogs_info[0])
            print(' ', file = self.filelogs_info[0])
            print(100 * '-',file = self.filelogs_info[0])
            print('', file = self.filelogs_info[0])

            results = self.get_results(y_true, y_pred)

            results = self.result_wrapper(title = self.environment_parametres['title'] + '_' + self.environment_parametres['version'],
                                            acc = results.accuracy,
                                            prec = results.precision,
                                            rec = results.recall,
                                            f1 = results.f1_score,
                                            parametrs = parametrs_string)

            # print(results)

            model_state_info['model_logs'][epoch] = results

            if self.model_train_parametres['test_code']:
                break

            else:
                continue

        self.make_statistics(index_all = index_all, 
                             y_true = y_true, 
                             y_pred = y_pred, 
                             probas = probas,
                             loader_case = 'train', 
                             model_logs = model_state_info['model_logs'])
        
        model_state_info['model_state_dict'] = self.model.state_dict()
        model_state_info['optimizer_state_dict'] = self.optimizer.state_dict()
        model_state_info['total_loss_train'] = total_loss / norm_coeff

        torch.save(model_state_info, os.path.join(self.model_directory,self.environment_parametres['title'] + '_model_state_info.pth'))


        print(100 * '.', file = self.filelogs_info[0])
        print('', file = self.filelogs_info[0])
        print('Finished Training', file = self.filelogs_info[0])
        print('', file = self.filelogs_info[0])
        print('TOTAL_RESULTS_TRAIN : ', file = self.filelogs_info[0])
        print('', file = self.filelogs_info[0])
        print('LOSS : ', total_loss / norm_coeff, file = self.filelogs_info[0])
        print('', file = self.filelogs_info[0])
        print(classification_report(y_true, y_pred), file = self.filelogs_info[0])
        print('', file = self.filelogs_info[0])
  

        self.test_model(loader_key = 'test')

        
           




            










        






