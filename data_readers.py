import re
import os
from tqdm import tqdm

def parse_ner_file(path, pos_exist):
    
    with open(path, 'r', encoding = 'UTF-8') as file:
        data = file.readlines()
    
    data = [re.sub('\t', ' ', token.strip()) for token in data]
    data = [token.split() for token in data if token and token != '<DOCSTART>']

    tag_index = 2 if pos_exist else 1

    tags = [parsed_token[tag_index] for parsed_token in data]
    sentences = [parsed_token[0] for parsed_token in data]

    return (tags, sentences)


def parse_ner_folder(abs_path, pos_exist):
    
    valid_files = ['train.txt', 'valid.txt', 'test.txt']
    files = [filename for filename in os.listdir(abs_path) if filename in valid_files]

    dataset = {filename : parse_ner_file(os.path.join(abs_path, filename), pos_exist) for filename in files}

    return dataset





