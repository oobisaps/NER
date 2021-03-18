__version__ = "0.2.0"

from pytorch_helpers import (NER_Dataset, 
                             Data_Pipeline, 
                             Bert_Markers_Adder, 
                             Pytorch_Wordpiece_Tokenizer, 
                             Token_To_Id_Transformer, 
                             Pad_Trunc_Sequence_Getter, 
                             print_results, 
                             load_vocab,
                             add_artefacts_tags, 
                             convert_tags_to_ids, 
                             make_attention_mask,
                             make_long_tensor)





from ml_helpers import Bert_For_Token_Classification, Model_Helper

x = True
y = False
z = False
if not x or y:
    print(1)
elif not x or not y and z:
    print(2)
elif not x or y or not y and x:
    print(3) 
else:
    print(4)