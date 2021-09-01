import torch
from tqdm import tqdm
import re
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn.functional as F
import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
        
    # wiki_set    
    parser.add_argument('--wiki_set', default='21k_true_wiki_sents_vis_sec_clu', help='(non)-filtered Wikipedia dataset')

    # pooling
    parser.add_argument('--pool', default='avg_pool', help='1) cls_pool: A [CLS] token used for the sentence embedding, 2) avg_pool: averaging words for the sentence embedding ')

    # filter
    parser.add_argument('--flt', default='vis_sec_clu', help='Types of filtering - 1) vis_sec: visual sections, 2) vis_clu: visual clusters, 3) vis_sec_clu: visual sections and clusters, 4) no_filter: NO filtering')

    # maximum sequence length
    parser.add_argument('--max_seq_len', type=int, default=64, help='Maximum sequence length for an input sentence')

    # the number of sentences
    parser.add_argument('--max_sent', default='all', help='the number of sentences in a document used to build the semantic representation of a class')

    args = parser.parse_args()
    return args

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

args = parse_args()
data_name = args.wiki_set
case = args.flt
pooling = args.pool
max_seq_len = args.max_seq_len
max_sents = args.max_sent # 'all' or number

path = 'wiki_bert_' + str(max_sents) + '_sents_' + case + '_' + pooling + '/'
if not os.path.exists(path):
    os.makedirs(path)


d = json.load(open(args.wiki_set + '.json'))

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


p_d = []
not_included_sents = []
for i in tqdm(range(len(d))):
    sents = d[i]
    e = []
    cnt = 0

    if max_sents=='all':
        n_sents= len(sents)
    else:
        n_sents = max_sents

    if len(sents) < n_sents:
        for j in range(len(sents)):
            #check sent contains english words
            if None != re.search('[a-zA-Z]', sents[j]):
                ## pre-processing
                # remove section names(e.g. '== Range ==\n')
                text = re.sub('^.*=\n',"", sents[j], flags=re.DOTALL).lstrip()
                # remove strings inside brackets or parentheses and add '[CLS]' and '[SEP]'
                text = '[CLS] ' + re.sub("[\(\[].*?[\)\]]", "", text) + ' [SEP]'
                tokenized_text = tokenizer.tokenize(text)
                if len(tokenized_text) <= max_seq_len:
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    segments_ids = [0]*len(tokenized_text)
                    segments_tensors = torch.tensor([segments_ids])
                    tokens_tensors = torch.tensor([indexed_tokens])
                else: # larger than max_seq_len
                    tokenized_text= tokenized_text[0: max_seq_len-1]
                    sep_str = '[SEP]'
                    tokenized_text.append(sep_str)
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    tokens_tensors = torch.tensor([indexed_tokens])
                    segments_ids = [0]*len(tokenized_text)                
                    segments_tensors = torch.tensor([segments_ids])
                tokens_tensors = tokens_tensors.to('cuda')
                segments_tensors = segments_tensors.to('cuda')

                ## run BERT
                model.to('cuda')
                with torch.no_grad():
                    if pooling == 'cls_pool':
                        encoded_layers, _ = model(tokens_tensors, segments_tensors, output_all_encoded_layers=False)
                        if cnt==0:
                            tmp = F.normalize(encoded_layers[0][0],dim=0)
                            sents_v = torch.unsqueeze(tmp, dim=0)
                        else:
                            tmp = F.normalize(encoded_layers[0][0],dim=0)
                            tmp = torch.unsqueeze(tmp,dim=0)
                            sents_v = torch.cat((sents_v,tmp), dim=0)
                    elif pooling == 'avg_pool':
                        encoded_layers, _ = model(tokens_tensors, segments_tensors)
                        avg_v = torch.mean(encoded_layers[-2][0], dim=0)
                        if cnt==0:
                            tmp = F.normalize(avg_v, dim=0)
                            sents_v = torch.unsqueeze(tmp, dim=0)
                        else:
                            tmp = F.normalize(avg_v, dim=0)
                            tmp = torch.unsqueeze(tmp, dim=0)
                            sents_v = torch.cat((sents_v, tmp), dim=0)
                    cnt+=1
            else:
                e.append(j)

        sents_v = torch.mean(sents_v, dim=0)

    else:
        for j in range(n_sents):
            # check sent contains english words
            if None != re.search('[a-zA-Z]', sents[j]):
                ## pre_processing
                # remove section names(e.g. '== Range ==\n')
                text = re.sub('^.*=\n',"", sents[j], flags=re.DOTALL).lstrip()
                # remove strings inside brackets or parentheses and add '[CLS]' and '[SEP]'
                text = '[CLS] ' + re.sub("[\(\[].*?[\)\]]", "", text) + ' [SEP]'
                tokenized_text = tokenizer.tokenize(text)
                if len(tokenized_text) <= max_seq_len:
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    segments_ids = [0]*len(tokenized_text)
                    segments_tensors = torch.tensor([segments_ids])
                    tokens_tensors = torch.tensor([indexed_tokens])
                else: # larger than max_seq_len
                    tokenized_text= tokenized_text[0: max_seq_len-1]
                    sep_str = '[SEP]'
                    tokenized_text.append(sep_str)
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    tokens_tensors = torch.tensor([indexed_tokens])
                    segments_ids = [0]*len(tokenized_text)                
                    segments_tensors = torch.tensor([segments_ids])
                tokens_tensors = tokens_tensors.to('cuda')
                segments_tensors = segments_tensors.to('cuda')

                ## run BERT
                model.to('cuda')
                with torch.no_grad():
                    if pooling == 'cls_pool':
                        encoded_layers, _ = model(tokens_tensors, segments_tensors, output_all_encoded_layers=False)
                        if cnt==0:
                            tmp = F.normalize(encoded_layers[0][0],dim=0)
                            sents_v = torch.unsqueeze(tmp,dim=0)
                        else:
                            tmp = F.normalize(encoded_layers[0][0],dim=0)
                            tmp = torch.unsqueeze(tmp, dim=0)
                            sents_v = torch.cat((sents_v, tmp), dim=0)
                    elif pooling == 'avg_pool':
                        encoded_layers, _ = model(tokens_tensors, segments_tensors)
                        avg_v = torch.mean(encoded_layers[-2][0], dim=0)
                        if cnt==0:
                            tmp=F.normalize(avg_v, dim=0)
                            sents_v = torch.unsqueeze(tmp,dim=0)
                        else:
                            tmp = F.normalize(avg_v, dim=0)
                            tmp = torch.unsqueeze(tmp, dim=0)
                            sents_v = torch.cat((sents_v,tmp), dim=0)
                    cnt+=1
            else:
                e.append(j)
        
        sents_v = torch.mean(sents_v, dim=0)

    if len(sents_v) !=0:
        if i==0:
            sents_c = torch.unsqueeze(sents_v, dim=0)
        else:
            sents_c = torch.cat((sents_c,torch.unsqueeze(sents_v, dim=0)), dim=0)


# SAVE
torch.save(sents_c, path+'bert_sem.pt')
