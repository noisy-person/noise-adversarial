#! /usr/bin/env python
import os
import json
import random 
import argparse
import datetime
import torch
import numpy as np
import torchtext.data as data
import torchtext.datasets as datasets
import model
import pickle
import train
import collections
import mydatasets
from torchtext.datasets import text_classification
from torch.utils.data import DataLoader, Dataset
#from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
from torch._utils import _accumulate
from torch import randperm

if not os.path.isdir('data'):
    os.mkdir('data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=25, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=100, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=500, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-ngram', type=int, default=1, help='include ngram vocab')

# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

# preprocess
parser.add_argument('-pretrained_path', type=str, default='glove', help='comma-separated kernel size to use for convolution')


# noise
parser.add_argument('-noise_rate', type=int, default=0.7, help='noise rate')
parser.add_argument('-train_rate', type=int, default=0.9, help='how you set the train dev rate')
parser.add_argument('-train_size', type=int, default=108000, help='train data size')
parser.add_argument('-noise_file', type=str, default="sym", help='noise mode')
parser.add_argument('-noise_mode', type=str, default="sym", help='noise mode')
parser.add_argument('-dataset', type=str, default="AGNews", help='noise mode')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()
PAD_IDX = 1
np.random.seed()
def get_length(feature):
    seq_length = feature.size()[1]
    one_sum = torch.sum( \
        torch.eq( \
            feature,torch.ones(feature.size(),dtype=torch.long)).long() \
                ,-1)

    return seq_length-one_sum

class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def data_split(dataset, lengths,shuffeled_idx):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    
    splited_dataset=[]
    for offset, length in zip(_accumulate(lengths), lengths):
        splited = shuffeled_idx[offset - length:offset][:]
        print(splited)
        random.shuffle(splited)
        splited_dataset.append(Subset(dataset, splited))
    
    return splited_dataset


def generate_noiselabels(label):
    #shuffle the index and split train/dev first and split noise /not noise after 

    label_size=len(label)
    shuffeled_idx = randperm(label_size).tolist()

    #idx = list(range(label_size))

    
    train_size=int(label_size*args.train_rate)
    train_idx=shuffeled_idx[:train_size]
    #random.shuffle(train_idx)

    noise_size = int(args.noise_rate*train_size)   
    noise_idx = train_idx[:noise_size]
    noise_labels = []
    for i in range(label_size):
        if i in noise_idx:
            
            if args.noise_mode=='sym':
                if args.dataset=='AGNews': 
                    numbers = list(range(0,4))
                    #print(numbers)
                    numbers.remove(label[i])
                    #print(numbers)
                    noiselabel=random.choice(numbers)

                    #print(noiselabel)

                noise_labels.append(noiselabel)

        else:    
            noise_labels.append(label[i]) 
    print(noise_idx[:100])
    return noise_labels,shuffeled_idx


def generate_batch(data):
    fields=collections.namedtuple(  # pylint: disable=invalid-name
            "fields", ["text", "label"])
    labels,texts = list(zip(*data))
    #print(data)
    
    padded_texts= pad_sequence(list(texts),batch_first=True,padding_value=PAD_IDX)

    texts = torch.cat(texts,dim=-1)

    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    return fields(text=padded_texts, label=torch.LongTensor(labels))

def generate_batch_adv(data):
    fields=collections.namedtuple(  # pylint: disable=invalid-name
            "fields", ["text", "label","input_length"])
    labels,texts = list(zip(*data))
 
    
    padded_texts= pad_sequence(list(texts),batch_first=True,padding_value=PAD_IDX)

    input_length = get_length(padded_texts)

    return fields(text=padded_texts, label=torch.LongTensor(labels),input_length = input_length)

#load AG_NEWS  training samples is 108000/12000 and testing 7,600.
def AG_NEWS_noisedata( **kargs):

    
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
        root='./data',  ngrams=args.ngram ,vocab=None)
    with open("./data/AGNEWS.p",'rb') as f:
        pickle.dump((train_dataset,test_dataset),f)
    print(len(train_dataset.get_vocab()))
    #with open("./data/AGNEWS.p",'rb') as f:
    #    (train_dataset,test_dataset)=pickle.load(f)

    print("before")
    before_data=list(train_dataset._data)[:]
    seperate_data=list(zip(*train_dataset._data))
    
    #TODO train 만 noise 넣어주기 
    #print(id(a[0]))
    noise_labels, shuffeled_idx=generate_noiselabels(list(seperate_data[0]))

    #train_dataset._data = list(zip(*[tuple]))
    noise_updated_data = list(zip(*[tuple(noise_labels),seperate_data[1]]))
    train_dataset._data=noise_updated_data
    print("after")

    print(list(zip(*train_dataset._data))[0][:100])
    #print(noise_labels[:1200])
    #print(seperate_data[0][:1200])
    ##print("+++"*100)
    #print(noise_labels[-1200:])
    #print(seperate_data[0][-1200:])
    
    #check if it is true
    count=0
    print(shuffeled_idx[:100])
    asdf=[]
    for i in range(120000):#check every data 
        if before_data[i][0]!=train_dataset[i][0] :
            if i in shuffeled_idx[:32400]:
                asdf.append(i)

    print(len(asdf))
    print("finished")
    
    


    train_len = int(len(train_dataset) * args.train_rate)
    sub_train_, sub_valid_ = \
        data_split(train_dataset, [train_len, len(train_dataset) - train_len],shuffeled_idx)

    train_iter = DataLoader(sub_train_, batch_size=args.batch_size, shuffle=True,
                        collate_fn=generate_batch_adv)
    valid_iter = DataLoader(sub_valid_, batch_size=args.batch_size, shuffle=True,
                    collate_fn=generate_batch_adv)
    test_iter = DataLoader(test_dataset, batch_size=len(test_dataset), 
                    collate_fn=generate_batch_adv)
    return train_iter, valid_iter,test_iter , len(train_dataset.get_vocab()), len(test_dataset.get_labels())

# load data
print("\nLoading data...")
#text_field = data.Field(lower=True)
#label_field = data.Field(sequential=False)
train_iter,dev_iter, test_iter,vocab_size,class_size = AG_NEWS_noisedata(device=-1, repeat=False)

train_text=[]
train_label=[]

"""
count=0
for i,batch in enumerate(train_iter):
    feature, target = batch.text, batch.label

    train_text.append(feature)
    break
    #print(list(text_field.vocab.stoi.keys())[:10])
    #print(feature,feature.shape)
    #print(target,target.shape)
"""


print("save noisy labels to %s ..."%args.noise_file)        
with open("noise_data_"+str(args.noise_rate)+".p","wb") as f:
    pickle.dump((train_iter,dev_iter, test_iter,vocab_size,class_size),f)
#json.dump(noise_labels,open(noise_file,"w"))       