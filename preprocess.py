import torch
import torchtext
from torchtext.datasets import text_classification
from torch.nn.utils.rnn import pad_sequence
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import DataLoader, Dataset

import gensim
import numpy as np
from absl import app, flags, logging
import sys

import pickle
import os
import random
import collections



FLAGS = flags.FLAGS

# dataset
flags.DEFINE_string('dataset', 'ag_news', '')
flags.DEFINE_string('data_path', './data/ag_news', '')
flags.DEFINE_string('emb_path', './data/glove.840B.300d.txt', '')
flags.DEFINE_string('voc_path', './data/ag_news/vocab.txt', '')

flags.DEFINE_integer('ngram', 2, '')
# model parameters
flags.DEFINE_integer('emb_dim', 300, '')

#mode 
flags.DEFINE_bool('generate_dataset', False, '')
flags.DEFINE_bool('generate_noise_dataset', True, '')
flags.DEFINE_bool('generate_pretrained', False, '')

#noise 
flags.DEFINE_float('noise_rate', 0.7, '')
flags.DEFINE_string('noise_mode', 'sym', '')
flags.DEFINE_float('train_rate', 0.9, '')
flags.DEFINE_integer('train_size', 108000, '')

#hyperparameter
flags.DEFINE_integer('vocab_size', 30002, '')
flags.DEFINE_integer('batch_size', 100, '')

flags.DEFINE_integer('class_size', 4, '')

PAD_IDX = 1


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

def get_length(feature):
    seq_length = feature.size()[1]
    one_sum = torch.sum( \
        torch.eq( \
            feature,torch.ones(feature.size(),dtype=torch.long)).long() \
                ,-1)

    return seq_length-one_sum

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
        random.shuffle(splited)
        splited_dataset.append(Subset(dataset, splited))
    
    return splited_dataset

def generate_noiselabels(label):
    #shuffle the index and split train/dev first and split noise /not noise after 

    label_size=len(label)
    shuffeled_idx = randperm(label_size).tolist()

    
    train_size=int(label_size*FLAGS.train_rate)
    train_idx=shuffeled_idx[:train_size]
    #random.shuffle(train_idx)

    noise_size = int(FLAGS.noise_rate*train_size)   
    noise_idx = train_idx[:noise_size]
    noised_labels = []
 
    for i in range(label_size):
        if i in noise_idx:
            
            if FLAGS.noise_mode=='sym':
                if FLAGS.dataset=='ag_news': 
                    numbers = list(range(0,4))
                    numbers.remove(label[i])
                    noiselabel=random.choice(numbers)

                    #print(noiselabel)

                noised_labels.append(noiselabel)

        else:    
            noised_labels.append(label[i]) 
    return noised_labels,shuffeled_idx

def generate_noiselabels_withdevnoise(label):
    #shuffle the index and split train/dev first and split noise /not noise after 

    label_size=len(label)
    shuffeled_idx = randperm(label_size).tolist()
    
    
    train_size=int(label_size*FLAGS.train_rate)
    train_idx=shuffeled_idx[:train_size]

    dev_size = label_size-train_size 
    dev_idx=shuffeled_idx[train_size:]

    train_noise_size = int(FLAGS.noise_rate*train_size)  
    dev_noise_size = int(FLAGS.noise_rate*dev_size)


    noise_idx = train_idx[:train_noise_size] + dev_idx[:dev_noise_size]
    noised_labels = []

    for i in range(label_size):
        if i in noise_idx:
            
            if FLAGS.noise_mode=='sym':
                if FLAGS.dataset=='ag_news': 
                    numbers = list(range(0,4))
                    numbers.remove(label[i])
                    noiselabel=random.choice(numbers)

                    #print(noiselabel)

                noised_labels.append(noiselabel)

        else:    
            noised_labels.append(label[i]) 
    return noised_labels,shuffeled_idx


def generate_batch(data):
    fields=collections.namedtuple(  # pylint: disable=invalid-name
            "fields", ["text", "label","input_length"])
    labels,texts = list(zip(*data))
 
    
    padded_texts= pad_sequence(list(texts),batch_first=True,padding_value=PAD_IDX)

    input_length = get_length(padded_texts)

    return fields(text=padded_texts, label=torch.LongTensor(labels),input_length = input_length)


#load AG_NEWS  training samples is 108000/12000 and testing 7,600.
def AG_NEWS_noisedata(train_dataset, test_dataset, check_noise_data, **kwargs):

    

    logging.info("before")
    before = list(zip(*train_dataset._data))[0][:100]
    logging.info(before) # (label, input_ids)으로 구성 
    unzip_data=list(zip(*train_dataset._data)) #분리
    
    logging.info("generate noise")
    noised_labels, shuffeled_idx=generate_noiselabels_withdevnoise(list(unzip_data[0]))

    logging.info("update dataset")
    noise_updated_data = list(zip(*[tuple(noised_labels),unzip_data[1]]))
    train_dataset._data=noise_updated_data

    logging.info("after")
    logging.info(list(zip(*train_dataset._data))[0][:100])

    if check_noise_data:
        #check if it is true
        count=0
        print(shuffeled_idx[:100])
        asdf=[]
        for i in range(120000):#check every data 
            if before[i][0]!=train_dataset[i][0] :
                if i in shuffeled_idx[:int(10800*FLAGS.noise_rate)]:
                    asdf.append(i)

        print(f'the size of noise sample is {len(asdf)}')
        print("noise data check finished")
    
    


    train_len = int(len(train_dataset) * FLAGS.train_rate)
    sub_train_, sub_valid_ = \
        data_split(train_dataset, [train_len, len(train_dataset) - train_len],shuffeled_idx)

    train_iter = DataLoader(sub_train_, batch_size=FLAGS.batch_size, shuffle=True,
                        collate_fn=generate_batch)
    valid_iter = DataLoader(sub_valid_, batch_size=FLAGS.batch_size, shuffle=True,
                    collate_fn=generate_batch)
    test_iter = DataLoader(test_dataset, batch_size=len(test_dataset), 
                    collate_fn=generate_batch)
    return train_iter, valid_iter,test_iter 


def build_trained_embedding(emb_path: str, emb_dim: int, vocab: dict) -> np.ndarray:
    """
    Extract pre-trained word embeddings from file (.pkl file)
    Argument:
        emb_path: embedding path
        vocab: vocabulary for current dataset
    Return:
        numpy array which has 2-D embeddings (|vocab|, D)
    """
    iv = 0
    emb = np.random.uniform(low=-0.25, high=0.25, size=(len(vocab), emb_dim))
    
    if 'glove' in emb_path:
        with open(emb_path, mode='r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.replace('\n', '').split(' ')
                word, vec = line[0], line[1:]
                if word in vocab:
                    emb[vocab[word]] = np.asarray(vec)
                    iv += 1
    elif 'Google' in emb_path:
        model = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=True)
        for word in vocab:
            if word in model.vocab:
                emb[vocab[word]] = model.wv[word]
                iv += 1
    logging.info('Pre-trained embedding loaded [OOV = {}]'.format((len(vocab) - iv) / len(vocab)))
    return emb

def main(argv):
    del argv  # Unused.
    logging.info('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info))


    if FLAGS.generate_dataset ==True:
        #just for full size vocabulary 
        train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'] \
            (root=FLAGS.data_path,  ngrams=FLAGS.ngram ,vocab=None)
        vocab=train_dataset.get_vocab()# ['<unk>', '<pad>'] are first
        
        #generate new vocab that satisfy the max size 
        new_vocab = torchtext.vocab.Vocab(counter=vocab.freqs, max_size=30000, min_freq=5)
        #generate train_dataset with new_vocab
        new_train_dataset, new_test_dataset = text_classification.DATASETS['AG_NEWS'] \
            (root=FLAGS.data_path,  ngrams=FLAGS.ngram ,vocab=new_vocab)
        pickle.dump((new_train_dataset,new_test_dataset),\
            open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_clean.pkl'),mode='wb'))        
        logging.info('finished generating vocab and dataset  and saved')

        if FLAGS.generate_pretrained ==True:
            emb = build_trained_embedding(emb_path=FLAGS.emb_path, emb_dim=FLAGS.emb_dim, \
                vocab=new_vocab.stoi)
            pickle.dump(emb, open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_emb.pkl'), mode='wb'))   
            logging.info('finished generating pretrained embeddings and saved')
        

    if FLAGS.generate_noise_dataset ==True:
        (train_dataset,test_dataset)=pickle.load( \
            open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_clean.pkl'), mode='rb'))
        train_iter,dev_iter, test_iter = \
            AG_NEWS_noisedata(train_dataset, test_dataset, False)

        logging.info("save noisy labels to %s ..."%os.path.join(FLAGS.data_path, FLAGS.dataset))        
        pickle.dump(train_iter, open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_train_iter_{FLAGS.noise_rate}.pkl'), mode='wb'))  
        pickle.dump(dev_iter, open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_dev_iter_{FLAGS.noise_rate}.pkl'), mode='wb'))   
        pickle.dump(test_iter, open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_test_iter_{FLAGS.noise_rate}.pkl'), mode='wb'))   

if __name__ == '__main__':
    app.run(main)

