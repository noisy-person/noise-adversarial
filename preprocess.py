import torch
import torchtext
from torchtext.datasets import text_classification
from torch.nn.utils.rnn import pad_sequence
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import DataLoader, Dataset
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

import gensim
import numpy as np
from absl import app, flags, logging
import sys

import pickle
import os
import random
import collections

from data_utils import build_trained_embedding,Subset,data_split


FLAGS = flags.FLAGS
#DBpedia:14
# dataset
flags.DEFINE_string('dataset', 'TREC', '') # AG_NEWS,TREC,DBpedia
flags.DEFINE_string('data_path', './data/TREC', '')
flags.DEFINE_string('emb_path', './data/glove/glove.840B.300d.txt', '')

flags.DEFINE_integer('ngram', 2, '')
# model parameters
flags.DEFINE_integer('emb_dim', 300, '')

#mode 
flags.DEFINE_bool('generate_dataset', True, '')
flags.DEFINE_bool('generate_noise_dataset', True, '')
flags.DEFINE_bool('generate_pretrained', True, '')

#noise 
flags.DEFINE_float('noise_rate', 0.7, '')
flags.DEFINE_string('noise_mode', 'sym', '')
flags.DEFINE_bool('fake', True, '')

flags.DEFINE_float('train_rate', 0.7, '')

#hyperparameter
flags.DEFINE_integer('vocab_size', 9000, '')
flags.DEFINE_integer('batch_size', 100, '')


PAD_IDX = 1






def generate_batch(data):
    def get_length(feature):
        seq_length = feature.size()[1]
        one_sum = torch.sum( \
            torch.eq( \
                feature,torch.ones(feature.size(),dtype=torch.long)).long() \
                    ,-1)

        return seq_length-one_sum
    fields=collections.namedtuple(  # pylint: disable=invalid-name
            "fields", ["text", "label","input_length"])
    #labels should be a list of labels and texts should be list of torch tensor 

    if FLAGS.dataset == "AG_NEWS":
        print(type(data))
        labels,texts = list(zip(*data))
    if FLAGS.dataset == "TREC":
        print(type(data))
        labels,texts = [record.label for record in data], [torch.LongTensor(record.text) for record in data]

    padded_texts= pad_sequence(list(texts),batch_first=True,padding_value=PAD_IDX)

    input_length = get_length(padded_texts)

    return fields(text=padded_texts, label=torch.LongTensor(labels),input_length = input_length)


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
                if FLAGS.dataset=='AG_NEWS': 
                    numbers = list(range(0,4))
                    if not FLAGS.fake :
                        numbers.remove(label[i])
                    noiselabel=random.choice(numbers)

                    #print(noiselabel)
                if FLAGS.dataset == 'TREC':
                    #label candidate
                    label_cand = ['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM']
                    if not FLAGS.fake :
                        label_cand.remove(label[i])
                    noiselabel=random.choice(label_cand)


                noised_labels.append(noiselabel)

        else:    
            noised_labels.append(label[i]) 
    return noised_labels,shuffeled_idx



def TREC_noisedata(train_dataset, **kwargs):

    

    logging.info("before")
    train_labels  = [record.label for record in train_dataset]
    logging.info(train_labels[:100]) # (label, input_ids)으로 구성 
    logging.info("generate noise")
    noised_labels, shuffeled_idx=generate_noiselabels_withdevnoise(train_labels)

    logging.info("update dataset")
    for i,record in enumerate(train_dataset):
        train_dataset[i].label = noised_labels[i]

    logging.info("after")
    logging.info(noised_labels[:100])



    train_len = int(len(train_dataset) * FLAGS.train_rate)
    sub_train_, sub_valid_ = \
        data_split(train_dataset, [train_len, len(train_dataset) - train_len],shuffeled_idx)
  
    return sub_train_, sub_valid_


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

def main(argv):
    del argv  # Unused.
    logging.info('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info))

    if FLAGS.dataset=='TREC':
        if FLAGS.generate_dataset ==True:
                                    
            def only6label(dataset):
                for data in dataset:
                    data.label = data.label.split(":")[0]
                return dataset 

            TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)
            LABEL = torchtext.data.Field(sequential=False)

            train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)  # text data as list 

            train = only6label(train)
            test = only6label(test)

    
            #pickle.dump(glove.vectors.numpy(), open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_emb.pkl'), mode='wb'))

            #consider both train and test vocabulary
            TEXT.build_vocab(train,test, max_size=FLAGS.vocab_size, min_freq=1)
            LABEL.build_vocab(train)

            #to save it to pickle convert generator to list
            train = list(train)
            test = list(test)
            


        if FLAGS.dataset is not None and FLAGS.generate_pretrained ==True:
            emb = build_trained_embedding(emb_path=FLAGS.emb_path, emb_dim=FLAGS.emb_dim, \
                vocab=TEXT.vocab.stoi)
            pickle.dump(emb, open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_emb.pkl'), mode='wb'))   
            logging.info('finished generating pretrained embeddings and saved')
        


        if FLAGS.generate_noise_dataset ==True:             
            #make noise!!!
            sub_train, sub_valid = TREC_noisedata(train)

            
            #change word into id 
            print(sub_train[0].text)
            for i in range(len(sub_train)):
                record = sub_train[i]

                record.text = [TEXT.vocab.stoi[text] for text in record.text]
                record.label= LABEL.vocab.stoi[record.label] 
            print(sub_train[0].text)
    
            for i in range(len(sub_valid)):
                record = sub_valid[i]

                record.text = [TEXT.vocab.stoi[text] for text in record.text]
                record.label= LABEL.vocab.stoi[record.label] 

        
            for i in range(len(test)):
                record = test[i]

                record.text = [TEXT.vocab.stoi[text] for text in record.text]
                record.label= LABEL.vocab.stoi[record.label] 


            train_iter = DataLoader(sub_train, batch_size=FLAGS.batch_size, shuffle=True,
                                collate_fn=generate_batch)
            dev_iter = DataLoader(sub_valid, batch_size=FLAGS.batch_size, shuffle=True,
                            collate_fn=generate_batch)
            test_iter = DataLoader(test, batch_size=len(test), 
                            collate_fn=generate_batch)
            
            print(next(iter(train_iter)))
        
    if FLAGS.dataset=='AG_NEWS' or FLAGS.dataset=='DBpedia':
        if FLAGS.generate_dataset ==True:
            #just for full size vocabulary 
            train_dataset, test_dataset = text_classification.DATASETS[FLAGS.dataset] \
                (root=FLAGS.data_path,  ngrams=FLAGS.ngram ,vocab=None)

            vocab_freqs=train_dataset.get_vocab().freqs + test_dataset.get_vocab().freqs# ['<unk>', '<pad>'] are first
            
            #generate new vocab that satisfy the max size 
            new_vocab = torchtext.vocab.Vocab(counter=vocab_freqs, max_size=FLAGS.vocab_size, min_freq=5)
            #generate train_dataset with new_vocab
            new_train_dataset, new_test_dataset = text_classification.DATASETS[FLAGS.dataset] \
                (root=FLAGS.data_path,  ngrams=FLAGS.ngram ,vocab=new_vocab)

            pickle.dump((new_train_dataset,new_test_dataset),\
                open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_clean.pkl'),mode='wb'))        
            logging.info('finished generating vocab and dataset  and saved')




            if FLAGS.dataset is not None and FLAGS.generate_pretrained ==True:
                emb = build_trained_embedding(emb_path=FLAGS.emb_path, emb_dim=FLAGS.emb_dim, \
                    vocab=new_vocab.stoi)
                pickle.dump(emb, open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_emb.pkl'), mode='wb'))   
                logging.info('finished generating pretrained embeddings and saved')
            

        if FLAGS.generate_noise_dataset ==True:
            (train_dataset,test_dataset)=pickle.load( \
                open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_clean.pkl'), mode='rb'))

            train_iter,dev_iter, test_iter = AG_NEWS_noisedata(train_dataset, test_dataset, False)
            print(next(iter(train_iter)))
                   

            
    logging.info("save noisy labels to %s ..."%os.path.join(FLAGS.data_path, FLAGS.dataset)) 
    #save it 
    if FLAGS.fake == True:

        pickle.dump(train_iter, open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_train_iter_fake_{FLAGS.noise_rate}.pkl'), mode='wb'))  
        pickle.dump(dev_iter, open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_dev_iter_fake_{FLAGS.noise_rate}.pkl'), mode='wb'))   
        pickle.dump(test_iter, open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_test_iter_fake_{FLAGS.noise_rate}.pkl'), mode='wb'))   
    else:
        pickle.dump(train_iter, open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_train_iter_{FLAGS.noise_rate}.pkl'), mode='wb'))  
        pickle.dump(dev_iter, open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_dev_iter_{FLAGS.noise_rate}.pkl'), mode='wb'))   
        pickle.dump(test_iter, open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_test_iter_{FLAGS.noise_rate}.pkl'), mode='wb'))   

if __name__ == '__main__':
    app.run(main)

