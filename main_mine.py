#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train_mine as train
import mydatasets
import pickle 
from absl import app, flags, logging
import sys


FLAGS = flags.FLAGS

# dataset
flags.DEFINE_string('dataset', 'ag_news', '')
flags.DEFINE_string('data_path', './data', '')
flags.DEFINE_string('emb_path', '../noisy-sequence/glove/glove.840B.300d.txt', '')
flags.DEFINE_string('voc_path', '../noisy-sequence/data/ag_news/vocab.txt', '')



def main(argv):
    logging.info('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info))
    train_iter=pickle.load( \
        open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_train_iter.pkl'), mode='rb'))
    dev_iter=pickle.load( \
        open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_dev_iter.pkl'), mode='rb'))
    test_iter=pickle.load( \
        open(os.path.join(FLAGS.data_path, FLAGS.dataset + '_test_iter.pkl'), mode='rb'))

    

if __name__ == '__main__':
    app.run(main)

