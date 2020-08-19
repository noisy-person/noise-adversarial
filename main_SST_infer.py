#! /usr/bin/env python
import os
import datetime
import torch

import model

import pickle 
from absl import app, flags, logging
import sys
from data_utils import Subset
from utils import get_length

from torch.nn.utils.rnn import pad_sequence
import collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(2)

PAD_IDX = 1


FLAGS = flags.FLAGS
#AG_NEWS vocab 50002 / class 4
#DBpedia vocab 50002 / class 14 /kernelnum 400 
#TREC vocab 8982 / class 6 / batch 50 /patience 12 / lr0005/ eps 2.5 
#SST vocab 19538 / class 2 / 3,4,5 / feature size 100 /batch 100
# dataset hyperparameter
"""
flags.DEFINE_string('dataset', 'AG_NEWS', '')
flags.DEFINE_string('data_path', './data/AG_NEWS', '')
flags.DEFINE_string('emb_path', './data/AG_NEWS/AG_NEWS_emb.pkl', '')
flags.DEFINE_integer('embed_num', 50002, '')
flags.DEFINE_integer('class_num', 4, '')

flags.DEFINE_string('dataset', 'TREC', '')
flags.DEFINE_string('data_path', './data/TREC', '')
flags.DEFINE_string('emb_path', './data/TREC/TREC_emb.pkl', '')
flags.DEFINE_integer('embed_num', 8982, '')
flags.DEFINE_integer('class_num', 6, '')

flags.DEFINE_string('dataset', 'DBpedia', '')
flags.DEFINE_string('data_path', './data/DBpedia', '')
flags.DEFINE_string('emb_path', './data/DBpedia/DBpedia_emb.pkl', '')
flags.DEFINE_integer('embed_num', 50002, '')
flags.DEFINE_integer('class_num', 14, '')
"""
flags.DEFINE_string('dataset', 'SST2', '')
flags.DEFINE_string('data_path', './data/SST', '')
#flags.DEFINE_string('emb_path', './data/SST/SST_emb.pkl', '')

flags.DEFINE_string('emb_path', '../baseline/baseline/python/mead/SST2_pretrained_w2v.p', '')
flags.DEFINE_integer('embed_num', 17241, '')
flags.DEFINE_integer('class_num', 2, '')


# learning
flags.DEFINE_float('lr',  default=1.0, help='initial learning rate [default: 0.001]')
flags.DEFINE_integer('epochs',  default=20, help='number of epochs for train [default: 256]')
flags.DEFINE_integer('batch_size',  default=50, help='batch size for training [default: 64]')
flags.DEFINE_integer('log_interval',   default=1,   help='how many steps to wait before logging training status [default: 1]')
flags.DEFINE_integer('test_interval',  default=100, help='how many steps to wait before testing [default: 100]')
flags.DEFINE_integer('save_interval',  default=500, help='how many steps to wait before saving [default:500]')
flags.DEFINE_string('save_dir',  default='snapshot', help='where to save the snapshot')
flags.DEFINE_integer('early_stop',  default=300, help='iteration numbers to stop without performance increasing')
flags.DEFINE_bool('save_best',default=True, help='whether to save when get best performance')
flags.DEFINE_integer('ngram',  default=2, help='include ngram vocab')


# model
flags.DEFINE_string('mode',  default='transition', help='choose one of [transition,clean,adv] dont forget to import different train code ')
flags.DEFINE_string('noise_mode',  default='uni', help='uni or rand')
flags.DEFINE_float('dropout',  default=0.5, help='the probability for dropout [default: 0.5]')
flags.DEFINE_float('max_norm',  default=3.0, help='l2 constraint of parameters [default: 3.0]')
flags.DEFINE_integer('embed_dim',  default=300, help='number of embedding dimension [default: 128]')
flags.DEFINE_integer('kernel_num',  default=100, help='number of each kind of kernel')
flags.DEFINE_string('kernel_sizes',  default='3,4,5', help='comma-separated kernel size to use for convolution')
flags.DEFINE_bool('static',  default=False, help='fix the embedding')

# optionasd
flags.DEFINE_string('snapshot',  default='snapshot/2020-08-12_07-33-47/best_SST2_adv_uni_0.2_1.0_0.5_steps_900.pt', help='filename of model snapshot [default: None] ex)snapshot/2020-07-07_09-14-42/best_steps_22100.pt')
#flags.DEFINE_string('snapshot',  default='snapshot/2020-07-26_10-32-47/best_steps_3900.pt', help='filename of model snapshot [default: None] ex)snapshot/2020-07-07_09-14-42/best_steps_22100.pt')
flags.DEFINE_string('predict',  default=None, help='predict the sentence given')
flags.DEFINE_bool('test',  default=True, help='train or test')
flags.DEFINE_integer('patience',  default=5, help='the probability for dropout [default: 0.5]')
flags.DEFINE_float('noise_rate',  default=0.1, help='the probability for dropout [default: 0.5]')
flags.DEFINE_float('epsilon',  default=1.15, help='the probability for dropout [default: 0.5]')
flags.DEFINE_bool('fake',  default=True, help='fake dataset')
flags.DEFINE_bool('multi_gpu',  default=False, help='use multi gpus')


def main(argv):
    logging.info('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info))
    
    if FLAGS.mode =='transition':
        
        import train
    elif FLAGS.mode == 'adv':
        import train_adv as train
    elif FLAGS.mode =='clean':
        import train_clean as train
    else :
        print("wrong mode")


    train_iter = pickle.load( \
        open(f"../baseline/baseline/python/mead/{FLAGS.dataset}_dataset_train_{FLAGS.noise_rate}_{FLAGS.noise_mode}.p", mode='rb'))
    dev_iter = pickle.load( \
        open(f"../baseline/baseline/python/mead/{FLAGS.dataset}_dataset_dev_{FLAGS.noise_rate}_{FLAGS.noise_mode}.p", mode='rb'))
    test_iter = pickle.load( \
        open(f"../baseline/baseline/python/mead/{FLAGS.dataset}_dataset_test_{FLAGS.noise_rate}_{FLAGS.noise_mode}.p", mode='rb'))


    # update FLAGS and print

    FLAGS.kernel_sizes = [int(k) for k in FLAGS.kernel_sizes.split(',')]
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\nParameters:")
    for attr, value in FLAGS.__flags.items():
        print("\t{}={}".format(attr.upper(), value.value))


    # model
    if FLAGS.mode=='adv':
        cnn = model.CNN_Text_adv(FLAGS)
    elif FLAGS.mode == 'transition':
        cnn = model.CNN_Text_Noise(FLAGS)
    else :
        cnn = model.CNN_Text(FLAGS)

    if FLAGS.snapshot is not None:
        print('\nLoading model from {}...'.format(FLAGS.snapshot))
        cnn.load_state_dict(torch.load(FLAGS.snapshot))
        FLAGS.batch_size=1
    cnn = cnn.to(device)

    if FLAGS.snapshot == None:
        train.train(train_iter,dev_iter, cnn, FLAGS)
    print("test")
    if FLAGS.test == True:

        train.test(test_iter, cnn, FLAGS)
    

if __name__ == '__main__':
    app.run(main)

