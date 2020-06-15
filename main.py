#! /usr/bin/env python
import os
import datetime
import torch

import model
#import train
import train_adv as train
#import train_clean as train
import pickle 
from absl import app, flags, logging
import sys
from data_utils import Subset
from utils import get_length

from torch.nn.utils.rnn import pad_sequence
import collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)

PAD_IDX = 1


FLAGS = flags.FLAGS

# dataset hyperparameter
flags.DEFINE_string('dataset', 'ag_news', '')
flags.DEFINE_string('data_path', './data/ag_news', '')
flags.DEFINE_string('emb_path', './data/glove/glove.840B.300d.txt', '')
flags.DEFINE_string('voc_path', './data/ag_news/vocab.txt', '')
flags.DEFINE_integer('embed_num', 30002, '')
flags.DEFINE_integer('class_num', 4, '')

# learning
flags.DEFINE_float('lr',  default=0.001, help='initial learning rate [default: 0.001]')
flags.DEFINE_integer('epochs',  default=40, help='number of epochs for train [default: 256]')
flags.DEFINE_integer('batch_size',  default=100, help='batch size for training [default: 64]')
flags.DEFINE_integer('log_interval',   default=1,   help='how many steps to wait before logging training status [default: 1]')
flags.DEFINE_integer('test_interval',  default=100, help='how many steps to wait before testing [default: 100]')
flags.DEFINE_integer('save_interval',  default=500, help='how many steps to wait before saving [default:500]')
flags.DEFINE_string('save_dir',  default='snapshot', help='where to save the snapshot')
flags.DEFINE_integer('early_stop',  default=300, help='iteration numbers to stop without performance increasing')
flags.DEFINE_bool('save_best',default=True, help='whether to save when get best performance')
flags.DEFINE_integer('ngram',  default=2, help='include ngram vocab')

# data 
flags.DEFINE_bool('shuffle',  default=False, help='shuffle the data every epoch')
# model
flags.DEFINE_string('mode',  default='adv', help='choose one of [transition,clean,adv] dont forget to import different train code ')
flags.DEFINE_float('dropout',  default=0.5, help='the probability for dropout [default: 0.5]')
flags.DEFINE_float('max_norm',  default=3.0, help='l2 constraint of parameters [default: 3.0]')
flags.DEFINE_integer('embed_dim',  default=300, help='number of embedding dimension [default: 128]')
flags.DEFINE_integer('kernel_num',  default=100, help='number of each kind of kernel')
flags.DEFINE_string('kernel_sizes',  default='3,4,5', help='comma-separated kernel size to use for convolution')
flags.DEFINE_bool('static',  default=False, help='fix the embedding')

# option
flags.DEFINE_string('snapshot',  default=None, help='filename of model snapshot [default: None] ex)snapshot/2020-06-13_13-10-02/best_steps_4000.pt')
flags.DEFINE_string('predict',  default=None, help='predict the sentence given')
flags.DEFINE_bool('train',  default=True, help='train or test')
flags.DEFINE_bool('test',  default=True, help='train or test')
flags.DEFINE_float('noise_rate',  default=0.7, help='the probability for dropout [default: 0.5]')



def generate_batch(data):
    fields=collections.namedtuple(  # pylint: disable=invalid-name
            "fields", ["text", "label","input_length"])
    labels,texts = list(zip(*data))
 
    
    padded_texts= pad_sequence(list(texts),batch_first=True,padding_value=PAD_IDX)

    input_length = get_length(padded_texts)

    return fields(text=padded_texts, label=torch.LongTensor(labels),input_length = input_length)

def main(argv):
    logging.info('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info))
    train_iter = pickle.load( \
        open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_train_iter_{FLAGS.noise_rate}.pkl'), mode='rb'))
    dev_iter = pickle.load( \
        open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_dev_iter_{FLAGS.noise_rate}.pkl'), mode='rb'))
    test_iter = pickle.load( \
        open(os.path.join(FLAGS.data_path, FLAGS.dataset + f'_test_iter_{FLAGS.noise_rate}.pkl'), mode='rb'))
    

    """
    for i,batch in enumerate(train_iter):
        feature, target, input_length = batch.text, batch.label,batch.input_length
        #print(list(text_field.vocab.stoi.keys())[:10])
        print(feature,feature.shape)
        print(input_length)
        print(target,target.shape)
        if(i==2):
            break
    """
    
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

    cnn = cnn.to(device)

    if FLAGS.train ==True:
        train.train(train_iter,dev_iter, cnn, FLAGS)
    print("test")
    if FLAGS.test == True:

        train.test(test_iter, cnn, FLAGS)
    

if __name__ == '__main__':
    app.run(main)

