#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import pickle
import train
import collections
import mydatasets
from torchtext.datasets import text_classification
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
if not os.path.isdir('data'):
    os.mkdir('data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=40, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=100, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=300, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-ngram', type=int, default=2, help='include ngram vocab')

# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()
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

def generate_batch(data):
    fields=collections.namedtuple(  # pylint: disable=invalid-name
            "fields", ["text", "label"])
    labels,texts = list(zip(*data))
 
    
    padded_texts= pad_sequence(list(texts),batch_first=True,padding_value=PAD_IDX)

    texts = torch.cat(texts,dim=-1)

    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    return fields(text=padded_texts, label=torch.LongTensor(labels))
#load AG_NEWS
def AG_NEWS(text_field, label_field,  **kargs):


    #train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    #    root='./data',  ngrams=args.ngram ,vocab=None)
    with open("./data/AGNEWS.p",'rb') as f:
        (train_dataset,test_dataset)=pickle.load(f)

    train_len = int(len(train_dataset) * 0.9)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    train_iter = DataLoader(sub_train_, batch_size=args.batch_size, shuffle=True,
                        collate_fn=generate_batch)
    valid_iter = DataLoader(sub_valid_, batch_size=args.batch_size, shuffle=True,
                    collate_fn=generate_batch)
    test_iter = DataLoader(test_dataset, batch_size=len(test_dataset), 
                    collate_fn=generate_batch)
    return train_iter, valid_iter,test_iter , len(train_dataset.get_vocab()), len(test_dataset.get_labels())

    
# load SST dataset
def sst(text_field, label_field,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter 


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter


# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
#train_iter,dev_iter, test_iter,vocab_size,class_size = AG_NEWS(text_field, label_field, device=-1, repeat=False)
with open("./data/ag_news_train_iter_0.7.pkl",'rb') as f:
    train_iter=pickle.load(f)
with open("./data/ag_news_dev_iter_0.7.pkl",'rb') as f:
    dev_iter=pickle.load(f)
with open("./data/ag_news_test_iter_0.7.pkl",'rb') as f:
    test_iter=pickle.load(f)

#train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
#train_iter, dev_iter, test_iter = sst(text_field, label_field, device=None, repeat=False)
"""
for i,batch in enumerate(train_iter):
    feature, target = batch.text, batch.label
    #print(list(text_field.vocab.stoi.keys())[:10])
    print(feature,feature.shape)
    print(target,target.shape)
    if(i==2):
        break
"""
# update args and print
#args.embed_num = len(text_field.vocab)
args.embed_num = 30002
#args.class_num = len(label_field.vocab) - 1
args.class_num = 4
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
cnn = model.CNN_Text_Noise(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

cnn = cnn.to(device)
        

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field )
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, test_iter, cnn, args)
        print("test")
        train.test(test_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

