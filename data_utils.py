from torch.utils.data import DataLoader, Dataset
from torch._utils import _accumulate
import random
import gensim
import numpy as np
from absl import app, flags, logging
import sys

def generate_noisematrix( n_class,noise_rate):

    def minDiagonalSwap(nums, index):
        """ Return inplace swaped.
        Swap the diagonal element with the minimum element in the array
        return inplace altered list.
        """
        ind = np.argmin(nums)
        temp = nums[index]
        nums[index] = nums[ind]
        nums[ind] = temp
        return
   
    
    noise_matrix = np.zeros([n_class, n_class], dtype='float')

    if n_class <= 2:
        noise_matrix =  noise_rate*(np.ones(n_class, dtype='float') - np.eye(n_class, dtype='float'))
    else:
        # Defines random noise over unit simplex
        
        for a in range(n_class):
            nums = [np.random.randint(0, 10)*1.0 for x in  range(n_class)]
            #print(nums)
            #print(nums[1]/sum(np.array(nums)))
            nums = noise_rate*(nums/sum(np.array(nums)))
            #print(nums)
            minDiagonalSwap(nums, a)
            noise_matrix[a, :] = nums
            #print(noise_matrix)
    return noise_matrix

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
        #glove = GloVe(name='840B', dim=300)
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