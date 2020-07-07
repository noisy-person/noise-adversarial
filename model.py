import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import pickle
PAD_IDX = 1

class NoiseModel(nn.Module):
    def __init__(self, num_class: int):
        super(NoiseModel, self).__init__()
        self.num_class = num_class
        self.transition_mat = Parameter(self.num_class*torch.eye(num_class))

    def forward(self, x):
        """
        x:
            shape = (batch, num_class) (probability distribution)
        return:
            noise distribution
        """
        out = torch.matmul(x, self.transition_mat.T)

        return out



class CNN_Text_adv(nn.Module):
    
    def __init__(self, FLAGS):
        super(CNN_Text_adv, self).__init__()

        self.FLAGS = FLAGS
        
        V = FLAGS.embed_num
        D = FLAGS.embed_dim
        C = FLAGS.class_num
        Ci = 1
        Co = FLAGS.kernel_num
        Ks = FLAGS.kernel_sizes



        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        
        for conv in self.convs1:
            nn.init.kaiming_normal_(conv.weight)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(FLAGS.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        nn.init.kaiming_normal_(self.fc1.weight.data)




    def forward(self, x):
        

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)


        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)

        return logit



class CNN_Text(nn.Module):
    
    def __init__(self, FLAGS):
        super(CNN_Text, self).__init__()
        with open(FLAGS.emb_path,'rb') as f:
            pretrained=pickle.load(f)

        self.FLAGS = FLAGS
        
        V = FLAGS.embed_num
        D = FLAGS.embed_dim
        C = FLAGS.class_num
        Ci = 1
        Co = FLAGS.kernel_num
        Ks = FLAGS.kernel_sizes

        self.embed = nn.Embedding(V, D,padding_idx=1)
        self.embed.weight.data.copy_(torch.FloatTensor(pretrained))
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        
        for conv in self.convs1:
            nn.init.kaiming_normal_(conv.weight)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(FLAGS.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        nn.init.kaiming_normal_(self.fc1.weight.data)




    def infer(self, x):
        # x.size() =  (N, W, D)
        


        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)


        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)

        return logit

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)


        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)

        return logit

class CNN_Text_Noise(nn.Module):
    
    def __init__(self, FLAGS):
        super(CNN_Text_Noise, self).__init__()
        with open(FLAGS.emb_path,'rb') as f:
            pretrained=pickle.load(f)

        self.FLAGS = FLAGS
        
        V = FLAGS.embed_num
        D = FLAGS.embed_dim
        C = FLAGS.class_num
        Ci = 1
        Co = FLAGS.kernel_num
        Ks = FLAGS.kernel_sizes

        self.embed = nn.Embedding(V, D,padding_idx=1)
        self.embed.weight.data.copy_(torch.FloatTensor(pretrained))
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        for conv in self.convs1:
            nn.init.kaiming_normal_(conv.weight)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(FLAGS.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        nn.init.kaiming_normal_(self.fc1.weight.data)

        self.nm = NoiseModel(num_class=C)
        self.transition = nn.Linear(C,C)
        self.transition.weight.data.copy_(torch.eye(C))
   


    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        if self.FLAGS.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        clean_softmax=nn.functional.softmax(logit,dim=-1)
        #print(clean_softmax)
        #print(self.transition)
        #noise_logit=self.transition(clean_softmax)
        noise_logit = self.nm(clean_softmax)
        return noise_logit,logit


