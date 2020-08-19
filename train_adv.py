import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import pickle 
from utils import idx2emb, vat_loss,vat_loss_ours
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#logit - max 
def get_diff(logit,target):

    target_logit =torch.gather(logit,1,target.unsqueeze(-1))
    max_logit =  torch.max(logit,dim=1,keepdim=True)[0]
    mean_diff= torch.mean(torch.abs(target_logit-max_logit))

    return mean_diff.item()


# logit  - average logits over all mislabeled logits
def get_diff_2(logit,target):
    class_num = logit.shape[1]
    target_logit =torch.gather(logit,1,target.unsqueeze(-1))
    sum_logit =  torch.sum(logit,dim=1,keepdim=True)
    mean_diff = torch.abs((sum_logit -target_logit)/(class_num-1)-target_logit)
    mean_diff= torch.mean(mean_diff)

    return mean_diff.item()


#logit -max if target idx and max logit idx is different else logit 
def get_diff_3(logit,target):

    target_logit =torch.gather(logit,1,target.unsqueeze(-1))

    target_eq_max = torch.eq(target,torch.max(logit,dim=1)[1]).long()

    max_logit =  torch.max(logit,dim=1,keepdim=True)[0]
   
    diff = torch.abs(target_logit-(1-target_eq_max).unsqueeze(-1)*max_logit)
    mean_diff = torch.mean(diff)
    

    return mean_diff.item()



def train(train_iter, dev_iter, model, FLAGS):
    if FLAGS.multi_gpu :
        model = nn.DataParallel(model)

    model.to(device)

    #patience =0
    #optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=FLAGS.lr)
    decayRate = 0.9
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    steps = 0
    best_acc = 0
    last_step = 0
    if FLAGS.multi_gpu :
        _idx2emb = nn.DataParallel(idx2emb(FLAGS).to(device))
    else:
        _idx2emb = idx2emb(FLAGS).to(device)
    epsilon =FLAGS.epsilon

    patience=0
    for epoch in range(1, FLAGS.epochs+1):

        if patience>=FLAGS.patience and saved==0:
            break
        if epoch!=1 and saved==0:
            patience+=1 
        saved=0
        print(f"\nepoch : {epoch}")
            
        for batch in train_iter:
            model.train()
            #feature, target, input_length = batch.text.to(device), batch.label.to(device)  , batch.input_length.to(device) 
            feature, target,input_length = torch.from_numpy(batch['word']).to(device), \
                                           torch.from_numpy(batch['y']).to(device), \
                                            torch.from_numpy(batch['word_lengths']).to(device) 
            #feature.t_(), target.sub_(1)

                
            embedd_matrix = _idx2emb(feature)



            logit = model(embedd_matrix)
         
            #print('logit vector', logit.size())
            #print('target vector', target.size())
            ##if steps %1000==0 and steps> 2000:
            #    epsilon=epsilon-0.001
            #v_loss = vat_loss(model, embedd_matrix, logit,input_length, eps=epsilon)
            v_loss = vat_loss_ours(model, model.context_vec, logit,input_length, eps=epsilon)
            ce_loss = criterion(logit, target) 
            loss = ce_loss + v_loss

            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            steps += 1
            #if steps %1000==0 :
            #    my_lr_scheduler.step()
            
            if steps % FLAGS.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/feature.shape[0]


                writer.add_scalar('Accuracy/train',accuracy,steps)
                writer.add_scalar('Loss/train',loss.item(),steps)
                #writer.add_histogram('distribution centers', logit, steps)
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             feature.shape[0]))
            if steps % FLAGS.test_interval == 0:
                dev_acc = eval(dev_iter, model, steps, FLAGS)

                diff = get_diff(logit, target)
                diff_2 = get_diff_2(logit, target)
                diff_3 = get_diff_3(logit, target)
                writer.add_scalar('Diff/train',diff,steps)
                writer.add_scalar('Diff_2/train',diff_2,steps)
                writer.add_scalar('Diff_3/train',diff_2,steps)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if FLAGS.save_best:
                        print(f"best accuracy : {best_acc}")
                        if steps>3000:
                            save(model, FLAGS, 'best'+'_'+FLAGS.dataset+'_'+FLAGS.mode+'_'+FLAGS.noise_mode+'_'+str(FLAGS.noise_rate)+'_'+str(FLAGS.lr)+'_'+str(FLAGS.epsilon), steps)
                        saved=1
                        patience=0



def eval(data_iter, model, steps,FLAGS):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    corrects, avg_loss = 0, 0
    _idx2emb = idx2emb(FLAGS).to(device)

    mean_diff=0
    mean_diff_2=0
    mean_diff_3=0
    cnt=0
    for batch in data_iter:
        #feature, target, input_length = batch.text.to(device), batch.label.to(device)  , batch.input_length.to(device) 
        feature, target,input_length = torch.from_numpy(batch['word']).to(device), \
                                        torch.from_numpy(batch['y']).to(device), \
                                        torch.from_numpy(batch['word_lengths']).to(device) 
        #feature.t_(), target.sub_(1)
    
        #feature.t_(), target.sub_(1)
        embedd_matrix = _idx2emb(feature)
        logit = model(embedd_matrix)
        loss = criterion(logit, target)

        mean_diff += get_diff(logit, target)
        mean_diff_2 += get_diff_2(logit, target)
        mean_diff_3 += get_diff_3(logit, target)
        
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        cnt+=1

    size = cnt*FLAGS.batch_size
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    writer.add_scalar('Accuracy/eval',accuracy,steps)
    writer.add_scalar('Loss/eval',loss.item(),steps)
    writer.add_scalar('Diff/eval',mean_diff,steps)
    writer.add_scalar('Diff_2/eval',mean_diff_2,steps)
    writer.add_scalar('Diff_3/eval',mean_diff_3,steps)
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

def test(data_iter, model, FLAGS):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    corrects, avg_loss = 0, 0
    _idx2emb = idx2emb(FLAGS).to(device)
    cnt=0
    for batch in data_iter:
        #feature, target, input_length = batch.text.to(device), batch.label.to(device)  , batch.input_length.to(device) 
        feature, target,input_length = torch.from_numpy(batch['word']).to(device), \
                                        torch.from_numpy(batch['y']).to(device), \
                                        torch.from_numpy(batch['word_lengths']).to(device) 

        #feature.t_(), target.sub_(1)

        #feature.t_(), target.sub_(1)

        embedd_matrix = _idx2emb(feature)
        logit = model(embedd_matrix)
        loss = criterion(logit, target)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        cnt+=1

    size = cnt*FLAGS.batch_size
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nTest - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy
"""
def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    x = x.to(device)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]
"""

def save(model, FLAGS, save_prefix, steps):
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    save_prefix = os.path.join(FLAGS.save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    if FLAGS.multi_gpu:
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
