import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import pickle
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

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
    model.to(device)
    lambda_=0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    diff_list=[]
    acc_list=[]
    dev_diff_list=[]
    dev_acc_list=[]
    step_arr=[]
    for epoch in range(1, FLAGS.epochs+1):
        for batch in train_iter:
            feature, target,input_length = torch.from_numpy(batch['word']).to(device), \
                                           torch.from_numpy(batch['y']).to(device), \
                                            torch.from_numpy(batch['word_lengths']).to(device) 

            
            optimizer.zero_grad()
            logit = model(feature)

            loss = criterion(logit, target)


            #print(loss)
            #
            loss.backward()
            optimizer.step()

            steps += 1



            if steps % FLAGS.log_interval == 0:



                #print(torch.max(logit, 1))
                #print(target.data)
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/feature.shape[0]
                

                writer.add_scalar('Accuracy/train',accuracy,steps)
                writer.add_scalar('Loss/train',loss.item(),steps)

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
                        save(model, FLAGS, 'best', steps)
                else:
                    if steps - last_step >= FLAGS.early_stop:
                        
                        print('early stop by {} steps.'.format(FLAGS.early_stop))
                        break
            elif steps % FLAGS.save_interval == 0:
                save(model, FLAGS, 'snapshot', steps)
    pickle.dump((step_arr,diff_list,acc_list,dev_diff_list,dev_acc_list), open('visualize.pkl', mode='wb'))


def eval(data_iter, model, steps,FLAGS):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    corrects, avg_loss = 0, 0

    mean_diff=0
    mean_diff_2=0
    mean_diff_3=0

    for batch in data_iter:
        feature, target,input_length = torch.from_numpy(batch['word']).to(device), \
                                        torch.from_numpy(batch['y']).to(device), \
                                        torch.from_numpy(batch['word_lengths']).to(device) 
        logit = model(feature)
        loss = criterion(logit, target)

        mean_diff += get_diff(logit, target)
        mean_diff_2 += get_diff_2(logit, target)
        mean_diff_3 += get_diff_3(logit, target)
        
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter)*FLAGS.batch_size
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

    mean_diff=0
    
    for batch in data_iter:
        
        feature, target,input_length = torch.from_numpy(batch['word']).to(device), \
                                        torch.from_numpy(batch['y']).to(device), \
                                        torch.from_numpy(batch['word_lengths']).to(device)  
        #feature.t_(), target.sub_(1)
        clean_logit = model(feature)
        loss = criterion(clean_logit, target)


        mean_diff += get_diff(clean_logit,target)
        avg_loss += loss.item()
        corrects += (torch.max(clean_logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter)*FLAGS.batch_size
    avg_loss /= size
    mean_diff 
    accuracy = 100.0 * corrects/size
    print('\Test - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy,mean_diff
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
