import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import pickle 
from utils import idx2emb, vat_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_iter, dev_iter, model, FLAGS):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    _idx2emb = idx2emb(FLAGS).to(device)
    for epoch in range(1, FLAGS.epochs+1):
        for batch in train_iter:
            feature, target, input_length = batch.text.to(device), batch.label.to(device)  , batch.input_length.to('cpu') 
            #feature.t_(), target.sub_(1)

            

            embedd_matrix = _idx2emb(feature)
            logit = model(embedd_matrix)
            #print('logit vector', logit.size())
            #print('target vector', target.size())
            v_loss = vat_loss(model, embedd_matrix, logit,input_length, eps=2.5)
            ce_loss = criterion(logit, target) #+0.5*lambda_*l2_norm #fix with one l2_norm
            loss = ce_loss + v_loss

            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % FLAGS.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/feature.shape[0]
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             feature.shape[0]))
            if steps % FLAGS.test_interval == 0:
                dev_acc = test(dev_iter, model, FLAGS)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if FLAGS.save_best:
                        print(f"best accuracy : {best_acc}")
                        save(model, FLAGS.save_dir, 'best', steps)
                else:
                    if steps - last_step >= FLAGS.early_stop:
                        
                        print('early stop by {} steps.'.format(FLAGS.early_stop))
                        break
            elif steps % FLAGS.save_interval == 0:
                save(model, FLAGS.save_dir, 'snapshot', steps)


def eval(data_iter, model, FLAGS):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    corrects, avg_loss = 0, 0
    _idx2emb = idx2emb(FLAGS).to(device)

    for batch in data_iter:
        feature, target = batch.text.to(device), batch.label.to(device)
        #feature.t_(), target.sub_(1)
        
        embedd_matrix = _idx2emb(feature)
        logit= model(embedd_matrix)
        loss = criterion( logit, target)

        avg_loss += loss.item()
        corrects += (torch.max(logit,  1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
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

    for batch in data_iter:
        feature, target = batch.text.to(device), batch.label.to(device)
        #feature.t_(), target.sub_(1)
        embedd_matrix = _idx2emb(feature)
        logit = model(embedd_matrix)
        loss = criterion(logit, target)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\Test - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
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

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
