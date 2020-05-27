import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_iter, dev_iter, model, args):
    model.to(device)
    lambda_=0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text.to(device), batch.label.to(device)
            #feature.t_(), target.sub_(1)

            l2_norm=torch.norm(model.nm.transition_mat,p=2)
            #l2_norm=model.nm.transition_mat.norm(2)
            
            optimizer.zero_grad()
            noise_logit,clean_logit = model(feature)
            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss = criterion(noise_logit, target)+0.5*lambda_*l2_norm #fix with one l2_norm
            #print(loss)
            #
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(noise_logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/feature.shape[0]
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             feature.shape[0]))
            if steps % args.test_interval == 0:
                dev_acc = test(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text.to(device), batch.label.to(device)
        #feature.t_(), target.sub_(1)
        noise_logit,clean_logit = model(feature)
        loss = criterion(clean_logit, target)

        avg_loss += loss.item()
        corrects += (torch.max(clean_logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

def test(data_iter, model, args):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text.to(device), batch.label.to(device)
        #feature.t_(), target.sub_(1)
        noise_logit,clean_logit = model(feature)
        loss = criterion(clean_logit, target)

        avg_loss += loss.item()
        corrects += (torch.max(clean_logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\Test - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

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


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
