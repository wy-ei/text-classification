import os
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def train(model, optimizer, train_dl, val_dl, device=None, epochs=10):
    model.cuda(device)
        
    history = {
        'acc': [], 'loss': [],
        'val_acc': [], 'val_loss': []
    }
    
    batch_num = int(len(train_dl.dataset) / train_dl.batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        
        steps = 0
        total_loss = 0.
        correct_num = 0
        
        for (x, y) in train_dl:       
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            scores = model(x)
            
            loss = F.cross_entropy(scores, y) 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            y_pred = torch.max(scores, 1)[1]
            correct_num += (y_pred == y).sum().item()
                    
            steps += 1

            if steps % 100 == 0:
                info = 'epoch {:<2}: {:.2%}'.format(epoch, steps / batch_num)
                sys.stdout.write('\b' * len(info))
                sys.stdout.write(info)
                sys.stdout.flush()

        sys.stdout.write('\b' * len(info))
        sys.stdout.flush()

        train_acc = correct_num / len(train_dl.dataset) 
        train_loss = total_loss / len(train_dl.dataset)
        
        history['acc'].append(train_acc)
        history['loss'].append(train_loss)

        val_loss, val_acc = evaluate(model, val_dl, device=device)

        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        logger.info("epoch {} - loss: {:.2f} acc: {:.2f} - val_loss: {:.2f} val_acc: {:.2f}"\
              .format(epoch, train_loss, train_acc, val_loss, val_acc))
        
    return history


def predict(model, dl, device=None):
    model.eval()
    y_pred = []
    for x, _ in dl:        
        x = x.to(device)
        scores = model(x)
        y_pred_batch = torch.max(scores, 1)[1]
        y_pred.append(y_pred_batch)

    y_pred = torch.cat(y_pred, dim=0)
    return y_pred.cpu().numpy()


def evaluate(model, dl, device=None):
    model.eval()
    
    total_loss = 0.0
    correct_num = 0
    
    for x, y in dl:        
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        loss = F.cross_entropy(scores, y)
        
        total_loss += loss.item()
        y_pred = torch.max(scores, 1)[1]
        correct_num += (y_pred == y).sum().item()
    
    avg_loss = total_loss / len(dl.dataset)
    avg_acc = correct_num / len(dl.dataset)

    return avg_loss, avg_acc