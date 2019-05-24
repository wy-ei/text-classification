import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import logging

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

from model import TextCNN
from data import build_dict, NewsDataSet, CATEGIRY_LIST


def train(model, optimizer, train_dl, val_dl, epochs=10):
    model.cuda(device)
        
    history = {
        'acc': [], 'loss': [],
        'val_acc': [], 'val_loss': []
    }
    
    for epoch in range(1, epochs + 1):
        model.train()

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
                    
        train_acc = correct_num / len(train_dl.dataset) 
        train_loss = total_loss / len(train_dl.dataset)
        
        history['acc'].append(train_acc)
        history['loss'].append(train_loss)

        val_loss, val_acc = evaluate(model, val_dl)

        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        logger.info("epoch: {} - loss: {:.2f} acc: {:.2f} - val_loss: {:.2f} val_acc: {:.2f}"\
              .format(epoch, train_loss, train_acc, val_loss, val_acc))
        
    return history


def predict(model, dl):
    model.eval()
    y_pred = []
    for x, _ in dl:        
        x = x.to(device)
        scores = model(x)
        y_pred_batch = torch.max(scores, 1)[1]
        y_pred.append(y_pred_batch)

    y_pred = torch.cat(y_pred, dim=0)
    return y_pred.cpu().numpy()


def evaluate(model, dl):
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


if __name__ == "__main__":
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    logger.info('using device: {}'.format(device))

    train_file = './datasets/cnews/cnews.train.txt'
    valid_file = './datasets/cnews/cnews.val.txt'
    test_file = './datasets/cnews/cnews.test.txt'

    logger.info('load and preprocess data...')
    
    # build dictionary
    num_words = 5000 # the size of dictionary
    dct = build_dict([train_file, valid_file], num_words=num_words)

    # build dataset and dataloader
    train_ds = NewsDataSet(train_file, dct)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    valid_ds = NewsDataSet(valid_file, dct)
    valid_dl = DataLoader(valid_ds, batch_size=64)

    test_ds = NewsDataSet(test_file, dct)
    test_dl = DataLoader(test_ds, batch_size=64)

    # build model
    model = TextCNN(class_num=len(CATEGIRY_LIST),
                embed_num=len(dct),
                embed_dim=64,
                kernel_num=128,
                kernel_size_list=[3,4,5],
                dropout=0.5)

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train
    logger.info('training...')
    history = train(model, optimizer, train_dl, valid_dl, epochs=10)

    # evaluate
    loss, acc = evaluate(model, valid_dl)

    # predict
    logger.info('predicting...')
    y_pred = predict(model, test_dl)

    y_true = test_ds.labels
    test_acc = (y_true == y_pred).sum() / y_pred.shape[0]
    logger.info('test - acc: {}'.format(test_acc))
