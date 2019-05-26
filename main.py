import os
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

from cnn.model import TextCNN
from rcnn.model import TextRCNN
from rnn_attention.model import Bi_RNN_ATTN

from data import build_dict, NewsDataSet, CATEGIRY_LIST
import trainer


if __name__ == "__main__":
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    logger.info('using device: {}'.format(device))

    
    train_file = os.path.abspath('../../datasets/cnews/cnews.train.txt')
    valid_file = os.path.abspath('../../datasets/cnews/cnews.val.txt')
    test_file = os.path.abspath('../../datasets/cnews/cnews.test.txt')

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
                    embed_size=len(dct))

    # model = TextRCNN(class_num=len(CATEGIRY_LIST),
    #                  embed_size=len(dct),
    #                  device=device)

    # model = Bi_RNN_ATTN(class_num=len(CATEGIRY_LIST),
    #                     embed_size=len(dct),
    #                     embed_dim=64,
    #                     device=device)


    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train
    logger.info('training...')
    history = trainer.train(model, optimizer, train_dl, valid_dl, device=device, epochs=5)

    # evaluate
    loss, acc = trainer.evaluate(model, valid_dl, device=device)

    # predict
    logger.info('predicting...')
    y_pred = trainer.predict(model, test_dl, device=device)

    y_true = test_ds.labels
    test_acc = (y_true == y_pred).sum() / y_pred.shape[0]
    logger.info('test - acc: {}'.format(test_acc))
