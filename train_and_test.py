import argparse
import ast
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from model import MODEL
from utils.tools import words_index
from utils.dataloader import *


def train_loop(model, optimizer, loss_fn, model_name='bilstm', question_type=1, pointwise=True, epochs=100,
               batch_size=64, emb_index=None, model_path='', cc_num=10000, use_test_data=False, eva_span=3):
    train_time = datetime.now()
    validation_data = DataLoader(
        MyDataset(data_path % {'filename': 'validation'}, model_name=model_name,
                  pointwise=pointwise, question_type=question_type,
                  emb_index=emb_index, test=True, cc_num=cc_num),
        batch_size=1, num_workers=1, collate_fn=evaluate_collate)  # , pin_memory=True)

    test_data = DataLoader(
        MyDataset(data_path % {'filename': 'test'}, model_name=model_name, pointwise=pointwise,
                  question_type=question_type, emb_index=emb_index, test=True, cc_num=cc_num),
        batch_size=1, num_workers=1, collate_fn=evaluate_collate)  # , pin_memory=True)

    train_data = DataLoader(
        MyDataset(data_path % {'filename': 'train'}, model_name=model_name, pointwise=pointwise,
                  question_type=question_type, emb_index=emb_index, test=False, cc_num=cc_num),
        batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_collate)  # , pin_memory=True)
    best_res = 0
    all_loss = []
    for epoch in range(epochs):
        print('%d epoch train start' % (int(epoch) + 1))
        epoch_time = datetime.now()
        epoch_loss = []
        #  如果没有
        clip = 0.5

        for i, item in enumerate(train_data):
            # print(item)
            # print(len(item['ques']))
            # pass
            epoch_loss.append(model.train(item, optimizer, loss_fn, clip).item())

        epoch_loss = np.mean(epoch_loss)
        all_loss.append(epoch_loss)
        print('%d epoch train loss is ' % (int(epoch) + 1), epoch_loss,
              ', used time ', datetime.now() - epoch_time)
        if epoch % eva_span == 0:
            print('validate start ...') if not use_test_data else print('test start ...')
            eva_res = evaluate(model, validation_data) if not use_test_data else evaluate(model, test_data)
            if eva_res > best_res:
                print('saving model ...')
                best_res = eva_res
                model.save(model_path, best_res)
        print('\n')
    print('best cca is ', best_res)
    print('all loss are', all_loss, '\n train used time is',
          datetime.now() - train_time, '\n\n')


# todo 完善测试功能
# CCA, MMR , precision, recall and F1
def evaluate(model, data_loader):
    print('evaluating ...')
    start_time = datetime.now()
    predict_pred = []
    true_pred = []
    predict_index = []
    origin_data = []
    for i, item in enumerate(data_loader):
        # print(item)
        # p_pred, t_pred, p_index, origin_data_ = model.predict(item)
        p_pred, t_pred, p_index = model.predict(item)
        predict_pred.append(p_pred)
        true_pred.append(t_pred)
        predict_index.append(p_index)
        # origin_data.append(origin_data_)
    # print('predict_pred:', predict_pred)
    # print('true_pred:', true_pred)
    cca = 0
    error = 0
    # out = []
    for i in range(len(predict_pred)):
        # data = {
        #     'id': origin_data[i]['id'],
        #     'question': origin_data[i]['question'],
        #     'predict_pred': predict_pred[i],
        #     'true_pred': true_pred[i]
        # }
        if predict_pred[i] == true_pred[i]:
            cca += 1
            if predict_index[i] != 0:
                error += 1
        #     data['isTrue'] = True
        # else:
        #     data['isTrue'] = False
        # out.append(data)
    cca = cca / len(predict_pred)
    print('error(Fake true predict) number is ', error, 'all number is ', len(predict_pred))
    print('CCA is %.3f' % cca)
    print('evaluate time is ', datetime.now() - start_time)
    # json.dump(out, open('result.json', 'w+'), indent=4)
    # print('json dump finished!')
    return cca


if __name__ == '__main__':
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='model, uni_enc, pointwise, question_type, gpu\n'
                                                 'epochs, batch_size, dropout is_load, is_train')
    parser.add_argument('--model', '-m', help='bilstm, dam, smm 默认：bilstm', default='bilstm', type=str)
    parser.add_argument('--uni_enc', '-uni', help='bool', default=True, type=ast.literal_eval)
    parser.add_argument('--pointwise', '-p', help='bool', default=True, type=ast.literal_eval)
    parser.add_argument('--question_type', '-type', help='int: 0 or 1', default=1, type=int)
    parser.add_argument('--gpu', '-gpu', help='str :0, 1, 选择GPU， 默认0', default='0', type=str)
    parser.add_argument('--epochs', '-epochs', help='int: 100', default=100, type=int)
    parser.add_argument('--batch_size', '-batch', help='int: 4', default=4, type=int)
    parser.add_argument('--dropout', '-drop', help='float: 0', default=0, type=float)
    parser.add_argument('--is_load', '-load', help='bool, 默认False', default=False, type=ast.literal_eval)
    parser.add_argument('--is_train', '-train', help='bool, 默认True', default=True, type=ast.literal_eval)
    parser.add_argument('--cc_num', '-cc', help='int, 默认10000', default=10000, type=int)
    parser.add_argument('--eva_span', '-eva', help='int, 默认3', default=3, type=int)
    parser.add_argument('--T', '-T', help='False, 默认False, validation to test', default=False, type=ast.literal_eval)
    parser.add_argument('--lr', '-lr', help='学习率, 默认1e-3', default=1e-3, type=float)
    args = parser.parse_args()

    is_load = args.is_load
    is_train = args.is_train
    epochs = args.epochs
    model_name = args.model
    uni_enc = args.uni_enc
    pointwise = args.pointwise
    dropout = args.dropout
    question_type = args.question_type
    batch_size = args.batch_size
    gpu = args.gpu
    cc_num = args.cc_num
    eva_span = args.eva_span
    lr = args.lr
    is_test_data = args.T

    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')  # 检测GPU
    print('device name ', torch.cuda.get_device_name() if torch.cuda.is_available() else device)

    emb_index, emb_values = words_index()

    if model_name == 'dam':  # dam只有一个encoder
        uni_enc = True
    if model_name == 'smm':
        uni_enc = False

    model_path = 'model/' + model_name
    model = MODEL(model=model_name, uni_enc=uni_enc, pointwise=pointwise, device=device, emb_values=emb_values,
                  dropout=dropout)

    if not pointwise:
        # label 1/-1  pairwise
        model_path += '_pair'
        loss_fn = nn.MarginRankingLoss(margin=1, reduction='mean')
    else:
        # label 1/0   pointwise
        model_path += '_point'
        # loss_fn = nn.BCELoss(reduction='mean')
        loss_fn = nn.BCEWithLogitsLoss()

    model_path += '_type' + str(question_type)

    if uni_enc:
        model_path += '_uni'
        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.encoder.parameters())), lr=lr)
    else:
        model_path += '_dou'
        optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.encoder_q.parameters())) +
                               list(filter(lambda p: p.requires_grad, model.encoder_p.parameters())), lr=lr)

    model_path += '_cc-' + str(cc_num)

    print(model_path)
    if is_load or not is_train:
        print('load model ...')
        model.load(model_path)

    if is_train:
        train_loop(model, optimizer, loss_fn, model_name=model_name, question_type=question_type, pointwise=pointwise,
                   batch_size=batch_size, epochs=epochs, emb_index=emb_index, model_path=model_path, cc_num=cc_num,
                   use_test_data=is_test_data, eva_span=eva_span)
    else:
        test_data = DataLoader(
            MyDataset(data_path % {'filename': 'test'}, model_name=model_name, pointwise=pointwise,
                      question_type=question_type, emb_index=emb_index, test=True, cc_num=cc_num),
            batch_size=1, num_workers=1, collate_fn=evaluate_collate)  # , pin_memory=True)
        print('test start ...')
        evaluate(model, test_data)

    print('all used time is', datetime.now() - start_time)
