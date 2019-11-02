import sys

import torch
import torch.nn as nn

from networks import BiLSTM, DAM, SMM_q, SMM_p


class MODEL(object):
    def __init__(self, model=None, uni_enc=True, pointwise=True, emb_dim=300, hidden_dim=300, dropout=0.0, device='cpu',
                 emb_values=None):
        self.device = device
        self.uni_enc = uni_enc
        self.pointwise = pointwise
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.model = model

        if model == 'bilstm':
            if self.uni_enc:
                self.encoder = BiLSTM(emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                              device=self.device, emb_values=emb_values).to(device=self.device)
            else:
                self.encoder_q = BiLSTM(emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                                device=self.device, emb_values=emb_values).to(device=self.device)
                self.encoder_p = BiLSTM(emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                            device=self.device, emb_values=emb_values).to(device=self.device)
        elif model == 'dam':
            self.encoder = DAM(emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                              device=self.device, emb_values=emb_values).to(device=self.device)
        elif model == 'smm':
            self.encoder_q = SMM_q(emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                              device=self.device, emb_values=emb_values).to(device=self.device)
            self.encoder_p = SMM_p(emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                                 device=self.device, emb_values=emb_values).to(device=self.device)
        else:
            sys.exit('model name is wrong!')

    def pointwise_train(self, data, optimizer, loss_fn, clip):
        optimizer.zero_grad()
        if self.model == 'bilstm':
            if self.uni_enc:
                ques_enc, hq, _, _ = self.encoder(data['ques'])
                pred_enc, hp, _, _ = self.encoder(data['pred'])
            else:
                ques_enc, hq, _, _ = self.encoder_q(data['ques'])
                pred_enc, hp, _, _ = self.encoder_p(data['pred'])
            scores = torch.sum(ques_enc * pred_enc, -1)
        elif self.model == 'dam':
            scores = self.encoder(data['ques'], data['pred'])
        elif self.model == 'smm':
            ques_enc = self.encoder_q(data['ques'])
            pred_enc = self.encoder_p(data['p1'], data['p2'])
            scores = torch.sum(ques_enc * pred_enc, -1)

        label = torch.tensor(data['label'], dtype=torch.float).to(device=self.device)
        loss = loss_fn(scores, label)
        loss.backward()

        if self.uni_enc:
            nn.utils.clip_grad_norm_(list(self.encoder.parameters()), clip)
        else:
            nn.utils.clip_grad_norm_(list(self.encoder_q.parameters()) +
                                     list(self.encoder_p.parameters()), clip)
        optimizer.step()
        return loss

    def pairwise_train(self, data, optimizer, loss_fn, clip):
        optimizer.zero_grad()
        if self.model == 'bilstm':
            if self.uni_enc:
                ques_enc, hq, _, _ = self.encoder(data['ques'])
                pos_enc, hp, _, _ = self.encoder(data['pos'])
                neg_enc, hn, _, _ = self.encoder(data['neg'])
            else:
                ques_enc, hq, _, _ = self.encoder_q(data['ques'])
                pos_enc, hp, _, _ = self.encoder_p(data['pos'])
                neg_enc, hn, _, _ = self.encoder_p(data['neg'])

            pos_scores = torch.sum(ques_enc * pos_enc, -1)
            neg_scores = torch.sum(ques_enc * neg_enc, -1)
        elif self.model == 'dam':
            pos_scores = self.encoder(data['ques'], data['pos'])
            neg_scores = self.encoder(data['ques'], data['neg'])
        elif self.model == 'smm':
            ques_enc = self.encoder_q(data['ques'])
            pos_enc = self.encoder_p(data['pos_p1'], data['pos_p2'])
            neg_enc = self.encoder_p(data['neg_p1'], data['neg_p2'])

            pos_scores = torch.sum(ques_enc * pos_enc, -1)
            neg_scores = torch.sum(ques_enc * neg_enc, -1)

        label = torch.tensor(data['label'], dtype=torch.float).to(device=self.device)
        loss = loss_fn(pos_scores, neg_scores, label)
        loss.backward()

        if self.uni_enc:
            nn.utils.clip_grad_norm_(list(self.encoder.parameters()), clip)
        else:
            nn.utils.clip_grad_norm_(list(self.encoder_q.parameters()) +
                                     list(self.encoder_p.parameters()), clip)
        optimizer.step()
        return loss

    def train(self, data, optimizer, loss_fn, clip=1):
        if self.pointwise:
            return self.pointwise_train(data, optimizer, loss_fn, clip)
        else:
            return self.pairwise_train(data, optimizer, loss_fn, clip)

    def predict(self, data):
        """
        same code work for pointwise and pairwise
        :return:
        """
        with torch.no_grad():
            if self.model == 'bilstm':
                if self.uni_enc:
                    self.encoder.eval()
                    ques_enc, hq, _, _ = self.encoder(data['ques'])
                    pred_enc, hp, _, _ = self.encoder(data['pred'])
                    self.encoder.train()
                else:
                    self.encoder_q.eval()
                    self.encoder_p.eval()
                    ques_enc, hq, _, _ = self.encoder_q(data['ques'])
                    pred_enc, hp, _, _ = self.encoder_p(data['pred'])
                    self.encoder_q.train()
                    self.encoder_p.train()
                scores = torch.sum(ques_enc * pred_enc, -1)
            elif self.model == 'dam':
                self.encoder.eval()
                scores = self.encoder(data['ques']*len(data['pred']), data['pred'])
                self.encoder.train()
            elif self.model == 'smm':
                self.encoder_q.eval()
                self.encoder_p.eval()
                ques_enc = self.encoder_q(data['ques'])
                pred_enc = self.encoder_p(data['p1'], data['p2'])
                scores = torch.sum(ques_enc * pred_enc, -1)
                self.encoder_q.train()
                self.encoder_p.train()

            predict_index = torch.argmax(scores).item()
            # print(predict_index)
            return data['origin_pred'][predict_index], data['true_pred'], predict_index  #, data['origin_data']

    def save(self, model_path, result):
        if self.uni_enc:
            torch.save({
                'encoder': self.encoder.state_dict(),
                'result': result
            }, model_path + '.pt')
        else:
            torch.save({
                'encoder_q': self.encoder_q.state_dict(),
                'encoder_p': self.encoder_p.state_dict(),
                'result': result
            }, model_path + '.pt')

    def load(self, model_path):
        state_dict = torch.load(model_path + '.pt')
        print('loading model:', model_path + '.pt')
        print('result: CCA = ', state_dict['result'])
        if self.uni_enc:
            self.encoder.load_state_dict(state_dict['encoder'])
        else:
            self.encoder_q.load_state_dict(state_dict['encoder_q'])
            self.encoder_p.load_state_dict(state_dict['encoder_p'])
