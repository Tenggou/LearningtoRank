import copy
import json
import random

from torch.utils.data import Dataset

from utils.tools import words2index, uri2words

data_path = 'data/%(filename)s.json'
K_train = 100  # 保存K个predicates chains（negative）


class MyDataset(Dataset):
    def __init__(self, filepath, model_name='bilstm', pointwise=False, question_type=1, device='cpu', emb_index=None,
                 test=False, cc_num=10000):
        self.data = json.load(open(filepath, 'r'))
        self.device = device
        self.question_type = question_type
        self.emb_index = emb_index
        self.pointwise = pointwise
        self.test = test
        self.model_name = model_name

        sum = 0
        # todo 没有正理和负例的时候？？
        for i in range(len(self.data)):
            if not self.data[i]['true_predicate']:
                self.data[i]['true_predicate'] = ['<e>', '<e>']
                # print(self.data[i]['true_predicate'])
            if not self.data[i]['predicates']:
                self.data[i]['predicates'] = [['+', '-']]
                # print(self.data[i]['predicates'])
            # 训练集选负例
            if len(self.data[i]['predicates']) > cc_num:
                self.data[i]['predicates'] = random.sample(self.data[i]['predicates'], cc_num)
            sum += len(self.data[i]['predicates']) + len(self.data[i]['true_predicate'])
        print(filepath + '\'s number of core chains is', sum/len(self.data))

    def __getitem__(self, item):
        if len(self.data[item]['predicates']) > K_train and not self.test:
            origin_predicates = random.sample(self.data[item]['predicates'], K_train)
        else:
            origin_predicates = copy.deepcopy(self.data[item]['predicates'])
        # origin_predicates = copy.deepcopy(self.data[item]['predicates'])

        ques = self.data[item]['updated_question'][self.question_type].split(' ')
        ques = words2index(ques, self.emb_index)
        # 问题为空的情况
        if not ques:
            ques = words2index(['<e>'], self.emb_index)
            print(self.data[item])
        # print(ques)

        true_predicate = []
        true_p1 = []
        true_p2 = []
        if len(self.data[item]['true_predicate']) == 2:
            words = [self.data[item]['true_predicate'][0]]
            words += uri2words(self.data[item]['true_predicate'][1])
            true_predicate = words2index(words, self.emb_index)
            true_p1 = words2index(words, self.emb_index)
        elif len(self.data[item]['true_predicate']) == 4:
            words = [self.data[item]['true_predicate'][0]]
            words += uri2words(self.data[item]['true_predicate'][1])
            true_p1 = words2index(words, self.emb_index)

            words_ = [self.data[item]['true_predicate'][2]]
            words_ += uri2words(self.data[item]['true_predicate'][3])
            true_p2 = words2index(words_, self.emb_index)

            true_predicate = words2index(words+words_, self.emb_index)
        predicates = []
        p1 = []
        p2 = []
        for predicate in origin_predicates:
            p = []
            p1_ = []
            p2_ = []
            if len(predicate) == 2:
                words = [predicate[0]]
                words += uri2words(predicate[1])
                p = words2index(words, self.emb_index)
                p1_ = words2index(words, self.emb_index)
            elif len(predicate) == 4:
                words = [predicate[0]]
                words += uri2words(predicate[1])
                p1_ = words2index(words, self.emb_index)
                words_ = [predicate[2]]
                words_ += uri2words(predicate[3])
                p2_ = words2index(words_, self.emb_index)
                p = words2index(words+words_, self.emb_index)
            predicates.append(p)
            p1.append(p1_)
            p2.append(p2_)

        length = len(predicates)
        if self.test:
            predicates.insert(0, true_predicate)
            p1.insert(0, true_p1)
            p2.insert(0, true_p2)
            origin_predicates.insert(0, self.data[item]['true_predicate'])
            if self.model_name == 'smm':
                return {
                    # 'origin_data': self.data[item]['origin_data'],
                    'ques': [ques],
                    'p1': p1,
                    'p2': p2,
                    'true_pred': self.data[item]['true_predicate'],
                    'origin_pred': origin_predicates
                }
            else:
                return {
                    'ques': [ques],
                    'pred': predicates,
                    'true_pred': self.data[item]['true_predicate'],
                    'origin_pred': origin_predicates
                }
        else:
            if self.pointwise:
                if self.model_name == 'smm':
                    return {
                        'ques': [ques] * (length + len([true_predicate]) * length),
                        'p1': p1 + [true_p1] * length,
                        'p2': p2 + [true_p2] * length,
                        'label': [0] * length + [1] * (len([true_predicate]) * length)
                    }
                return {
                    # 'origin_ques': self.data[item]['updated_question'][self.question_type].split(' '),
                    'ques': [ques] * (length + len([true_predicate]) * length),
                    'pred': predicates + [true_predicate] * length,
                    'label': [0] * length + [1] * (len([true_predicate]) * length)
                }
                # if not data['ques'] or not data['pred'] or not data['label']:
                #     print(data)
                #     print(ques)
                #     print(predicates)
                #     print(true_predicate)
                # return data
            else:
                if self.model_name == 'smm':
                    return {
                        'ques': [ques] * length,
                        'pos_p1': [true_p1] * length,
                        'pos_p2': [true_p2] * length,
                        'neg_p1': p1,
                        'neg_p2': p2,
                        'label': [1] * length
                    }
                return {
                    'ques': [ques] * length,
                    'pos': [true_predicate] * length,
                    'neg': predicates,
                    'label': [1] * length
                }

    def __len__(self):
        return len(self.data)


def train_collate(batch_data):
    if 'p1' in batch_data[0]:
        return {
            'ques': [ques for data in batch_data for ques in data['ques']],
            'p1': [pred for data in batch_data for pred in data['p1']],
            'p2': [pred for data in batch_data for pred in data['p2']],
            'label': [label for data in batch_data for label in data['label']]
        }
    elif 'pos_p1' in batch_data[0]:
        return {
            'ques': [ques for data in batch_data for ques in data['ques']],
            'pos_p1': [pred for data in batch_data for pred in data['pos_p1']],
            'pos_p2': [pred for data in batch_data for pred in data['pos_p2']],
            'neg_p1': [pred for data in batch_data for pred in data['neg_p1']],
            'neg_p2': [pred for data in batch_data for pred in data['neg_p2']],
            'label': [label for data in batch_data for label in data['label']]
        }
    elif 'pred' in batch_data[0]:
        return {
            'ques': [ques for data in batch_data for ques in data['ques']],
            'pred': [pred for data in batch_data for pred in data['pred']],
            'label': [label for data in batch_data for label in data['label']]
        }
    else:
        return {
            'ques': [ques for data in batch_data for ques in data['ques']],
            'pos': [pos for data in batch_data for pos in data['pos']],
            'neg': [neg for data in batch_data for neg in data['neg']],
            'label': [label for data in batch_data for label in data['label']]
        }


def evaluate_collate(batch_data):
    return batch_data[0]