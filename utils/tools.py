import json
import re

from SPARQLWrapper import SPARQLWrapper, JSON

# DBpedia_endpoint = "https://dbpedia.org/sparql"
DBpedia_endpoint = "http://10.201.180.179:8890/sparql"


def DBpedia_query(_query):
    """
    :param _query: sparql query statement
    :return:
    """
    sparql = SPARQLWrapper(DBpedia_endpoint)
    sparql.setQuery(_query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    return sparql.query().convert()


def uri2words(uri):
    words = uri.split('/')[-1]
    words = words[0].upper() + words[1:]
    # 按大小写分词，并转化为小写
    return [word.lower() for word in re.findall('[A-Z][^A-Z]*', words)]
    # return [words.lower()]


def words2index(words, emb_index):
    index = [emb_index[word] for word in words if word != ' ' and word != '']
    return index


def words_index():
    """
    建立glove keys和values的索引
    :return:
    """
    glove = json.load(open('resources/glove_need.json', 'r'))
    # 为后面pad做准备
    values = [[0] * 300]
    values += list(glove.values())
    keys = list(glove.keys())
    for i, key in enumerate(keys):
        glove[key] = i + 1
    print('size of glove index and values', len(glove), len(values))
    return glove, values


def get_true_answer(data):
    response = DBpedia_query(data['sparql_query'])
    if data['classx'] == '4':  # ask
        res = response['boolean']
    elif data['classx'] in ['0', '1']:  # select
        uri_res = [
            x['uri']['value'] for x in response['results']['bindings']
        ]
        res = []
        for uri in uri_res:
            uri = str(uri).split('/')
            res.append(uri[-1].lower())
        # print(res)
    elif data['classx'] in ['2', '3']:  # count
        res = int(response['results']['bindings'][0]['callret-0']['value'])
    else:  # other
        res = None
    return res


def ans_rate():
    dataset = json.load(open('../data/LC_QuAD.json', 'r', encoding='UTF-8'))
    # doubt = []
    num = 0
    no_ans = 0
    for node in dataset:
        response = DBpedia_query(node['sparql_query'])
        if 'ASK' in node['sparql_query']:  # asktry
            ans = response['boolean']
            if type(ans) == bool:
                num += 1
                # if not node['is_get']:
                #     doubt.append(node['origin_data'])
            if not ans:
                no_ans += 1
        elif 'COUNT' in node['sparql_query']:  # count
            ans = int(response['results']['bindings'][0]['callret-0']['value'])
            if ans > 0:
                num += 1
                # if not node['is_get']:
                #     doubt.append(node['origin_data'])
        elif 'SELECT' in node['sparql_query']:  # select
            uri_res = [
                x['uri']['value'] for x in response['results']['bindings']
            ]
            ans = []
            for uri in uri_res:
                uri = str(uri).split('/')
                ans.append(uri[-1].lower())
            if ans:
                num += 1
                # if not node['is_get']:
                #     doubt.append(node['origin_data'])
            # else:
            #     print(ans)
    # json.dump(doubt, open('doubt.json', 'w+'), indent=4)
    # print('size of which have ans but no pred chains ', len(doubt))
    # 4998/5000 0
    print('ans : %d / %d, ' % (num, len(dataset)), ' ASK No num :', no_ans)


def compute_NumOfPredicates():
    train_data = json.load(open('data/train.json', 'r'))
    validation_data = json.load(open('data/validation.json', 'r'))
    test_data = json.load(open('data/test.json', 'r'))
    dataset = (train_data, validation_data, test_data)
    p = [0, 0]
    for data in dataset:
        p_ = [0, 0]
        for node in data:
            p1 = []
            p2 = []
            if len(node['true_predicate']) == 4:
                p1.append(tuple(node['true_predicate'][:2]))
                p2.append(tuple(node['true_predicate'][2:]))
            elif len(node['true_predicate']) == 2:
                p1.append(tuple(node['true_predicate'][:2]))
            for predicate in node['predicates']:
                if len(predicate) == 2:
                    p1.append(tuple(predicate[:2]))
                elif len(predicate) == 4:
                    p1.append(tuple(predicate[:2]))
                    p2.append(tuple(predicate[2:]))
            p1 = list(set(p1))
            p2 = list(set(p2))
            p_[0] += len(p1)
            p_[1] += len(p2)
            p[0] += len(p1)
            p[1] += len(p2)
        print('p1 : %.3f,  p2 : %.3f' % (p_[0]/len(data), p_[1]/len(data)))
    print('p1 : %.3f,  p2 : %.3f' % (p[0]/5000, p[1]/5000))


def numPredicatesWithQuestion():
    dataset = json.load(open('data/test.json', 'r')) + json.load(open('data/train.json', 'r')) + json.load(
        open('data/validation.json', 'r'))
    num = [0] * 10
    for node in dataset:
        if len(node['predicates']) > 10000:
            num[0] += 1
        if len(node['predicates']) > 5000:
            num[1] += 1
        if len(node['predicates']) > 1000:
            num[2] += 1
        if len(node['predicates']) > 500:
            num[3] += 1
        if len(node['predicates']) > 200:
            num[4] += 1
        if len(node['predicates']) > 100:
            num[5] += 1
    print(num)

    # 143, 132, 1457, 719, 565, 171
    # 143, 275, 1732, 2451, 3016, 3187


if __name__ == '__main__':
    from datetime import datetime

    start_time = datetime.now()
    # ans_rate()
    compute_NumOfPredicates()
    print('used time ', datetime.now() - start_time)
