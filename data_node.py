import string
import re
import numpy as np
import random

from utils.tools import DBpedia_query

# sparql for one hop one entity
u_p_e = 'SELECT DISTINCT ?predicate WHERE { ?subjective_resource ?predicate %(given_resource)s }'
e_p_u = 'SELECT DISTINCT ?predicate WHERE { %(given_resource)s ?predicate ?objective_resource }'

# sparql for two hops one entity
u1_p1_e_u2_p2_u1 = 'SELECT DISTINCT ?p1 ?p2 WHERE { ?uri1 ?p1 %(given_entity)s . ?uri2 ?p2 ?uri1 }'
u1_p1_e_u1_p2_u2 = 'SELECT DISTINCT ?p1 ?p2 WHERE { ?uri1 ?p1 %(given_entity)s . ?uri1 ?p2 ?uri2 }'
e_p1_u1_u2_p2_u1 = 'SELECT DISTINCT ?p1 ?p2 WHERE { %(given_entity)s ?p1 ?uri1 . ?uri2 ?p2 ?uri1 }'
e_p1_u1_u1_p2_u2 = 'SELECT DISTINCT ?p1 ?p2 WHERE { %(given_entity)s ?p1 ?uri1 . ?uri1 ?p2 ?uri2 }'

# sparql for two entities
u_p1_e1_e2_p2_u = 'SELECT DISTINCT ?p1 ?p2 WHERE { ?uri ?p1 %(given_entity1)s . %(given_entity2)s ?p2 ?uri }'
u_p1_e1_u_p2_e2 = 'SELECT DISTINCT ?p1 ?p2 WHERE { ?uri ?p1 %(given_entity1)s . ?uri ?p2 %(given_entity2)s }'
e1_p1_u_e2_p2_u = 'SELECT DISTINCT ?p1 ?p2 WHERE { %(given_entity1)s ?p1 ?uri . %(given_entity2)s ?p2 ?uri }'
e1_p_e2 = 'SELECT DISTINCT ?predicate WHERE { %(given_entity1)s ?predicate %(given_entity2)s }'


class DataNode(object):
    def __init__(self):
        # 谓语黑名单中的谓语代表的是什么 为什么要黑名单
        self.predicate_base = open('resources/predicate.blacklist').readlines()
        self.predicate_base = [r[:-1] for r in self.predicate_base]

        self.stopwords = np.loadtxt('resources/stopwords.txt', dtype=str)

        # self.glove = json.load(open('resources/glove_need.json', 'r'))

    def is_1entity(self, data):
        return True if data['entity2_mention'] == '' and data['entity2_uri'] == '' else False

    def is_get_true_predicate(self, res):
        # 2019.10.2
        data = res['origin_data']
        true_predicate = []
        if data['predicate1_uri']:  # 解决 1hop
            if data['entity1_uri'] and data['entity2_uri'] and not data['predicate2_uri']:  # 2 entity 1 hop (ASK)
                true_predicate.append('+')
            elif data['sparql_query'].find(data['entity1_uri'],
                                        data['sparql_query'].find(data['predicate1_uri']) + 1) != -1:
                true_predicate.append('-')
            else:
                true_predicate.append('+')
            true_predicate.append(data['predicate1_uri'])
        if data['predicate2_uri']:  # 解决2hops
            start = data['sparql_query'].find(data['predicate2_uri']) + 1
            if data['predicate1_uri'] == data['predicate2_uri']:  # 两个谓语相同
                start = data['sparql_query'].find(data['predicate2_uri'], start) + 1

            if data['entity2_uri']:  # 2 entities 2 hops
                if data['sparql_query'].find(data['entity2_uri'], start) != -1:
                    true_predicate.append('+')
                else:
                    true_predicate.append('-')
            else:  # 1 entities 2 hops
                if data['type_uri'] != "":  # 需要 type
                    x = data['sparql_query'].find('?x', start)
                    uri = data['sparql_query'].find('?uri', start)
                    if x == -1:  # type uri
                        true_predicate.append('+')
                    elif uri == -1:  # type x
                        true_predicate.append('-')
                    elif x < uri:  # type uri
                        true_predicate.append('-')
                    else:  # type x
                        true_predicate.append('+')
                else:  # 不需要type
                    if data['sparql_query'].find('?x', start) != -1:
                        true_predicate.append('-')
                    else:
                        true_predicate.append('+')
            true_predicate.append(data['predicate2_uri'])
        if true_predicate in res['predicates']:
            return True, true_predicate
        print('id:', data['id'])
        print('question:', data['question'])
        print('true_predicate', true_predicate)
        return False, true_predicate

    # 2019.9.8 look for true predicate chain and reduce the candidate chains
    def get_true_predicate(self, res):
        is_get, true_predicate = self.is_get_true_predicate(res)
        # print(true_predicate)
        if true_predicate in res['predicates']:
            res['predicates'].remove(true_predicate)

        # if len(res['predicates']) > K:
        #     res['predicates'] = random.sample(res['predicates'], K)
        return res['predicates'], true_predicate, is_get

    def update_question(self, question, entity1='', entity2=''):
        question = question.lower()

        updated_question1 = question
        if entity1 is not '':
            updated_question1 = updated_question1.replace(entity1.lower(), '<e>')
        if entity2 is not '':
            updated_question1 = updated_question1.replace(entity2.lower(), '<e>')

        # remove punctuation 去除标点符号
        # updated_question1 = re.sub('[%s]' % re.escape(string.punctuation), ' ', updated_question1)
        updated_question1 = updated_question1.replace('?', '')
        updated_question1 = updated_question1.replace(',', '')

        updated_question2 = [word for word in updated_question1.split(' ') if word not in self.stopwords]
        updated_question2 = ' '.join(updated_question2)

        return updated_question1, updated_question2

    def filter_one_predicates(self, predicates):
        return [x for x in predicates if x not in self.predicate_base]
        # return [x for x in predicates if x not in self.predicate_base and x.startswith('http://dbpedia.org')]

    def filter_two_predicates(self, predicates):
        return [
            [x[0], x[1]] for x in predicates if x[0] not in self.predicate_base and x[1] not in self.predicate_base
            # and
            # x[0].startswith('http://dbpedia.org') and
            # x[1].startswith('http://dbpedia.org')
        ]

    def query_and_get_one_predicates(self, query):
        response = DBpedia_query(query)
        predicates = [
            x['predicate']['value'] for x in response['results']['bindings']
        ]
        return predicates

    def query_and_get_two_predicates(self, query):
        response = DBpedia_query(query)
        predicates = [
            [x['p1']['value'], x['p2']['value']] for x in response['results']['bindings']
        ]
        return predicates

    def get_one_entity_one_hop_predicates(self, entity1_uri):
        """
        one entity one hop
        :param entity1_uri:
        :return:
        """
        in_predicates = []
        out_predicates = []
        entity1_uri = '<' + entity1_uri + '>'

        query = e_p_u % {'given_resource': entity1_uri}
        predicates = self.query_and_get_one_predicates(query)
        predicates = self.filter_one_predicates(predicates)
        for predicate in predicates:
            out_predicates.append(['+', predicate])
        # print('one entity one hop predicates num of predicates is %d' % len(predicates))
        no_direction_predicates = predicates
        # print('one entity one hop predicates num of no_direction_predicates is %d' % len(no_direction_predicates))

        query = u_p_e % {'given_resource': entity1_uri}
        predicates = self.query_and_get_one_predicates(query)
        predicates = self.filter_one_predicates(predicates)
        for predicate in predicates:
            in_predicates.append(['-', predicate])
        # print('one entity one hop predicates num of predicates is %d' % len(predicates))
        no_direction_predicates += predicates
        # print('one entity one hop predicates num of no_direction_predicates is %d' % len(no_direction_predicates))

        one_hop_predicates = out_predicates
        one_hop_predicates += in_predicates
        return one_hop_predicates, no_direction_predicates

    def get_one_entity_two_hops_predicates(self, entity1_uri, one_hop_predicates):
        """
        one entity two hops
        :param entity1_uri:
        :param one_hop_predicates:
        :return:
        """
        predicates = []
        entity1_uri = '<' + entity1_uri + '>'

        in_in = u1_p1_e_u2_p2_u1 % {'given_entity': entity1_uri}
        in_out = u1_p1_e_u1_p2_u2 % {'given_entity': entity1_uri}
        out_in = e_p1_u1_u2_p2_u1 % {'given_entity': entity1_uri}
        out_out = e_p1_u1_u1_p2_u2 % {'given_entity': entity1_uri}

        in_in = self.query_and_get_two_predicates(in_in)
        in_in = [x for x in in_in if x[0] in one_hop_predicates]
        in_in = self.filter_two_predicates(in_in)
        for predicate in in_in:
            predicates.append(['-', predicate[0], '-', predicate[1]])

        in_out = self.query_and_get_two_predicates(in_out)
        in_out = [x for x in in_out if x[0] in one_hop_predicates]
        in_out = self.filter_two_predicates(in_out)
        for predicate in in_out:
            predicates.append(['-', predicate[0], '+', predicate[1]])

        out_in = self.query_and_get_two_predicates(out_in)
        out_in = [x for x in out_in if x[0] in one_hop_predicates]
        out_in = self.filter_two_predicates(out_in)
        for predicate in out_in:
            predicates.append(['+', predicate[0], '-', predicate[1]])

        out_out = self.query_and_get_two_predicates(out_out)
        out_out = [x for x in out_out if x[0] in one_hop_predicates]
        out_out = self.filter_two_predicates(out_out)
        for predicate in out_out:
            predicates.append(['+', predicate[0], '+', predicate[1]])

        return predicates

    def one_entity_core_chains(self, entity1_uri):
        """
        There is only one topic entity in the question.
        Aiming to find all predicates in the two hops range
        :param entity1_uri:
        :return:
        """
        # one hop
        one_hop_predicates, no_direction_predicates = self.get_one_entity_one_hop_predicates(entity1_uri)

        # two hops
        two_hops_predicates = self.get_one_entity_two_hops_predicates(entity1_uri, no_direction_predicates)
        num = len(one_hop_predicates) + len(two_hops_predicates)

        # print('after filter, the num of predicates is %d %d respectively'
        #       % (len(one_hop_predicates), len(two_hops_predicates)))
        # filter by similarity between question and predicates chains

        # one_hop_predicates, two_hops_predicates = self.one_entity_filter_by_similarity(
        #     one_hop_predicates, two_hops_predicates, question)

        # return one_hop_predicates, two_hops_predicates, num
        return one_hop_predicates + two_hops_predicates, num

    def two_entities_core_chains(self, entity1_uri, entity2_uri):
        """
        There are two topic entities in question
        The paper is assuming that the answer is between the two topic entities
        :param entity1_uri: first entity
        :param entity2_uri: second entity
        :return:
        """
        entity1_uri = '<' + entity1_uri + '>'
        entity2_uri = '<' + entity2_uri + '>'

        in_in = u_p1_e1_e2_p2_u % {'given_entity1': entity1_uri, 'given_entity2': entity2_uri}
        in_out = u_p1_e1_u_p2_e2 % {'given_entity1': entity1_uri, 'given_entity2': entity2_uri}
        out_in = e1_p1_u_e2_p2_u % {'given_entity1': entity1_uri, 'given_entity2': entity2_uri}
        out_out = e1_p_e2 % {'given_entity1': entity1_uri, 'given_entity2': entity2_uri}

        predicates = []
        in_in = self.query_and_get_two_predicates(in_in)
        in_in = self.filter_two_predicates(in_in)
        for predicate in in_in:
            predicates.append(['-', predicate[0], '-', predicate[1]])

        in_out = self.query_and_get_two_predicates(in_out)
        in_out = self.filter_two_predicates(in_out)
        for predicate in in_out:
            predicates.append(['-', predicate[0], '+', predicate[1]])

        out_in = self.query_and_get_two_predicates(out_in)
        out_in = self.filter_two_predicates(out_in)
        for predicate in out_in:
            predicates.append(['+', predicate[0], '-', predicate[1]])

        out_out = self.query_and_get_one_predicates(out_out)
        out_out = self.filter_one_predicates(out_out)
        for predicate in out_out:
            predicates.append(['+', predicate])

        num = len(predicates)

        # predicates = self.two_entities_filter_by_similarity(predicates, question)

        return predicates, num

    def node_process(self, data):
        """
        process each data node
        :param data: data node
        :return:
        """

        # data node's structure
        # answer type： data['classx']
        res = {
            'origin_data': data,
            'updated_question': [],  # 0. pattern entity -> <e> 1. pattern after stopwords
            # 'answer': '',
            'true_predicate': [],
            'predicates': [],
            'type_for': ''
        }

        res['updated_question'] = self.update_question(
            data['question'], data['entity1_mention'], data['entity2_mention'])

        if self.is_1entity(data):
            # res['1entity']['1hop'], res['1entity']['2hops'], num = self.one_entity_core_chains(
            #     data['entity1_uri'], res['updated_question2'])
            res['predicates'], num = self.one_entity_core_chains(data['entity1_uri'])
        else:
            # res['2entities'], num = self.two_entities_core_chains(
            #     data['entity1_uri'], data['entity2_uri'], res['updated_question2'])
            res['predicates'], num = self.two_entities_core_chains(data['entity1_uri'], data['entity2_uri'])

        # res['answer'] = self.get_true_answer(data)
        #  1.随机挑选99个predicates 2.获得正确的predicates 3.是否有正确的predicates
        res['predicates'], res['true_predicate'], is_get = self.get_true_predicate(res)

        # res['is_get'] = is_get
        return res, num, is_get
