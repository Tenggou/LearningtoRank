import json
import numpy as np

from utils.tools import uri2words

save_glove_path = 'resources/glove_need.json'
out_words_path = 'resources/out_words.txt'


def find_and_save_glove_need(dataset):
    """
    reduce the size of glove, get needed words' embedding (needed glove)
    :param dataset:
    :return:
    """
    words = []
    for data in dataset:
        words += data['updated_question'][0].split(' ')
        words += data['updated_question'][1].split(' ')

        if len(data['true_predicate']) == 2:
            # words.append(uri2word(data['true_predicate'][1]))
            words += uri2words(data['true_predicate'][1])
        elif len(data['true_predicate']) == 4:
            words += uri2words(data['true_predicate'][1])
            words += uri2words(data['true_predicate'][3])
        for predicate in data['predicates']:
            if len(predicate) == 2:
                words += uri2words(predicate[1])
            elif len(predicate) == 4:
                words += uri2words(predicate[1])
                words += uri2words(predicate[3])
    words.append('+')
    words.append('-')
    words.append('<e>')
    words = list(set(words))  # remove the repeated words
    if '' in words:
        words.remove('')
    if ' ' in words:
        words.remove(' ')

    print('all words num is', len(words))
    print('start loading origin glove.txt')
    origin_glove = np.loadtxt('resources/glove.6B.300d.txt', dtype=str, encoding='utf-8', delimiter=' ')
    print('glove.txt loaded!, the shape is ', origin_glove.shape)

    print('start selecting')
    glove_need = {}
    for i in range(origin_glove.shape[0]):
        if origin_glove[i][0] in words:
            glove_need[origin_glove[i][0]] = origin_glove[i][1:].astype(float).tolist()
            words.remove(origin_glove[i][0])

    print('random num is ', len(words))
    out_words_file = open(out_words_path, 'w')
    for word in words:
        glove_need[word] = np.random.randn(300).tolist()
        out_words_file.write(word)
        out_words_file.write('\n')
    print('save words which are out of glove.6B.300d')

    print('size of glove_need', len(glove_need))
    json.dump(glove_need, open(save_glove_path, 'w+'), indent=4)
    print('saved needed glove!')
