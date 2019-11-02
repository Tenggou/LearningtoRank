import json
import multiprocessing
from datetime import datetime

from data_node import DataNode
from utils.glove import find_and_save_glove_need

save_path = 'data/%(filename)s'
dataset_path_origin = 'data/LC_QuAD.json'
dataset_path_ranked = 'data/lcquad_ranked.json'


def data_process(dataset, index, num_of_each_dataset):
    """
    load and  preprocess origin data, get what we will put into ranking model
    :param dataset:
    :param index:
    :param num_of_each_dataset:
    :return:
    """
    start_time = datetime.now()
    res = []

    get_true_predicates = 0
    total_predicates = 0

    data_node = DataNode()
    start_index = int(num_of_each_dataset * index)
    end_index = int(num_of_each_dataset * (index + 1)) if num_of_each_dataset * (index + 1) < len(dataset) else len(
        dataset)
    dataset = dataset[start_index: end_index]

    for node in dataset:
        res_node, num, is_get = data_node.node_process(node)
        res.append(res_node)
        total_predicates += num
        get_true_predicates += int(is_get)

    print('%d process finished, used time %s' % (index + 1, datetime.now() - start_time))
    return res, total_predicates, get_true_predicates


# multi-processes
def run(dataset_path=dataset_path_ranked, num_of_process=10, reduce_glove=True):
    start_time = datetime.now()

    dataset = json.load(open(dataset_path, 'r'))
    print('Size of dataset is %d' % len(dataset))

    print('multiprocess start!')

    num_of_each_dataset = len(dataset) / num_of_process

    result = []
    total_predicates = 0
    num_of_get_true_predicates = 0

    pool = multiprocessing.Pool(num_of_process)
    for index in range(num_of_process):
        print('process%d has started!' % (index + 1))
        # pool.apply_async(data_process, args=(dataset, index, num_of_each_dataset, ))
        result.append(pool.apply_async(data_process,
                                       args=(dataset, index, num_of_each_dataset,)))
    pool.close()
    pool.join()

    res = []
    for i in range(int(num_of_process)):
        data, num, is_get = result[i].get()
        res += data
        total_predicates += num
        num_of_get_true_predicates += is_get
    # json.dump(res, open(save_path % {'filename': 'data.json'}, 'w+'), indent=4)
    # print('saved data, size is %d' % (len(res)))

    train = res[:int(len(res)*0.7)]
    json.dump(train, open(save_path % {'filename': 'train.json'}, 'w+'), indent=4)

    validation = res[int(len(res)*0.7):int(len(res)*0.8)]
    json.dump(validation, open(save_path % {'filename': 'validation.json'}, 'w+'), indent=4)

    test = res[int(len(res)*0.8): len(res)]
    json.dump(test, open(save_path % {'filename': 'test.json'}, 'w+'), indent=4)

    print('saved data, train is %d, validation is %d, test is %d' % (len(train), len(validation), len(test)))
    print('the average num of predicates is %d, the percentage of get true predicates is %f'
          % (int(total_predicates / len(dataset)), num_of_get_true_predicates / len(dataset)))

    if reduce_glove is True:
        # find_and_save_glove_need(train + validation + test)
        find_and_save_glove_need(res)

    print('All is done! used time %s' % (datetime.now() - start_time))


def sort_dataset():
    """
    论文使用的数据集是更具id排序的
    :return:
    """
    dataset = json.load(open(dataset_path_origin, 'r'))
    dataset.sort(key=lambda x: int(x['id']))
    # print(dataset[0])
    json.dump(dataset, open(dataset_path_ranked, 'w'), indent=4)
    print('ranking the dataset by id is ok')


if __name__ == '__main__':
    """
    the average num of predicates is 1386, the percentage of get true predicates is 0.9966
    """
    sort_dataset()
    run(dataset_path_ranked, num_of_process=10, reduce_glove=True)
