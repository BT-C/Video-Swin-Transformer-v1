import os
import json
import numpy as np

from torch import _test_serialization_subcmul

def test_relation(file_name):
    file = json.load(open(file_name, 'r'))
    relations = np.array([[0 for _ in range(17)] for _ in range(17)], dtype=np.float32)
    count_arr = np.array([0 for _ in range(17)])
    for key in file:
        arr = file[key]
        for i in range(len(arr)):
            count_arr[arr[i]] += 1
            for j in range(len(arr)):
                a = arr[i]
                b = arr[j]
                if i == j:
                    continue
                relations[a][b] += 1
                relations[b][a] += 1

    relations /= count_arr
    print(relations)

    t = 0.001
    relations[relations < t] = 0
    relations[relations >= t] = 1

    # _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    relations = relations * 0.25 / (relations.sum(0, keepdims=True) + 1e-6)
    # _adj = _adj + np.identity(num_classes, np.int)
    relations = relations + np.identity(17, np.int32)
    

    for re in relations:
        print(re)

    print(count_arr)


def test_result_json(file_name):
    file = json.load(open(file_name, 'r'))
    count = 0
    for key in file:
        arr = file[key]
        if arr[0] > 0.1:
            flag = 0
            for i in range(1, len(arr)):
                if arr[i] > 0.01:
                    flag = 1 
                    break
            if flag == 1:
                count += 1
                print(arr)
    print(count)

if __name__ == '__main__':
    file_name = '/mnt/hdd1/chenbeitao/data/datasets/UrbanPipe-Track/train.json'
    # file_name = '/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/result/json_result/test_result_backup.json'
    # file_name = '/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/Recurrent/result/json_result/test_result.json'
    test_relation(file_name)
    # test_result_json(file_name)