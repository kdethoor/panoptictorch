import os
import json
import numpy as np

def get_dict_id_stuff_from_file(path):
    id_dict = {}
    data = []
    with open(path) as f:
        data = f.readlines()
        data = [x.strip().split('\t') for x in data]
        for i in range(len(data)):
            id_dict[data[i][1]] = int(data[i][0])
    return id_dict

def get_dict_id_panoptic_from_file(path):
    id_dict = {}
    data = []
    with open(path) as f:
        data = json.load(f)
        for i in range(len(data)):
            id_dict[data[i]['name']] = data[i]['id']
    return id_dict

def get_dict_cls_merged_from_file(path):
    data = {}
    with open(path) as f:
        data = json.load(f)
    return data

def get_dict_id_stuff_to_panoptic(path_to_merged_stuff_labels, path_to_stuff_labels, path_to_panoptic_labels, void_id=255):
    cls_id_panoptic = get_dict_id_panoptic_from_file(path_to_panoptic_labels)
    cls_id_stuff = get_dict_id_stuff_from_file(path_to_stuff_labels)
    cls_merged = get_dict_cls_merged_from_file(path_to_merged_stuff_labels)
    id_merged = {}
    for k, v in cls_merged.items():
        val = void_id
        if v != 'void':
            val = cls_id_panoptic[v]
        id_merged[cls_id_stuff[k]] = val
    return id_merged

def convert_void_id(img, previous_id, new_id):
    img = np.where(img == previous_id, new_id, img)
    return img

def convert(img, id_dict):
    for k, v in id_dict.items():
        img = np.where(img == k, v, img)
    return img