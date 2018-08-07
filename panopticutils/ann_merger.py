import os
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_mask_png(img_file_name, source_path):
    mask_path = '{}/{}'.format(source_path, img_file_name+'.png')
    mask = io.imread(mask_path)
    return mask
    
def load_mask_ann(img_id, coco):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    try:
        size = anns[0]['segmentation']['size']
        merged_mask = np.zeros(size, np.uint8)
        for ann in anns:
            mask = coco.annToMask(ann)
            cat_id = ann['category_id']
            merged_mask = np.where(mask != 1, merged_mask, cat_id)
    except:
        merged_mask = None
    return merged_mask

def create_instance_mask(img_id, coco):
    cls_count = dict()
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    size = anns[0]['segmentation']['size']
    merged_mask = np.zeros(size, np.uint8)
    for ann in anns:
        mask = coco.annToMask(ann)
        cat_id = ann['category_id']
        if (not cat_id in cls_count):
            cls_count[cat_id] = 1
        else:
            cls_count[cat_id] += 1
        merged_mask = np.where(mask != 1, merged_mask, cls_count[cat_id])
        #no overlap consideration
        if (not cls_count[cat_id] in merged_mask):
            cls_count[cat_id] -= 1
    return merged_mask
        
def show_mask(mask, title=''):
    plt.imshow(mask)
    plt.title(title)
    plt.show()
    
def merge_stuff_instance(stuff, instance, shape):
    cv_shape = (shape[1], shape[0])
    resized_stuff = stuff
    if (stuff.shape != shape):
         resized_stuff= cv2.resize(stuff, cv_shape)
    resized_instance = instance
    if (instance.shape != shape):
        resized_instance = cv2.resize(instance, cv_shape)
    merged_shape = (*shape, 3)
    merged = np.zeros(merged_shape, dtype=np.uint8)
    merged[:, :, 0] = resized_stuff
    merged[:, :, 1] = resized_instance
    return merged
    