import click
import os
import json
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
import panopticutils.ann_merger as merger
import panopticutils.label_converter as label_converter
import cv2

def load_config(path):
    config = {}
    with open(path) as f:
        config = json.load(f)
    return config

def process_img(img_id, img_file_name, coco, coco_res, masks_dir, id_merged, void_id=255):
    # Load stuff and intance results
    try:
        stuff_mask = merger.load_mask_png(img_file_name, masks_dir)
        detection_mask = merger.load_mask_ann(img_id, coco_res)
        instance_mask = merger.create_instance_mask(img_id, coco_res)
        # Convert labels
        converted = label_converter.convert(stuff_mask, id_merged)
        converted = label_converter.convert_void_id(converted, 255, void_id)
        # Convert to panoptic 2ch png format
        panoptic_2ch_png = merger.merge_stuff_instance(stuff_mask, instance_mask, instance_mask.shape)
    except:
        panoptic_2ch_png = None
    return panoptic_2ch_png

def save_img(img, img_file_name, out):
    path = '{}/{}.png'.format(out, img_file_name)
    cv2.imwrite(path, img)
    return
    
@click.command()
@click.option("--config_path", type=str, required=True)
def main(config_path):
    config = load_config(config_path)
    
    # Task annotations
    ann_data_dir = config['ann_data_dir']
    ann_data_name = config['ann_data_name']
    ann_data_file = '{}/{}'.format(ann_data_dir, ann_data_name)
    # Stuff masks
    masks_dir = config['stuff_masks_dir']
    # Detection results
    res_data_dir = config['res_data_dir']
    res_data_name = config['res_data_name']
    res_data_file='{}/{}'.format(res_data_dir, res_data_name)
    # Labels
    merged_stuff_label_file = config['merged_stuff_label_file']
    panoptic_label_file = config['panoptic_label_file']
    stuff_label_file = config['stuff_label_file']
    void_id = config['void_id']
    id_merged = label_converter.get_dict_id_stuff_to_panoptic(merged_stuff_label_file, stuff_label_file, panoptic_label_file)
    # Coco images and results for detection 
    coco = COCO(ann_data_file)
    coco_res = coco.loadRes(res_data_file)
    # Out folder
    out_dir = config['out_dir']
    
    # Load random image
    ids = coco.getImgIds()
    n_imgs = len(ids)
    count = 0
    for ii in ids:
        count += 1
        print('Processing {}/{} images...'.format(count, n_imgs), end='\r')
        img_id = [ii]

        # Get info
        img_info = coco.loadImgs(img_id)
        img_file_name = os.path.splitext(img_info[0]['file_name'])[0]

        #Process image and save output
        try:
            panoptic_2ch_png = process_img(img_id, img_file_name, coco, coco_res, masks_dir, id_merged, void_id)
            save_img(panoptic_2ch_png, img_file_name, out_dir)
        except:
            print('Couldn\'t process image {} (id {}), skipping it.'.format(count, ii), end='\n')
    print('Conversion completed; output is at {}'.format(out_dir))
    return

if __name__ == '__main__':
    main()