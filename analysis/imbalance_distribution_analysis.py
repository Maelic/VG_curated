# coding=utf8
# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

import argparse, json, string
from collections import Counter
import math

from math import floor
import h5py as h5
import numpy as np
import pprint
import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
from os import path
import pandas as pd

from imbalance_degree import *

"""
A script for generating an hdf5 ROIDB from the VisualGenome dataset
"""

def preprocess_object_labels(data, alias_dict={}):
    for img in data:
        for obj in img['objects']:
            obj['ids'] = [obj['object_id']]
            names = []
            for name in obj['names']:
                label = sentence_preprocess(name)
                if label in alias_dict:
                    label = alias_dict[label]
                names.append(label)
            obj['names'] = names
            if 'name' in obj.keys():
                label = sentence_preprocess(obj['name'])
                if label in alias_dict:
                    obj['name'] = alias_dict[label]

def preprocess_predicates(data, alias_dict={}):
    for img in data:
        for relation in img['relationships']:
            predicate = sentence_preprocess(relation['predicate'])
            if predicate in alias_dict:
                predicate = alias_dict[predicate]
            relation['predicate'] = predicate


def extract_object_token(data, num_tokens, rel_data=[], obj_list=[], img_list=[], verbose=True):
    """ Builds a set that contains the object names. Filters infrequent tokens. """

    print('Filtering object by number of relationships')
    my_file = open("VG/object_negative_list.txt", "r")
    negative_list = [line.replace('\n', '') for line in my_file.readlines()]

    token_counter = Counter()
    rel_counter = {}
    if len(rel_data) != 0:
        for img in rel_data:
            if not img_list or img['image_id'] in img_list:
                for relation in img['relationships']:
                    obj_id = relation['object']['object_id']
                    subj_id = relation['subject']['object_id']
                    rel_counter.setdefault(obj_id,[1]).append(1)
                    rel_counter.setdefault(subj_id,[1]).append(1)
    real_distri = Counter()
    token_counter = Counter()
    for img in data:
        for region in img['objects']:
            for name in region['names']:
                if name not in negative_list:
                    if not obj_list or name in obj_list:
                        if len(rel_data) == 0:
                            token_counter.update([name])
                            real_distri.update([name])
                        elif region['object_id'] in rel_counter:
                            token_counter.update([name]*len(rel_counter[region['object_id']]))
                            real_distri.update([name])
    tokens = set()
    # pick top N tokens
    token_counter_return = {}
    if not obj_list and num_tokens==0:
        for token, count in token_counter.most_common():
            tokens.add(token)
            token_counter_return[token] = real_distri[token]
    elif not obj_list:
        for token, count in token_counter.most_common():
            tokens.add(token)
            token_counter_return[token] = real_distri[token]
            if len(tokens) == num_tokens:
                break
    else:
        for token, count in token_counter.most_common():
            if token in obj_list:
                tokens.add(token)
                token_counter_return[token] = real_distri[token]
            
    if verbose:
        print(('Keeping %d / %d objects'
                  % (len(tokens), len(token_counter))))
    return tokens, token_counter_return

def extract_object_token_no_rel(data, num_tokens, obj_list=[],  img_list=[], verbose=True):
    """ Builds a set that contains the object names. Filters infrequent tokens. """
    print("Filtering object by number of occurences")
    token_counter = Counter()

    for img in data:
        if not img_list or img['image_id'] in img_list:
            for region in img['objects']:
                for name in region['names']:
                    if not obj_list or name in obj_list:
                        token_counter.update([name])
    tokens = set()
    # pick top N tokens
    token_counter_return = {}
    if not obj_list and num_tokens==0:
        for token, count in token_counter.most_common():
            tokens.add(token)
            token_counter_return[token] = count
    elif not obj_list:
        for token, count in token_counter.most_common():
            tokens.add(token)
            token_counter_return[token] = count
            if len(tokens) == num_tokens:
                break
    else:
        for token, count in token_counter.most_common():
            if token in obj_list:
                tokens.add(token)
                token_counter_return[token] = count
            
    if verbose:
        print(('Keeping %d / %d objects'
                  % (len(tokens), len(token_counter))))
    return tokens, token_counter_return


def extract_predicate_token(data, num_tokens, pred_list=[], img_list=[], verbose=True):
    """ Builds a set that contains the relationship predicates. Filters infrequent tokens. """

    my_file = open("VG/predicate_negative_list.txt", "r")
    negative_list = [line.replace('\n', '') for line in my_file.readlines()]

    token_counter = Counter()
    nb_image = 0
    for img in data:
        if not img_list or img['image_id'] in img_list:
            for relation in img['relationships']:
                predicate = relation['predicate']
                if predicate not in negative_list:
                    if not pred_list or predicate in pred_list:
                        token_counter.update([predicate])
            nb_image += 1

    tokens = set()
    token_counter_return = {}

    if not pred_list and num_tokens==0:
        for token, count in token_counter.most_common():
            tokens.add(token)
            token_counter_return[token] = count
    elif not pred_list:
        for token, count in token_counter.most_common():
            tokens.add(token)
            token_counter_return[token] = count
            if len(tokens) == num_tokens:
                break
    else:
        for token, count in token_counter.most_common():
            if token in pred_list:
                tokens.add(token)
                token_counter_return[token] = count

    if verbose:
        print(('Keeping %d / %d predicates'
                  % (len(tokens), len(token_counter))))
    return tokens, token_counter_return

def merge_duplicate_boxes(data):
    def IoU(b1, b2):
        if b1[2] <= b2[0] or \
            b1[3] <= b2[1] or \
            b1[0] >= b2[2] or \
            b1[1] >= b2[3]:
            return 0

        b1b2 = np.vstack([b1,b2])
        minc = np.min(b1b2, 0)
        maxc = np.max(b1b2, 0)
        union_area = (maxc[2]-minc[0])*(maxc[3]-minc[1])
        int_area = (minc[2]-maxc[0])*(minc[3]-maxc[1])
        return float(int_area)/float(union_area)

    def to_x1y1x2y2(obj):
        x1 = obj['x']
        y1 = obj['y']
        x2 = obj['x'] + obj['w']
        y2 = obj['y'] + obj['h']
        return np.array([x1, y1, x2, y2], dtype=np.int32)

    def inside(b1, b2):
        return b1[0] >= b2[0] and b1[1] >= b2[1] \
            and b1[2] <= b2[2] and b1[3] <= b2[3]

    def overlap(obj1, obj2):
        b1 = to_x1y1x2y2(obj1)
        b2 = to_x1y1x2y2(obj2)
        iou = IoU(b1, b2)
        if all(b1 == b2) or iou > 0.9: # consider as the same box
            return 1
        elif (inside(b1, b2) or inside(b2, b1))\
            and obj1['names'][0] == obj2['names'][0]: # same object inside the other
            return 2
        elif iou > 0.6 and obj1['names'][0] == obj2['names'][0]: # multiple overlapping same object
            return 3
        else:
            return 0  # no overlap

    num_merged = {1:0, 2:0, 3:0}
    print('merging boxes..')
    for img in tqdm(data):
        # mark objects to be merged and save their ids
        objs = img['objects']
        num_obj = len(objs)
        for i in range(num_obj):
            if 'M_TYPE' in objs[i]:  # has been merged
                continue
            merged_objs = [] # circular refs, but fine
            for j in range(i+1, num_obj):
                if 'M_TYPE' in objs[j]:  # has been merged
                    continue
                overlap_type = overlap(objs[i], objs[j])
                if overlap_type > 0:
                    objs[j]['M_TYPE'] = overlap_type
                    merged_objs.append(objs[j])
            objs[i]['mobjs'] = merged_objs

        # merge boxes
        filtered_objs = []
        merged_num_obj = 0
        for obj in objs:
            if 'M_TYPE' not in obj:
                ids = [obj['object_id']]
                dims = [to_x1y1x2y2(obj)]
                prominent_type = 1
                for mo in obj['mobjs']:
                    ids.append(mo['object_id'])
                    obj['names'].extend(mo['names'])
                    dims.append(to_x1y1x2y2(mo))
                    if mo['M_TYPE'] > prominent_type:
                        prominent_type = mo['M_TYPE']
                merged_num_obj += len(ids)
                obj['ids'] = ids
                mdims = np.zeros(4)
                if prominent_type > 1: # use extreme
                    mdims[:2] = np.min(np.vstack(dims)[:,:2], 0)
                    mdims[2:] = np.max(np.vstack(dims)[:,2:], 0)
                else:  # use mean
                    mdims = np.mean(np.vstack(dims), 0)
                obj['x'] = int(mdims[0])
                obj['y'] = int(mdims[1])
                obj['w'] = int(mdims[2] - mdims[0])
                obj['h'] = int(mdims[3] - mdims[1])

                num_merged[prominent_type] += len(obj['mobjs'])

                obj['mobjs'] = None
                obj['names'] = list(set(obj['names']))  # remove duplicates

                filtered_objs.append(obj)
            else:
                assert 'mobjs' not in obj

        img['objects'] = filtered_objs
        assert(merged_num_obj == num_obj)

    print('# merged boxes per merging type:')
    print(num_merged)

    # saving to json
    print('saving to json..')
    with open('VG/merged_boxes.json', 'w') as f:
        json.dump(data, f)


def build_token_dict(vocab):
    """ build bi-directional mapping between index and token"""
    token_to_idx, idx_to_token = {}, {}
    next_idx = 1
    vocab_sorted = sorted(list(vocab)) # make sure it's the same order everytime
    for token in vocab_sorted:
        token_to_idx[token] = next_idx
        idx_to_token[next_idx] = token
        next_idx = next_idx + 1

    return token_to_idx, idx_to_token


def encode_box(region, org_h, org_w, im_long_size):
    x = region['x']
    y = region['y']
    w = region['w']
    h = region['h']
    scale = float(im_long_size) / max(org_h, org_w)
    image_size = im_long_size
    # recall: x,y are 1-indexed
    x, y = math.floor(scale*(region['x']-1)), math.floor(scale*(region['y']-1))
    w, h = math.ceil(scale*region['w']), math.ceil(scale*region['h'])

    # clamp to image
    if x < 0: x = 0
    if y < 0: y = 0

    # box should be at least 2 by 2
    if x > image_size - 2:
        x = image_size - 2
    if y > image_size - 2:
        y = image_size - 2
    if x + w >= image_size:
        w = image_size - x
    if y + h >= image_size:
        h = image_size - y

    # also convert to center-coord oriented
    box = np.asarray([x+floor(w/2), y+floor(h/2), w, h], dtype=np.int32)
    assert box[2] > 0  # width height should be positive numbers
    assert box[3] > 0
    return box


def encode_objects(obj_data, token_to_idx, token_counter, org_h, org_w, im_long_sizes):
    encoded_labels = []
    encoded_boxes  = {}
    for size in im_long_sizes:
        encoded_boxes[size] = []
    im_to_first_obj = np.zeros(len(obj_data), dtype=np.int32)
    im_to_last_obj = np.zeros(len(obj_data), dtype=np.int32)
    obj_counter = 0
    obj_distribution = {}

    for i, img in enumerate(obj_data):  
        im_to_first_obj[i] = obj_counter
        img['id_to_idx'] = {}  # object id to region idx
        for obj in img['objects']:
           # pick a label for the object
            max_occur = 0
            obj_label = None
            for name in obj['names']:
                # pick the name that has maximum occurance
                if name in token_to_idx and token_counter[name] > max_occur:
                    obj_label = name
                    max_occur = token_counter[obj_label]

            if obj_label is not None:
                # encode region
                for size in im_long_sizes:
                    encoded_boxes[size].append(encode_box(obj, org_h[i], org_w[i], size))

                encoded_labels.append(token_to_idx[obj_label])

                for obj_id in obj['ids']: # assign same index for merged ids
                    img['id_to_idx'][obj_id] = obj_counter

                obj_counter += 1


        if im_to_first_obj[i] == obj_counter:
            im_to_first_obj[i] = -1
            im_to_last_obj[i] = -1
        else:
            im_to_last_obj[i] = obj_counter - 1
            nb_obj = obj_counter - im_to_first_obj[i]
            if nb_obj in obj_distribution.keys():
                obj_distribution[nb_obj] += 1
            else:
                obj_distribution[nb_obj] = 1

    for k, boxes in encoded_boxes.items():
        encoded_boxes[k] = np.vstack(boxes)
    
    print('Distribution of objects per image: ', dict(sorted(obj_distribution.items(), key=lambda x: x[1])))

    print('%i out of %i valid images have at least one object' % (sum(obj_distribution.values()), len(obj_data)))

    return np.vstack(encoded_labels), encoded_boxes, im_to_first_obj, im_to_last_obj


def encode_relationship(sub_id, obj_id, id_to_idx):
    # builds a tuple of the index of object and subject in the object list
    sub_idx = id_to_idx[sub_id]
    obj_idx = id_to_idx[obj_id]
    return np.asarray([sub_idx, obj_idx], dtype=np.int32)

def get_label(obj):
    if 'names' in obj.keys():
        return obj['names'][0]
    else:
        return obj['name']

def check_part_whole2(img, part_whole_df):

    values = part_whole_df['sentence'].to_list()
    rels = []
    G = nx.MultiDiGraph()
    init_size = len(img['relationships'])
    for relation in img['relationships']:
        subj = relation['subject']

        obj = relation['object']
        predicate = relation['predicate'].lower()
        
        obj_label = get_label(obj)
        subj_label = get_label(subj)

        full_rel = str(subj_label + ' ' + predicate + ' ' + obj_label)
        rels.append(full_rel)

        if full_rel in values:
            #print("part-whole rel: ", full_rel)
            G.add_edge(subj['object_id'], obj['object_id'], weight=0, label=predicate)
        else:
            G.add_edge(subj['object_id'], obj['object_id'], weight=1, label=predicate)
    
    weight = [e for e in G.edges(data=True) if e[2]['weight']==0]
    for edge in weight:
        if G.out_degree(edge[0]) == 1 and G.in_degree(edge[1]) == 1:
            G.remove_edges_from([(edge[0], edge[1])])
        elif G.out_degree(edge[1]) == 0:
            G.remove_edges_from([(edge[0], edge[1])])
    break_out = False

    while not break_out:
        to_remove = []
        leaves = [x for x in G.nodes() if G.out_degree(x)==0]
        for l in leaves:

            weight = [e for e in G.in_edges(l,data=True)]
            if weight == []:
                continue
            if all([w[2]['weight'] == 0 for w in weight]):
                to_remove.append((weight[0][0], weight[0][1]))
        G.remove_edges_from(to_remove)

        if len(to_remove) == 0:
            break_out = True

    new_rel = []
    for relation in img['relationships']:
        if (relation['subject']['object_id'], relation['object']['object_id']) in G.edges():
            new_rel.append(relation)
    number_filtered = init_size - len(new_rel)
    return number_filtered, new_rel

def check_part_whole(img, part_whole_df):
    values = part_whole_df['sentence'].to_list()
    output = []
    init_size = len(img['relationships'])
    for relation in img['relationships']:
        subj = relation['subject']

        obj = relation['object']
        predicate = relation['predicate'].lower()

        obj_label = get_label(obj)
        subj_label = get_label(subj)

        full_rel = str(subj_label + ' ' + predicate + ' ' + obj_label)

        if full_rel in values:
            continue
        else:
            output.append(relation)
    
    number_filtered = init_size - len(output)
    return number_filtered, output

def filter_part_whole(rel_data, part_whole_path=''):
    print('Filtering part-whole relationships')

    part_whole_df = pd.read_csv(part_whole_path, sep=',')
    part_whole_filtered = 0
    for img in tqdm(rel_data):
        # filter part-whole relationships using dict
        relations_filtered, img['relationships'] = check_part_whole(img, part_whole_df)
        part_whole_filtered += relations_filtered
    
    print('%i rel is filtered by part-whole' % part_whole_filtered)
    return rel_data

def encode_relationships(rel_data, token_to_idx, obj_data, img_list=[]):
    """MUST BE CALLED AFTER encode_objects!!!"""

    img_with_rels = []
    encoded_pred = []  # encoded predicates
    encoded_rel = []  # encoded relationship tuple
    im_to_first_rel = np.zeros(len(rel_data), dtype=np.int32)
    im_to_last_rel = np.zeros(len(rel_data), dtype=np.int32)
    rel_idx_counter = 0
    rel_distribution = {}
    no_rel_counter = 0
    obj_filtered = 0
    predicate_filtered = 0
    duplicate_filtered = 0
    relations_filtered = 0
    idx_img_rels = []
    print("img list length: ",len(img_list))
    for i, img in tqdm(enumerate(rel_data)):
        if len(img_list) > 0 and img['image_id'] not in img_list:
            # if image not in list, skip
            im_to_first_rel[i] = -1
            im_to_last_rel[i] = -1
            no_rel_counter += 1
            continue
        else:
            idx_img_rels.append(i)

            number_of_rel=0
            im_to_first_rel[i] = rel_idx_counter
            id_to_idx = obj_data[i]['id_to_idx']  # object id to object list idx

            for relation in img['relationships']:

                subj = relation['subject']
                obj = relation['object']
                predicate = relation['predicate']
                
                if subj['object_id'] not in id_to_idx or obj['object_id'] not in id_to_idx:
                    obj_filtered += 1
                    continue
                elif predicate not in token_to_idx:
                    predicate_filtered += 1
                    continue
                elif id_to_idx[subj['object_id']] == id_to_idx[obj['object_id']]: # sub and obj can't be the same box
                    duplicate_filtered += 1
                    continue
                else:
                    encoded_pred.append(token_to_idx[predicate])
                    encoded_rel.append(encode_relationship(subj['object_id'], obj['object_id'], id_to_idx))

                    rel_idx_counter += 1  # accumulate counter
                    number_of_rel += 1

            if im_to_first_rel[i] == rel_idx_counter:
                # if no qualifying relationship
                im_to_first_rel[i] = -1
                im_to_last_rel[i] = -1
                no_rel_counter += 1
            else:
                if number_of_rel in rel_distribution.keys():
                    rel_distribution[number_of_rel] += 1
                else:
                    rel_distribution[number_of_rel] = 1
                im_to_last_rel[i] = rel_idx_counter - 1
                img_with_rels.append(img['image_id'])

    with open("valid_image_id.json", "w") as fp:
        json.dump(list(img_with_rels), fp)

    print('%i rel is filtered by object' % obj_filtered)
    print('%i rel is filtered by predicate' % predicate_filtered)
    print('%i rel is filtered by duplicate' % duplicate_filtered)
    print('%i rel is manually filtered using dict' % relations_filtered)
    #print('%i rel is filtered by number per image' % images_filtered)

    print('%i rel remains ' % len(encoded_pred))
    print('Distribution of relationships per image: ', dict(sorted(rel_distribution.items(), key=lambda x: x[1])))

    print('%i out of %i valid images have relationships' % (len(rel_data)-no_rel_counter, len(rel_data)))
    return np.vstack(encoded_pred), np.vstack(encoded_rel), im_to_first_rel, im_to_last_rel, idx_img_rels

def sentence_preprocess(phrase):
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    replacements = {
      '½': 'half',
      '—' : '-',
      '™': '',
      '¢': 'cent',
      'ç': 'c',
      'û': 'u',
      'é': 'e',
      '°': ' degree',
      'è': 'e',
      '…': '',
    }
    #phrase = phrase.encode('utf-8')
    phrase = phrase.strip()
    for k, v in replacements.items():
        phrase = str(phrase).replace(k, v)
    return str(phrase).lower().translate(str.maketrans('','',string.punctuation))#.decode('utf-8', 'ignore')


def encode_splits(obj_data, opt=None):
    if opt is not None:
        val_begin_idx = opt['val_begin_idx']
        test_begin_idx = opt['test_begin_idx']
    split = np.zeros(len(obj_data), dtype=np.int32)
    for i, info in enumerate(obj_data):
        splitix = 0
        if opt is None: # use encode from input file
            s = info['split']
            if s == 'val': splitix = 1
            if s == 'test': splitix = 2
        else: # use portion split
            if i >= val_begin_idx: splitix = 1
            if i >= test_begin_idx: splitix = 2
        split[i] = splitix
    if opt is not None and opt['shuffle']:
        np.random.shuffle(split)

    print(('assigned %d/%d/%d to train/val/test split' % (np.sum(split==0), np.sum(split==1), np.sum(split==2))))
    return split

def encode_splits_rel(rel_data, idx_img_rels, test_ratio, val_ratio, shuffle=None):
    test_split = int(len(idx_img_rels) * test_ratio)
    val_split = int(len(idx_img_rels) * val_ratio)

    splitix = np.zeros(len(idx_img_rels), dtype=np.int32)
    splitix[test_split:] = [2]*(len(splitix)-test_split)

    splitix[test_split:val_split] = [1]*(val_split-test_split)

    if shuffle:
        np.random.shuffle(splitix)

    split = np.array([-1 for p in range(0,len(rel_data))])

    j = 0
    for i in range(len(rel_data)):
        if i in idx_img_rels:
            split[i] = int(splitix[j])
            j += 1

    # sanity check
    assert j == len(idx_img_rels)

    print('Relations: assigned %d/%d/%d to train/val/test split' % (np.sum(split==0), np.sum(split==1), np.sum(split==2)))
    return split

def make_alias_dict(dict_file):
    """create an alias dictionary from a file"""
    out_dict = {}
    vocab = []
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab


def make_list(list_file):
    """create a blacklist list from a file"""
    return [line.strip('\n').strip('\r') for line in open(list_file)]


def filter_object_boxes(data, heights, widths, area_frac_thresh):
    """
    filter boxes by a box area-image area ratio threshold
    """
    thresh_count = 0
    all_count = 0
    for i, img in tqdm(enumerate(data)):
        filtered_obj = []
        area = float(heights[i]*widths[i])
        for obj in img['objects']:
            if float(obj['h'] * obj['w']) > area * area_frac_thresh:
                filtered_obj.append(obj)
                thresh_count += 1
            all_count += 1
        img['objects'] = filtered_obj
    print(('box threshod: keeping %i/%i boxes' % (thresh_count, all_count)))


def filter_by_idx(data, valid_list):
    return [data[i] for i in valid_list]


def obj_rel_cross_check(obj_data, rel_data, verbose=False):
    """
    make sure all objects that are in relationship dataset
    are in object dataset
    """
    num_img = len(obj_data)
    num_correct = 0
    total_rel = 0
    for i in range(num_img):
        assert(obj_data[i]['image_id'] == rel_data[i]['image_id'])
        objs = obj_data[i]['objects']
        rels = rel_data[i]['relationships']
        ids = [obj['object_id'] for obj in objs]
        for rel in rels:
            if rel['subject']['object_id'] in ids \
                and rel['object']['object_id'] in ids:
                num_correct += 1
            elif verbose:
                if rel['subject']['object_id'] not in ids:
                    print((str(rel['subject']['object_id']) + 'cannot be found in ' + str(i)))
                if rel['object']['object_id'] not in ids:
                    print((str(rel['object']['object_id']) + 'cannot be found in ' + str(i)))
            total_rel += 1
    print(('cross check: %i/%i relationship are correct' % (num_correct, total_rel)))


def sync_objects(obj_data, rel_data):
    num_img = len(obj_data)
    for i in range(num_img):
        assert(obj_data[i]['image_id'] == rel_data[i]['image_id'])
        objs = obj_data[i]['objects']
        rels = rel_data[i]['relationships']

        ids = [obj['object_id'] for obj in objs]
        for rel in rels:
            if rel['subject']['object_id'] not in ids:
                rel_obj = rel['subject']
                rel_obj['names'] = [rel_obj['name']]
                objs.append(rel_obj)
            if rel['object']['object_id'] not in ids:
                rel_obj = rel['object']
                rel_obj['names'] = [rel_obj['name']]
                objs.append(rel_obj)

        obj_data[i]['objects'] = objs

def load_images_list(image_list_path):
    """
    load a list of image ids
    """
    image_id_list = []
    
    if '.csv' in image_list_path:
        image_id_list = pd.read_csv(image_list_path, sep=",", usecols=['Image_id'])
        image_id_list = image_id_list['Image_id'].tolist()
    if '.txt' in image_list_path:
        for line in open(image_list_path, 'r'):
            image_id_list.append(int(line.strip('\n').strip('\r')))
    if '.json' in image_list_path:
        image_id_list = json.load(open(image_list_path, 'r'))

    return image_id_list

def compute_imbalance(df_vg):
    df_vg["full_rel"] = df_vg['node1'].astype(str) +"-"+ df_vg["relation"].astype(str) +"-"+ df_vg["node2"].astype(str)
    df_val_counts = df_vg["full_rel"].value_counts()
    size_triplets = len(df_val_counts)

    df_val_counts = df_vg["relation"].value_counts()
    size_pred = len(df_val_counts)

    ir_pred = imbalance_ratio(df_vg, "relation")
    print("Imbalance ratio for predicate: ", ir_pred)
    id_pred = imbalance_degree(df_vg, "relation", size_pred)
    print("Imbalance degree for predicate: ", id_pred)
    llid_pred = log_likelihood_index(df_vg, "relation")
    print("Log likelihood index for predicate: ", llid_pred)

    ir_triplets = imbalance_ratio(df_vg, "full_rel")
    print("Imbalance ratio for triplets: ", ir_triplets)
    id_triplets = imbalance_degree(df_vg, "full_rel", size_triplets)
    print("Imbalance degree for triplets: ", id_triplets)
    llid_triplets = log_likelihood_index(df_vg, "full_rel")
    print("Log likelihood index for triplets: ", llid_triplets)

def main(args):
    print('start')
    pprint.pprint(args)

    obj_alias_dict = {}
    if len(args.object_alias) > 0:
        print(('using object alias from %s' % (args.object_alias)))
        obj_alias_dict, obj_vocab_list = make_alias_dict(args.object_alias)

    pred_alias_dict = {}
    if len(args.pred_alias) > 0:
        print(('using predicate alias from %s' % (args.pred_alias)))
        pred_alias_dict, pred_vocab_list = make_alias_dict(args.pred_alias)

    obj_list = []
    if len(args.object_list) > 0:
        print(('using object list from %s' % (args.object_list)))
        obj_list = make_list(args.object_list)
        assert(len(obj_list) >= args.num_objects)

    pred_list = []
    if len(args.pred_list) > 0:
        print(('using predicate list from %s' % (args.pred_list)))
        pred_list = make_list(args.pred_list)
        assert(len(pred_list) >= args.num_predicates)

    # read in the annotation data
    print('loading json files..')
    obj_data = json.load(open(args.object_input))
    rel_data = json.load(open(args.relationship_input))
    img_data = json.load(open(args.metadata_input))
    assert(len(rel_data) == len(obj_data) and len(obj_data) == len(img_data))

    print(('read image db from %s' % args.imdb))
    imdb = h5.File(args.imdb, 'r')
    num_im, _, _, _ = imdb['images'].shape
    img_long_sizes = [512, 1024]
    valid_im_idx = imdb['valid_idx'][:] # valid image indices
    print(('total number of valid images: %i' % len(valid_im_idx)))
    img_ids = imdb['image_ids'][:]
    obj_data = filter_by_idx(obj_data, valid_im_idx)
    rel_data = filter_by_idx(rel_data, valid_im_idx)
    img_data = filter_by_idx(img_data, valid_im_idx)

    # sanity check
    for i in range(num_im):
        assert(obj_data[i]['image_id'] \
               == rel_data[i]['image_id'] \
               == img_data[i]['image_id']  \
               == img_ids[i]
               )

    # may only load a fraction of the data
    if args.load_frac < 1:
        num_im = int(num_im*args.load_frac)
        obj_data = obj_data[:num_im]
        rel_data = rel_data[:num_im]
    print(('processing %i images' % num_im))

    # sync objects from rel to obj_data
    sync_objects(obj_data, rel_data)

    obj_rel_cross_check(obj_data, rel_data)

    # preprocess label data
    preprocess_object_labels(obj_data, alias_dict=obj_alias_dict)
    preprocess_predicates(rel_data, alias_dict=pred_alias_dict)

    if args.part_whole_path is not None:
        if os.path.exists("VG/rel_data_part_whole_filtered.json"):
            rel_data = json.load(open("VG/rel_data_part_whole_filtered.json", 'r'))
        else:
            print(('filtering part-whole relationships from %s' % args.part_whole_path))
            rel_data = filter_part_whole(rel_data, args.part_whole_path)
            json.dumb(rel_data, open('VG/rel_data_part_whole_filtered.json', 'w'))

    heights, widths = imdb['original_heights'][:], imdb['original_widths'][:]
    if args.min_box_area_frac > 0:
        # filter out invalid small boxes
        print(('threshold bounding box by %f area fraction' % args.min_box_area_frac))
        filter_object_boxes(obj_data, heights, widths, args.min_box_area_frac) # filter by box dimensions

    if not path.exists("VG/merged_boxxxes.json"):
        merge_duplicate_boxes(obj_data)
    else:
        print("using merged boxes from VG/merged_boxes.json")
        obj_data = json.load(open("VG/merged_boxes.json", 'r'))
    # build vocabulary

    pred_number = [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]

    for pred in pred_number:
        args.num_predicates = pred
        print("Generating with num_predicates= ", args.num_predicates)

        img_list = []
        if args.images_list is not None:
            img_list = load_images_list(args.images_list)
        predicate_tokens, predicate_token_counter = extract_predicate_token(rel_data, args.num_predicates, pred_list, img_list=img_list)
        if args.class_selection:
            object_tokens, object_token_counter = extract_object_token(obj_data, args.num_objects, rel_data, obj_list, img_list=img_list)
        else:
            object_tokens, object_token_counter = extract_object_token_no_rel(obj_data, args.num_objects, obj_list, img_list=img_list)
        predicate_to_idx, idx_to_predicate = build_token_dict(predicate_tokens)
        label_to_idx, idx_to_label = build_token_dict(object_tokens)

        # print out vocabulary
        print('objects: ')
        print(object_token_counter)
        print('relationships: ')
        print(predicate_token_counter)

        # encode object
        encoded_label, encoded_boxes, im_to_first_obj, im_to_last_obj = \
        encode_objects(obj_data, label_to_idx, object_token_counter, \
                    heights, widths, img_long_sizes)

        encoded_predicate, encoded_rel, im_to_first_rel, im_to_last_rel, idx_img_rels = \
        encode_relationships(rel_data, predicate_to_idx, obj_data, img_list=img_list)

    # build train/val/test splits
        print(('num objects = %i' % encoded_label.shape[0]))
        print(('num relationships = %i' % encoded_predicate.shape[0]))

        rel_array=[]
        idx_to_label = idx_to_label
        idx_to_predicate = idx_to_predicate
        for i in tqdm(range(encoded_predicate.shape[0])):
            objs = encoded_rel[i]
            rel_array.append([idx_to_label[str(encoded_label[objs[0]][0])], \
                idx_to_predicate[str(encoded_predicate[i][0])], \
                    idx_to_label[str(encoded_label[objs[1]][0])]])

        df_vg = pd.DataFrame(rel_array, columns=['node1', 'relation', 'node2'])
        compute_imbalance(df_vg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdb', default='imdb_1024.h5', type=str)
    parser.add_argument('--object_input', default='VG/objects_v1_2/objects.json', type=str)
    parser.add_argument('--relationship_input', default='VG/relationships.json', type=str)
    parser.add_argument('--metadata_input', default='VG/image_data.json', type=str)
    parser.add_argument('--object_alias', default='VG/object_alias_without_gender.txt', type=str)
    parser.add_argument('--pred_alias', default='VG/predicate_alias.txt', type=str)
    parser.add_argument('--object_list', default='', type=str)
    parser.add_argument('--pred_list', default='', type=str)
    parser.add_argument('--num_objects', default=150, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--num_predicates', default=50, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--min_box_area_frac', default=0.002, type=float)
    parser.add_argument('--load_frac', default=1, type=float)
    parser.add_argument('--use_input_split', default=False, type=bool)
    parser.add_argument('--train_frac', default=0.65, type=float)
    parser.add_argument('--val_frac', default=0.7, type=float)
    parser.add_argument('--shuffle', default=False, type=bool)
    parser.add_argument('--part_whole_path', default="", type=str, help="set to none to disable filtering part-whole relationships")
    parser.add_argument('--images_list', default="", type=str, help="list of images to process")
    parser.add_argument('--class_selection', default=True, type=str, help="how to select the object and predicate classes, if true, use the relationship occurence, else use the object occurence")

    args = parser.parse_args()


    main(args)
