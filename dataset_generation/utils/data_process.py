import os
import json
import h5py as h5
from .bboxes import merge_duplicate_boxes, filter_object_boxes, encode_box
import string
import pandas as pd
from tqdm import tqdm
import numpy as np
import copy
import torch
from .attributes import generate_attributes

def bounding_boxes_process(obj_data, min_box_area_frac, heights, widths, logger, base_dir="", save=False):
    # check if file exist
    file_path = os.path.join(base_dir, '.cache/merged_boxes.json')
    if os.path.isfile(file_path):
        logger.info("Using merged boxes from {}".format(file_path))
        obj_data = json.load(open(file_path, 'r'))
    else:
        if min_box_area_frac > 0:
            # filter out invalid small boxes
            logger.info('threshold bounding box by {} area fraction'.format(min_box_area_frac))
            filter_object_boxes(obj_data, heights, widths, min_box_area_frac) # filter by box dimensions

        merge_duplicate_boxes(obj_data, logger, save=save, dest_path=file_path)

def preprocess_data(args, logger):
    obj_list = []
    if len(args.object_list) > 0:
        logger.info('Using object list from {}'.format(args.object_list))
        obj_list = [line.strip('\n').strip('\r') for line in open(args.object_list)]
        assert(len(obj_list) >= args.num_objects)

    pred_list = []
    if len(args.pred_list) > 0:
        logger.info('Using predicate list from {}'.format(args.pred_list))
        pred_list = [line.strip('\n').strip('\r') for line in open(args.pred_list)]
        assert(len(pred_list) >= args.num_predicates)

    # check if dest folder exist
    if not os.path.exists(os.path.dirname(args.h5_file)):
        os.makedirs(os.path.dirname(args.h5_file))

    # read in the annotation data
    logger.info("Loading annotations files...")
    obj_data = json.load(open(args.object_input))
    rel_data = json.load(open(args.relationship_input))
    img_data = json.load(open(args.metadata_input))
    assert(len(rel_data) == len(obj_data) and len(obj_data))

    logger.info("Loaded {} images".format(len(obj_data)))
    imdb = h5.File(args.imdb, 'r')
    num_im, _, _, _ = imdb['images'].shape

    valid_im_idx = imdb['valid_idx'][:] # valid image indices
    logger.info("There is a total of {} valid images".format(len(valid_im_idx)))
    img_ids = imdb['image_ids'][:]
    obj_data = [obj_data[i] for i in valid_im_idx]
    rel_data = [rel_data[i] for i in valid_im_idx]
    img_data = [img_data[i] for i in valid_im_idx]

    #sanity check
    for i in range(num_im):
        assert(obj_data[i]['image_id'] \
               == rel_data[i]['image_id'] \
               == img_data[i]['image_id']  \
               == img_ids[i]
               )
    
    sync_objects(obj_data, rel_data, logger)

    img_heights, img_widths = imdb['original_heights'][:], imdb['original_widths'][:]

    # close imdb file tos ave memory
    imdb.close()
    # close img_data file to save memory
    img_data = None

    return obj_data, rel_data, img_heights, img_widths

def sync_objects(obj_data, rel_data, logger, verbose=False):

    num_correct = 0
    total_rel = 0

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

        # Object / rels cross check
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
    logger.info('Cross check: {}/{} relationship are correct'.format(num_correct, total_rel))

def preprocess_labels(obj_dat, pred_data, obj_alias, pred_alias, logger):
    """ preprocess object and predicate labels """

    # check if obj_alias file exist
    if not os.path.isfile(obj_alias):
        logger.info("Wrong object alias file {} or not provided, using default object list ".format(obj_alias))
        obj_alias = ""

    # check if obj_alias file exist
    if not os.path.isfile(pred_alias):
        logger.info("Wrong object alias file {} or not provided, using default object list ".format(pred_alias))
        pred_alias = ""

    preprocess_object_labels(obj_dat, obj_alias)
    preprocess_predicates(pred_data, pred_alias)

def make_alias_dict(dict_file):
    """create an alias dictionary from a file"""
    out_dict = {}
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
    return out_dict

def preprocess_object_labels(data, dict_file=""):
    if dict_file != "":
        alias_dict = make_alias_dict(dict_file)
    else:
        alias_dict = {}
    for img in data:
        for obj in img['objects']:
            obj['ids'] = [obj['object_id']]
            names = []
            for name in obj['names']:
                label = sentence_preprocess(name)
                if label in alias_dict.keys():
                    label = alias_dict[label]
                names.append(label)
            obj['names'] = names
            if 'name' in obj.keys():
                label = sentence_preprocess(obj['name'])
                if label in alias_dict.keys():
                    obj['name'] = alias_dict[label]

def preprocess_predicates(data, dict_file=""):
    if dict_file != "":
        alias_dict = make_alias_dict(dict_file)
    else:
        alias_dict = {}
    for img in data:
        for relation in img['relationships']:
            predicate = sentence_preprocess(relation['predicate'])
            if predicate in alias_dict:
                predicate = alias_dict[predicate]
            relation['predicate'] = predicate


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

def build_class_lists(obj_list_path, pred_list_path, logger):
    obj_list = []
    if len(obj_list_path) > 0:
        logger.info('Using object list from {}'.format(obj_list_path))
        obj_list = [line.strip('\n').strip('\r') for line in open(obj_list_path)]

    pred_list = []
    if len(pred_list_path) > 0:
        logger.info('Using predicate list from {}'.format(pred_list_path))
        pred_list = [line.strip('\n').strip('\r') for line in open(pred_list_path)]

    return obj_list, pred_list

def postprocess_data(args, logger, predicate_token_counter, object_token_counter, obj_data, rel_data, img_list, heights, widths, base_path):

    img_long_sizes = [512, 1024]
    img_data = json.load(open(args.metadata_input))

    predicate_to_idx, idx_to_predicate = build_token_dict(predicate_token_counter.keys())
    label_to_idx, idx_to_label = build_token_dict(object_token_counter.keys())

    logger.info('Object vocab: {}'.format(object_token_counter.keys()))
    logger.info('Predicate vocab: {}'.format(predicate_token_counter.keys()))

    # encode object
    encoded_label, encoded_boxes, im_to_first_obj, im_to_last_obj, new_obj_data = \
    encode_objects(obj_data, label_to_idx, object_token_counter, \
                   heights, widths, img_long_sizes, args.yolo_anno, img_data, img_list=img_list, logger=logger)

    encoded_predicate, encoded_rel, im_to_first_rel, im_to_last_rel, idx_img_rels = \
    encode_relationships(rel_data, predicate_to_idx, obj_data, img_list=img_list)

    opt = None
    if not args.use_input_split:
        opt = {}
        opt['val_begin_idx'] = int(len(obj_data) * args.train_frac)
        opt['test_begin_idx'] = int(len(obj_data) * args.val_frac)
        opt['shuffle'] = args.shuffle

    split = encode_splits(obj_data, logger, opt)

    split_rel = encode_splits_rel(rel_data, idx_img_rels, args.train_frac, args.val_frac, logger=logger, shuffle=args.shuffle)

    if args.yolo_anno == 'True':
        logger.info("Generating YOLO annotations to {}".format(base_path+'/yolo_anno'))
        generate_yolo_anno(new_obj_data, img_data, label_to_idx, base_path)

    # and write the additional json file
    json_struct = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'predicate_to_idx': predicate_to_idx,
        'idx_to_predicate': idx_to_predicate,
        'predicate_count': predicate_token_counter,
        'object_count': object_token_counter
    }

    # write the h5 file
    f = h5.File(args.h5_file, 'w')

    f.create_dataset('labels', data=encoded_label)
    for k, boxes in list(encoded_boxes.items()):
        f.create_dataset('boxes_%i' % k, data=boxes)
    f.create_dataset('img_to_first_box', data=im_to_first_obj)
    f.create_dataset('img_to_last_box', data=im_to_last_obj)

    print("size idx_img_rels: ", len(idx_img_rels))
    f.create_dataset('predicates', data=encoded_predicate)
    f.create_dataset('relationships', data=encoded_rel)
    f.create_dataset('img_to_first_rel', data=im_to_first_rel)
    f.create_dataset('img_to_last_rel', data=im_to_last_rel)

    if args.attributes:
        if args.attributes_input == "":
            raise ValueError("Attribute input file not provided")
        elif args.attributes_synsets == "":
            raise ValueError("Attribute synset file not provided")
        else:
            attris = json.load(open(args.attributes_input))
            attris_synset = json.load(open(args.attributes_synsets))
            json_struct, obj_attributes = generate_attributes(f, json_struct, attris, attris_synset, img_data,  num_attr=10, iou_thres=0.7)

            f.create_dataset('attributes', data=obj_attributes)

   # build train/val/test splits

    print(('num objects = %i' % encoded_label.shape[0]))
    print(('num relationships = %i' % encoded_predicate.shape[0]))

    if split is not None:
        f.create_dataset('split', data=split) # 1 = test, 0 = train

    if split_rel is not None:
        f.create_dataset('split_rel', data=split_rel) # 1 = test, 0 = train

    f.close()

    zero_shot = generate_zero_shot_triplets(im_to_first_rel, im_to_last_rel, encoded_rel, encoded_label, encoded_predicate, split_rel)
    
    zero_path = os.path.join(os.path.dirname(args.h5_file), 'zero_shot_triplets.pytorch')
    torch.save(zero_shot, zero_path)

    # print('Distribution of objects', object_token_counter[:100])
    # print('Distribution of predicates', predicate_token_counter[:100])

    with open(args.json_file, 'w') as f:
        json.dump(json_struct, f)

    return True

def encode_splits(obj_data, logger, opt=None):
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

    logger.info('assigned %d/%d/%d to train/val/test split' % (np.sum(split==0), np.sum(split==1), np.sum(split==2)))
    return split

def encode_splits_rel(rel_data, idx_img_rels, test_ratio, val_ratio, logger, shuffle=None):
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

    logger.info('Relations: assigned %d/%d/%d to train/val/test split' % (np.sum(split==0), np.sum(split==1), np.sum(split==2)))

    return split

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

def generate_zero_shot_triplets(img_to_firs_rel, img_to_last_rel, encoded_rel, encoded_label, encoded_predicate, split):
    train_triplets = set()
    test_triplets = set()
    split = split.tolist()
    idx_test = split.index(2)

    for i in tqdm(range(len(img_to_firs_rel))):
        eth_start = img_to_firs_rel[i]
        eth_end = img_to_last_rel[i]
        for j in range(eth_start, eth_end):
            rel = encoded_rel[j].tolist()
            predicate = int(encoded_predicate[j][0])
            subj = int(encoded_label[rel[0]])
            obj = int(encoded_label[rel[1]])
            if i < idx_test:
                train_triplets.add((subj, obj, predicate))
            else:
                test_triplets.add((subj, obj, predicate))
    print("Number of total triplets: {}".format(len(train_triplets)+len(test_triplets)))

    zrt = torch.tensor([list(x) for x in test_triplets if x not in train_triplets])
    print("Number of zero-shot triplets: {}".format(len(zrt)))
    return zrt

def encode_objects(obj_data, token_to_idx, token_counter, org_h, org_w, im_long_sizes, yolo_anno, img_data, img_list, logger):
    encoded_labels = []
    encoded_boxes  = {}
    for size in im_long_sizes:
        encoded_boxes[size] = []
    im_to_first_obj = np.zeros(len(obj_data), dtype=np.int32)
    im_to_last_obj = np.zeros(len(obj_data), dtype=np.int32)
    obj_counter = 0
    obj_distribution = {}
    new_obj_data = []

    for i, img in enumerate(tqdm(obj_data, desc='Encode objects')): 

        im_to_first_obj[i] = obj_counter
        img['id_to_idx'] = {}  # object id to region idx
        new_img = copy.deepcopy(img)
        new_img['objects'] = []
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
                new_obj = copy.deepcopy(obj)
                del new_obj['names']
                new_obj['name'] = obj_label
                new_img['objects'].append(new_obj)

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

        new_obj_data.append(new_img)

    for k, boxes in encoded_boxes.items():
        encoded_boxes[k] = np.vstack(boxes)

    # size = len([o for o in new_obj_data if len(o['objects']) > 0])
    # logger.info("Size of new object data {}".format(size))

    with open('IndoorVG_3/objects.json', 'w') as f:
        json.dump(new_obj_data, f)       

    logger.info('Distribution of objects per image: {}'.format(dict(sorted(obj_distribution.items(), key=lambda x: x[1]))))

    print('%i out of %i valid images have at least one object' % (sum(obj_distribution.values()), len(obj_data)))
    if yolo_anno == 'True':
        return np.vstack(encoded_labels), encoded_boxes, im_to_first_obj, im_to_last_obj, new_obj_data
    return np.vstack(encoded_labels), encoded_boxes, im_to_first_obj, im_to_last_obj, []

def generate_yolo_anno(obj_data, image_data, object_to_idx, dest_path):
    """
    generate yolo annotation file
    """
    output = os.path.join(dest_path, 'yolo_anno')
    if not os.path.exists(output):
        os.makedirs(output)
    for idx, img in enumerate(tqdm(obj_data, desc='Generate YOLO anno')):
        img_width = image_data[idx]['width']
        img_height = image_data[idx]['height']
        image_id = image_data[idx]['image_id']
        for obj in img['objects']:
            obj_name = obj['name']
            obj_x = obj['x']
            obj_y = obj['y']
            obj_width = obj['w']
            obj_height = obj['h']
            obj_cat = object_to_idx[obj_name]
            with open(os.path.join(output, str(image_id) + '.txt'), 'a') as f:
                xcenter = (obj_x + obj_width/2) / img_width
                ycenter = (obj_y + obj_height/2) / img_height
                w = obj_width / img_width
                h = obj_height / img_height
                obj_cat = obj_cat - 1
                f.write(str(obj_cat) + ' ' + str(xcenter) + ' ' + str(ycenter) + ' ' + str(w) + ' ' + str(h)+'\n')

def encode_relationship(sub_id, obj_id, id_to_idx):
    # builds a tuple of the index of object and subject in the object list
    sub_idx = id_to_idx[sub_id]
    obj_idx = id_to_idx[obj_id]
    return np.asarray([sub_idx, obj_idx], dtype=np.int32)

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
    new_rel_data = []

    for i, img in enumerate(tqdm(rel_data, desc='Encode relationships')):
        if len(img_list) > 0 and img['image_id'] not in img_list:
            # if image not in list, skip
            im_to_first_rel[i] = -1
            im_to_last_rel[i] = -1
            no_rel_counter += 1
            continue
        else:
            idx_img_rels.append(i)
            new_img = copy.deepcopy(img)
            new_img['relationships'] = []

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
                    new_img['relationships'].append(relation)
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
        new_rel_data.append(new_img)

    print('%i rel is filtered by object' % obj_filtered)
    print('%i rel is filtered by predicate' % predicate_filtered)
    print('%i rel is filtered by duplicate' % duplicate_filtered)
    print('%i rel is manually filtered using dict' % relations_filtered)
    #print('%i rel is filtered by number per image' % images_filtered)

    print('%i rel remains ' % len(encoded_pred))
    print('Distribution of relationships per image: ', dict(sorted(rel_distribution.items(), key=lambda x: x[1])))

    print('Average number of relations per image: ', len(encoded_pred)/(len(rel_data)-no_rel_counter))
    print('%i out of %i valid images have relationships' % (len(rel_data)-no_rel_counter, len(rel_data)))
    return np.vstack(encoded_pred), np.vstack(encoded_rel), im_to_first_rel, im_to_last_rel, idx_img_rels