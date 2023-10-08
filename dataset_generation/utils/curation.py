import networkx as nx
import pandas as pd
import os
from tqdm import tqdm

def get_label(obj):
    if 'names' in obj.keys():
        return obj['names'][0]
    else:
        return obj['name']
    
def check_part_whole2(idx, img, obj_data, part_whole_df):
    part_whole_count = 0
    values = part_whole_df['sentence'].to_list()
    rels = []
    G = nx.MultiDiGraph()
    init_size = len(img['relationships'])
    i = 0
    # construct weighted graph with part-whole edges = 0
    for relation in img['relationships']:
        subj = relation['subject']

        obj = relation['object']
        predicate = relation['predicate'].lower()

        subj_id = relation['subject']['object_id']
        obj_id = relation['object']['object_id']

        for obj in obj_data[idx]['objects']:
            if obj['object_id'] == subj_id:
                subj = obj
                break
            if obj['object_id'] == obj_id:
                obj = obj
                break
        obj_label = get_label(obj)
        subj_label = get_label(subj)

        full_rel = str(subj_label + ' ' + predicate + ' ' + obj_label)
        rels.append(full_rel)
        
        if full_rel in values:
            part_whole_count += 1
            G.add_edge(subj['object_id'], obj['object_id'], weight=1)
        else:
            G.add_edge(subj['object_id'], obj['object_id'], weight=0)

    break_out = False

    to_remove = []
    edges = [e for e in G.edges(data=True)]
    for e in edges:
        if e[2]['weight'] == 1:
            if G.out_degree(e[1]) == 0:
                if all([w[2]['weight'] == 0 for w in G.edges(data=True)]):
                    to_remove.append((e[0], e[1]))
    # leaves = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]

    # for l in leaves:
    #     weight = [e for e in G.in_edges(l,data=True)]
    #     if all([w[2]['weight'] == 0 for w in weight]):
    #         to_remove.append((weight[0][0], weight[0][1]))
    G.remove_edges_from(to_remove)

        # if len(to_remove) == 0:
        #     break_out = True

    new_rel = []
    for relation in img['relationships']:
        if (relation['subject']['object_id'], relation['object']['object_id']) in G.edges():
            new_rel.append(relation)
    number_filtered = init_size - len(new_rel)

    return number_filtered, new_rel, part_whole_count

def check_part_whole3(img, part_whole_df):

    values = part_whole_df['sentence'].to_list()

    G = nx.MultiDiGraph()
    init_size = len(img['relationships'])
    for relation in img['relationships']:
        subj = relation['subject']

        obj = relation['object']
        predicate = relation['predicate'].lower()
        
        obj_label = get_label(obj)
        subj_label = get_label(subj)

        full_rel = str(subj_label + ' ' + predicate + ' ' + obj_label)

        if full_rel in values:
            G.add_edge(subj['object_id'], obj['object_id'], weight=0, label=predicate)
        else:
            G.add_edge(subj['object_id'], obj['object_id'], weight=1, label=predicate)

    break_out = False

    while not break_out:
        to_remove = []
        leaves = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]

        for l in leaves:
            weight = [e for e in G.in_edges(l,data=True)]
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

def check_part_whole(idx, img, obj_data, part_whole_df):
    values = part_whole_df['sentence'].to_list()
    output = []
    init_size = len(img['relationships'])
    for relation in img['relationships']:
        subj = relation['subject']

        obj = relation['object']
        predicate = relation['predicate'].lower()

        subj_id = relation['subject']['object_id']
        obj_id = relation['object']['object_id']

        for obj in obj_data[idx]['objects']:
            if obj['object_id'] == subj_id:
                subj = obj
                break
            if obj['object_id'] == obj_id:
                obj = obj
                break
        obj_label = get_label(obj)
        subj_label = get_label(subj)

        predicate = relation['predicate'].lower()

        full_rel = str(subj_label + ' ' + predicate + ' ' + obj_label)

        if full_rel in values:
            continue
        else:
            output.append(relation)
    
    number_filtered = init_size - len(output)
    return number_filtered, output

def filter_part_whole(rel_data, obj_data, logger, part_whole_path=''):
    if part_whole_path is not None:
        # check that file exist
        if not os.path.isfile(part_whole_path):
            logger.info("Wrong part-whole file {}, skipping part whole filtering ".format(part_whole_path))
            return rel_data

    logger.info('Filtering part-whole relationships')

    part_whole_df = pd.read_csv(part_whole_path, sep=',')
    part_whole_filtered = 0
    for idx, img in enumerate(tqdm(rel_data)):
        # filter part-whole relationships using dict
        relations_filtered, img['relationships'] = check_part_whole(idx, img, obj_data, part_whole_df) #, obj_data, 
        part_whole_filtered += relations_filtered

    logger.info('{} rel is filtered by part-whole'.format(part_whole_filtered))

    return rel_data