import json
from tqdm import tqdm
import numpy as np
import networkx as nx
from collections import Counter
import os
from .curation import get_label


def find_most_connected_objects(rel_data, obj_data, logger, object_list, predicate_list, num_objects=150, num_predicates=50, img_list=[], use_negative_lists=False, input_dir=""):
    G, node_types, edge_types = build_graphs(rel_data, obj_data, use_negative_lists, img_list, input_dir)

    logger.debug("Number of nodes: {}".format(G.number_of_nodes()))
    logger.debug("Number of edges: {}".format(G.number_of_edges()))

    node_type_set = set(node_types.values())

    node_type_to_idx = {node_type: idx for idx, node_type in enumerate(node_type_set)}

    n = len(node_type_set)

    biadjacency_matrix = np.zeros((n, n))
    for node1, node2, data in tqdm(G.edges(data=True)):
        type1 = node_types[node1]
        type2 = node_types[node2]
        i = node_type_to_idx[type1]
        j = node_type_to_idx[type2]
        biadjacency_matrix[i][j] += data.get('weight', 1)
        biadjacency_matrix[j][i] += data.get('weight', 1)
    type_counts = biadjacency_matrix.sum(axis=0)
    node_type_counts = [(node_type, count) for node_type, count in zip(node_type_set, type_counts)]
    node_type_counts.sort(key=lambda x: x[1], reverse=True)
    if object_list != []:
        most_connected_node_types = {node_type: int(count) for node_type, count in node_type_counts if node_type in object_list}
    else:
        most_connected_node_types = {node_type: int(count) for node_type, count in node_type_counts[:num_objects]}

    # remove all nodes that are not in the most connected node types from G
    nodes_to_remove = []
    for node in G.nodes():
        if node_types[node] not in most_connected_node_types:
            nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)

    # select the n_slected_edge_types more present edge types in this new set
    edge_type_set = set(edge_types.values())
    edge_type_to_idx = {edge_type: idx for idx, edge_type in enumerate(edge_type_set)}
    n = len(edge_type_set)

    edge_type_counts = np.zeros(n)
    for node1, node2, data in tqdm(G.edges(data=True), desc='Counting edge types'):
        edge_type = edge_types[(node1, node2)]
        i = edge_type_to_idx[edge_type]
        edge_type_counts[i] += data.get('weight', 1)
    edge_type_counts = [(edge_type, count) for edge_type, count in zip(edge_type_set, edge_type_counts)]
    edge_type_counts.sort(key=lambda x: x[1], reverse=True)
    if predicate_list != []:
        most_connected_edge_types = {edge_type: int(count) for edge_type, count in edge_type_counts if edge_type in predicate_list}
    else:
        most_connected_edge_types = {edge_type: int(count) for edge_type, count in edge_type_counts[:num_predicates]}

    return most_connected_node_types, most_connected_edge_types

def build_graphs(rel_data, obj_data, use_neg_list, img_list=[], input_dir=""):
    # Create a multi-graph where each node type is a node and each edge represents a connection
    obj_neg_list = []
    pred_neg_list = []
    if use_neg_list:
        my_file = open(os.path.join(input_dir, "object_negative_list.txt"), "r")
        obj_neg_list = [line.replace('\n', '') for line in my_file.readlines()]

        my_file = open(os.path.join(input_dir, "predicate_negative_list.txt"), "r")
        pred_neg_list = [line.replace('\n', '') for line in my_file.readlines()]

    graph = nx.MultiDiGraph()
    node_types = {}
    edge_types = {}
    
    for idx, img in enumerate(tqdm(rel_data, desc='Building graphs')):
        if img_list == [] or img['image_id'] in img_list:
            for relation in img['relationships']:
                subj = relation['subject']

                obj = relation['object']
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
                if obj_label in obj_neg_list or subj_label in obj_neg_list:
                    continue
                if relation['predicate'] in pred_neg_list:
                    continue
                graph.add_edge(subj['object_id'], obj['object_id'], type=relation['predicate'], weight=1)
                node_types[subj['object_id']] = subj_label
                node_types[obj['object_id']] = obj_label
                edge_types[(subj['object_id'], obj['object_id'])] = relation['predicate']
    return graph, node_types, edge_types

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
    rel_distri = Counter()
    token_counter = Counter()
    for img in data:
        for region in img['objects']:
            for name in region['names']:
                if name not in negative_list:
                    if not obj_list or name in obj_list:
                        if len(rel_data) == 0:
                            token_counter.update([name])
                            rel_distri.update([name])
                        elif region['object_id'] in rel_counter:
                            token_counter.update([name]*len(rel_counter[region['object_id']]))
                            rel_distri.update([name])
    # pick top N tokens
    token_counter_return = {}
    if not obj_list and num_tokens==0:
        for token, count in token_counter.most_common():
            token_counter_return[token] = rel_distri[token]
    elif not obj_list:
        for token, count in token_counter.most_common():
            token_counter_return[token] = rel_distri[token]
            if len(token_counter_return.keys()) == num_tokens:
                break
    else:
        for token, count in token_counter.most_common():
            if token in obj_list:
                token_counter_return[token] = rel_distri[token]
            
    if verbose:
        print(('Keeping %d / %d objects'
                  % (len(token_counter_return.keys()), len(token_counter))))
    return token_counter_return

def extract_object_token_no_rel(data, num_tokens, obj_list=[],  img_list=[], verbose=True):
    """ Builds a set that contains the object names. Filters infrequent tokens. """

    print("Filtering object by number of occurences")
    token_counter = Counter()
    my_file = open("VG/object_negative_list.txt", "r")
    negative_list = [line.replace('\n', '') for line in my_file.readlines()]

    for img in data:
        if not img_list or img['image_id'] in img_list:
            for region in img['objects']:
                for name in region['names']:
                    if name not in negative_list:
                        if not obj_list or name in obj_list:
                            token_counter.update([name])
    # pick top N tokens
    token_counter_return = {}
    if not obj_list and num_tokens==0:
        for token, count in token_counter.most_common():
            token_counter_return[token] = count
    elif not obj_list:
        for token, count in token_counter.most_common():
            token_counter_return[token] = count
            if len(token_counter_return.keys()) == num_tokens:
                break
    else:
        for token, count in token_counter.most_common():
            if token in obj_list:
                token_counter_return[token] = count
        
        
    # remove objects with less than 10 occurences
    token_counter_return = {}
    for token, count in token_counter.most_common():
        if count > 10:
            token_counter_return[token] = count

    if verbose:
        print(('Keeping %d / %d objects'
                  % (len(token_counter_return.keys()), len(token_counter))))

    return token_counter_return


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

    token_counter_return = {}

    if not pred_list and num_tokens==0:
        for token, count in token_counter.most_common():
            if count > 5:
                token_counter_return[token] = count
    elif not pred_list:
        for token, count in token_counter.most_common():
            if count > 5:
                token_counter_return[token] = count
                if len(token_counter_return.keys()) == num_tokens:
                    break
    else:
        for token, count in token_counter.most_common():
            if token in pred_list:
                token_counter_return[token] = count

    if verbose:
        print(('Keeping %d / %d predicates'
                  % (len(token_counter_return.keys()), len(token_counter))))
        
    return token_counter_return