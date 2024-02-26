import json
from glob import glob
from os.path import basename

import networkx as nx
from texttable import Texttable


def one_hot_encode_labels(graph1, graph2):
    """
    将两个图的labels属性转成one-hot编码（需要保证graph1和graph2的labels类型总和不超过模型的处理范围）
    :param graph1:
    :param graph2:
    :return:
    """
    # 合并graph1和graph2的标签
    combined_labels = []
    # 创建一个集合用于跟踪已经添加的元素
    added_labels = set()
    # 分别遍历g1和g2
    for label in graph1["labels"]:
        if label not in added_labels:
            combined_labels.append(label)
            added_labels.add(label)
    for label in graph2["labels"]:
        if label not in added_labels:
            combined_labels.append(label)
            added_labels.add(label)

    # 创建标签到索引的映射
    label_to_index = {label: i for i, label in enumerate(combined_labels)}

    graph1_one_hot = [[0] * 29 for _ in range(len(graph1['labels']))]
    graph2_one_hot = [[0] * 29 for _ in range(len(graph2['labels']))]

    # 对graph1和graph2进行one-hot编码
    for i, label in enumerate(graph1['labels']):
        index = label_to_index[label]
        graph1_one_hot[i][index] = 1
    for i, label in enumerate(graph2['labels']):
        index = label_to_index[label]
        graph2_one_hot[i][index] = 1

    return graph1_one_hot, graph2_one_hot


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    rows = [["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    t.add_rows(rows)
    print(t.draw())


def sorted_nicely(l):
    """
    Sort file names in a fancy way.
    The numbers in file names are extracted and converted from str into int first,
    so file names can be sorted based on int comparison.
    :param l: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)


def get_file_paths(dir, file_format='json'):
    """
    Return all file paths with file_format under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths


def iterate_get_graphs(dir, file_format):
    """
    Read networkx (dict) graphs from all .gexf (.json) files under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ['gexf', 'json', 'onehot', 'anchor']
    graphs = []
    for file in get_file_paths(dir, file_format):
        gid = int(basename(file).split('.')[0])
        if file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            # g is a dict
            g = json.load(open(file, 'r'))
            g['gid'] = gid
        elif file_format in ['onehot', 'anchor']:
            # g is a list of onehot labels
            g = json.load(open(file, 'r'))
        graphs.append(g)
    return graphs


def load_labels(data_location, dataset_name):
    path = data_location + "json_data/" + dataset_name + "/labels.json"
    global_labels = json.load(open(path, 'r'))
    return global_labels


def load_ged(ged_dict, data_location='', dataset_name='AIDS', file_name='TaGED.json'):
    '''
    list(tuple)
    ged = [(id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, [best_node_mapping])]

    id_1 and id_2 are the IDs of a graph pair, e.g., the ID of 4.json is 4.
    The given graph pairs satisfy that n1 <= n2.

    ged_value = ged_nc + ged_in + ged_ie
    (ged_nc, ged_in, ged_ie) is the type-aware ged following the setting of TaGSim.
    ged_nc: the number of node relabeling
    ged_in: the number of node insertions/deletions
    ged_ie: the number of edge insertions/deletions

    [best_node_mapping] contains 10 best matching at most.
    best_node_mapping is a list of length n1: u in g1 -> best_node_mapping[u] in g2

    return dict()
    ged_dict[(id_1, id_2)] = ((ged_value, ged_nc, ged_in, ged_ie), best_node_mapping_list)
    '''
    path = "{}json_data/{}/{}".format(data_location, dataset_name, file_name)
    TaGED = json.load(open(path, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
        ged_dict[(id_1, id_2)] = (ta_ged, mappings)


def load_features(data_location, dataset_name, feature_name):
    features = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", feature_name) \
               + iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", feature_name)
    feature_dim = len(features[0][0])
    print('Load {} features (dim = {}) of {}.'.format(feature_name, feature_dim, dataset_name))
    return feature_dim, features
