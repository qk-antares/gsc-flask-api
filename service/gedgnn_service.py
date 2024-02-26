from model.gedgnn.src.param_parser import parameter_parser
from model.gedgnn.src.trainer import Trainer
from utils.convertor import *
import time

args = parameter_parser()
args.__setattr__("abs_path", "/root/workplace/python/gsc-flask-api/model/gedgnn/")
args.__setattr__("dataset", "AIDS")
args.__setattr__("model_epoch_start", 20)
args.__setattr__("model_epoch_end", 20)
args.__setattr__("model_train", 0)
args.__setattr__("model_name", "GedGNN")
args.__setattr__("model_path", "model_save/GEDGNN/")

trainer = Trainer(args)
trainer.load(args.model_epoch_start)
trainer.cur_epoch = args.model_epoch_start
trainer.model.eval()


def pre_ged(g1, g2):
    if g1['n'] > g2['n']:
        g1, g2 = g2, g1
    data = trainer.get_new_data(g1, g2)
    _, pre_ged, _ = trainer.model(data)
    return pre_ged


def compute_pre_geds(graph1, graph2s):
    g1 = convert_graph_data(graph1)
    g2s = [convert_graph_data(graph2) for graph2 in graph2s]
    return [pre_ged(g1, g2) for g2 in g2s]


def compute_pre_ged(graph1, graph2):
    g1 = convert_graph_data(graph1)
    g2 = convert_graph_data(graph2)
    return pre_ged(g1, g2)


def compute_paths_with_k(graph1, graph2, k):
    g1 = convert_graph_data(graph1)
    g2 = convert_graph_data(graph2)

    # 计算两个图之间的编辑距离
    start_time = time.time()

    if g1['n'] > g2['n']:
        g1, g2 = g2, g1
        graph1, graph2 = graph2, graph1

    data = trainer.get_new_data(g1, g2)

    model_out = trainer.test_matching(data, test_k=k)
    print(model_out)
    pre_ged = model_out[1]
    pre_permute = model_out[2]
    edit_nodes = get_pre_edit_nodes(g1, g2, graph1, graph2, pre_permute)
    edit_edges = get_pre_edit_edges(g1, g2, graph1, graph2, pre_permute)
    paths = {
        "nodes": edit_nodes,
        "edges": edit_edges
    }

    end_time = time.time()
    total_time = end_time - start_time

    return pre_ged, paths, total_time


def get_pre_edit_nodes(g1, g2, graph1, graph2, permute):
    # 统计节点替换
    edit_nodes = []
    for n_l, n_r in enumerate(permute):
        edit_nodes.append([graph1["nodes"][n_l]["id"], graph2["nodes"][n_r]["id"]])

    # 统计没有被映射的g2中的节点
    add_nodes = [node for node in range(0, g2["n"]) if node not in permute]

    return edit_nodes + [[None, graph2["nodes"][add]["id"]] for add in add_nodes]


def get_pre_edit_edges(g1, g2, graph1, graph2, permute):
    edges1 = g1['graph']
    edges2 = g2['graph']
    # 经过节点映射的edges1
    mapping_edges1 = [[permute[source], permute[target]] for source, target in g1['graph']]
    # edges排序
    sorted_edges1 = [[source, target] if source <= target else [target, source] for source, target in mapping_edges1]
    sorted_edges2 = [[source, target] if source <= target else [target, source] for source, target in edges2]

    edit_edges = []
    for index, edge in enumerate(sorted_edges2):
        if edge not in sorted_edges1:
            add_edge = [graph2["nodes"][edges2[index][0]]["id"],
                        graph2["nodes"][edges2[index][1]]["id"]]
            edit_edges.append([None, add_edge])

    for index, edge in enumerate(sorted_edges1):
        origin_edge = [graph1["nodes"][edges1[index][0]]["id"],
                       graph1["nodes"][edges1[index][1]]["id"]]
        if edge not in sorted_edges2:
            edit_edges.append([origin_edge, None])
        else:
            replace_edge = [graph2["nodes"][mapping_edges1[index][0]]["id"],
                            graph2["nodes"][mapping_edges1[index][1]]["id"]]
            edit_edges.append([origin_edge, replace_edge])

    return edit_edges