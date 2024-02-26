from param_parser import parameter_parser
from trainer import Trainer
from utils import tab_printer


def convert_graph_data(data):
    nodes = data["nodes"]
    edges = data["edges"]

    n = len(nodes)
    m = len(edges)

    # 获取节点标签列表
    labels = [node["label"] for node in nodes]
    # 创建一个id到索引的映射
    id_to_index = {node['id']: index for index, node in enumerate(nodes)}

    graph = []
    for edge in edges:
        source = id_to_index[edge['source']]
        target = id_to_index[edge['target']]
        graph.append([source, target])

    result = {"n": n, "m": m, "labels": labels, "graph": graph}
    return result

def main():
    args = parameter_parser()
    args.__setattr__("abs_path", "E:/Workplace/graph-compute-backend-flask/model/gedgnn/")
    args.__setattr__("dataset", "AIDS")
    args.__setattr__("model_epoch_start", 20)
    args.__setattr__("model_epoch_end", 20)
    args.__setattr__("model_train", 0)

    tab_printer(args)
    trainer = Trainer(args)
    trainer.load(args.model_epoch_start)
    trainer.cur_epoch = args.model_epoch_start
    trainer.model.eval()

    # 39
    graph1 = {
        "nodes": [
            {
                "id": "1",
                "label": "N"
            },
            {
                "id": "2",
                "label": "C"
            },
            {
                "id": "3",
                "label": "C"
            },
            {
                "id": "4",
                "label": "C"
            },
            {
                "id": "5",
                "label": "C"
            },
            {
                "id": "6",
                "label": "C"
            },
            {
                "id": "7",
                "label": "Cl"
            },
            {
                "id": "8",
                "label": "Cl"
            }
        ],
        "edges": [
            {
                "source": "8",
                "target": "6"
            },
            {
                "source": "4",
                "target": "1"
            },
            {
                "source": "6",
                "target": "3"
            },
            {
                "source": "7",
                "target": "5"
            },
            {
                "source": "2",
                "target": "1"
            },
            {
                "source": "2",
                "target": "5"
            },
            {
                "source": "1",
                "target": "3"
            }
        ]
    }
    # 37
    graph2 = {
        "nodes": [
            {
                "id": "1",
                "label": "S"
            },
            {
                "id": "2",
                "label": "C"
            },
            {
                "id": "3",
                "label": "C"
            },
            {
                "id": "4",
                "label": "N"
            },
            {
                "id": "5",
                "label": "C"
            },
            {
                "id": "6",
                "label": "N"
            },
            {
                "id": "7",
                "label": "C"
            },
            {
                "id": "8",
                "label": "N"
            },
            {
                "id": "9",
                "label": "N"
            },
            {
                "id": "10",
                "label": "C"
            }
        ],
        "edges": [
            {
                "source": "8",
                "target": "7"
            },
            {
                "source": "8",
                "target": "5"
            },
            {
                "source": "4",
                "target": "2"
            },
            {
                "source": "4",
                "target": "7"
            },
            {
                "source": "6",
                "target": "3"
            },
            {
                "source": "6",
                "target": "10"
            },
            {
                "source": "9",
                "target": "5"
            },
            {
                "source": "9",
                "target": "10"
            },
            {
                "source": "2",
                "target": "1"
            },
            {
                "source": "2",
                "target": "3"
            },
            {
                "source": "5",
                "target": "3"
            }
        ]
    }

    g1 = convert_graph_data(graph1)
    g2 = convert_graph_data(graph2)

    data = trainer.get_new_data(g1, g2)

    model_out = trainer.test_matching(data, test_k=100)
    print(model_out)
    pre_permute = model_out[2]
    edit_nodes = get_pre_edit_nodes(g1, g2, graph1, graph2, pre_permute)
    edit_edges = get_pre_edit_edges(g1, g2, pre_permute)


def get_pre_edit_nodes(g1, g2, graph1, graph2, permute):
    # 统计节点替换
    edit_nodes = []
    for n_l, n_r in enumerate(permute):
        edit_nodes.append([graph1["nodes"][n_l]["id"], graph2["nodes"][n_r]["id"]])

    # 统计没有被映射的g2中的节点
    add_nodes = [node for node in range(0, g2["n"]) if node not in permute]

    return edit_nodes + [[None, graph2["nodes"][add]["id"]] for add in add_nodes]


def get_pre_edit_edges(g1, g2, permute):
    edges1 = g1['graph']
    edges2 = g2['graph']
    # 经过节点映射的edges1
    mapping_edges1 = [[permute[source], permute[target]] for source, target in g1['graph']]
    # edges排序
    sorted_edges1 = [[source, target] if source <= target else [target, source] for source, target in mapping_edges1]
    sorted_edges2 = [[source, target] if source <= target else [target, source] for source, target in edges2]

    edit_edges = []
    for index, edge in enumerate(sorted_edges1):
        if edge not in sorted_edges2:
            edit_edges.append([edges1[index], None])
        else:
            edit_edges.append([edges1[index], mapping_edges1[index]])

    for index, edge in enumerate(sorted_edges2):
        if edge not in sorted_edges1:
            edit_edges.append([None, edges2[index]])

    return edit_edges


if __name__ == "__main__":
    main()



