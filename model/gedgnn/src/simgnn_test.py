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
    args.__setattr__("model_name", "SimGNN")
    args.__setattr__("model_path", "model_save/SimGNN/")

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
    # 985
    graph2 = {
        "nodes": [
            {
                "id": "1",
                "label": "O"
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
                "label": "O"
            },
            {
                "id": "5",
                "label": "N"
            },
            {
                "id": "6",
                "label": "C"
            },
            {
                "id": "7",
                "label": "N"
            },
            {
                "id": "8",
                "label": "C"
            },
            {
                "id": "9",
                "label": "S"
            }
        ],
        "edges": [
            {
                "source": "8",
                "target": "9"
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
                "target": "9"
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
                "target": "3"
            },
            {
                "source": "5",
                "target": "3"
            }
        ]
    }

    g2 = convert_graph_data(graph1)
    g1 = convert_graph_data(graph2)

    data = trainer.get_new_data(g1, g2)

    model_out = trainer.model(data)
    print(model_out)


if __name__ == "__main__":
    main()



