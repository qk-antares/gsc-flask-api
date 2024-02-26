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