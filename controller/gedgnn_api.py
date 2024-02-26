from flask import Blueprint, request, jsonify
from service.gedgnn_service import *

gedgnn_api = Blueprint('gedgnn_api', __name__)


@gedgnn_api.route('/gedgnn/value_batch', methods=['POST'])
def gedgnn_value_batch():
    # 获取POST请求中的JSON数据
    data = request.get_json()
    graph1 = data.get('graph1', {})
    graph2s = data.get('graph2s', [])

    pre_geds = compute_pre_geds(graph1, graph2s)
    return jsonify({
        'pre_geds': pre_geds
    })


@gedgnn_api.route('/gedgnn/value', methods=['POST'])
def gedgnn_value():
    # 获取POST请求中的JSON数据
    data = request.get_json()
    graph1 = data.get('graph1', {})
    graph2 = data.get('graph2', {})
    pre_ged = compute_pre_ged(graph1, graph2)
    return jsonify({
        'pre_ged': pre_ged
    })


@gedgnn_api.route('/gedgnn/paths', methods=['POST'])
def gedgnn_paths():
    # 获取POST请求中的JSON数据
    data = request.get_json()
    graph_pair = data.get('graphPair', {})
    k = data.get('k')
    graph1 = graph_pair['graph1']
    graph2 = graph_pair['graph2']
    pre_ged, paths, time_use = compute_paths_with_k(graph1, graph2, k)
    return jsonify({
        'pre_ged': pre_ged,
        "paths": paths,
        "time_use": time_use
    })
