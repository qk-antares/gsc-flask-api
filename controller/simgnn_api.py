from flask import Blueprint, request, jsonify
from service.simgnn_service import *

simgnn_api = Blueprint('simgnn_api', __name__)


@simgnn_api.route('/simgnn/value', methods=['POST'])
def simgnn_value():
    # 获取POST请求中的JSON数据
    data = request.get_json()
    graph1 = data.get('graph1', {})
    graph2 = data.get('graph2', {})
    pre_ged = compute_pre_ged(graph1, graph2)
    return jsonify({
        'pre_ged': pre_ged
    })


@simgnn_api.route('/simgnn/value_batch', methods=['POST'])
def simgnn_value_batch():
    # 获取POST请求中的JSON数据
    data = request.get_json()
    graph1 = data.get('graph1', {})
    graph2s = data.get('graph2s', [])

    pre_geds = compute_pre_geds(graph1, graph2s)
    return jsonify({
        'pre_geds': pre_geds
    })
