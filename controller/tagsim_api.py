from flask import Blueprint, request, jsonify
from service.tagsim_service import *

tagsim_api = Blueprint('tagsim_api', __name__)


@tagsim_api.route('/tagsim/value', methods=['POST'])
def tagsim_value():
    # 获取POST请求中的JSON数据
    data = request.get_json()
    graph1 = data.get('graph1', {})
    graph2 = data.get('graph2', {})
    pre_ged = compute_pre_ged(graph1, graph2)
    return jsonify({
        'pre_ged': pre_ged
    })


@tagsim_api.route('/tagsim/value_batch', methods=['POST'])
def simgnn_value_batch():
    # 获取POST请求中的JSON数据
    data = request.get_json()
    graph1 = data.get('graph1', {})
    graph2s = data.get('graph2s', [])

    pre_geds = compute_pre_geds(graph1, graph2s)
    return jsonify({
        'pre_geds': pre_geds
    })
