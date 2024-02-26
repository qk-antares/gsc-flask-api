from flask import Blueprint, request, jsonify
from service.ged_service import *

ged_api = Blueprint('ged_api', __name__)


@ged_api.route('/accurate/paths', methods=['POST'])
def get_accurate_paths():
    # 获取POST请求中的JSON数据
    data = request.get_json()
    graph_pair = data.get('graphPair', {})
    graph1 = graph_pair['graph1']
    graph2 = graph_pair['graph2']
    ged, paths, time_use = compute_ged(graph1, graph2)
    return jsonify({
        'pre_ged': ged,
        "paths": paths,
        "time_use": time_use
    })
