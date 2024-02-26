from model.gedgnn.src.param_parser import parameter_parser
from model.gedgnn.src.trainer import Trainer
from utils.convertor import *

args = parameter_parser()
args.__setattr__("abs_path", "E:/Workplace/graph-compute-backend-flask/model/gedgnn/")
args.__setattr__("dataset", "AIDS")
args.__setattr__("model_epoch_start", 20)
args.__setattr__("model_epoch_end", 20)
args.__setattr__("model_train", 0)
args.__setattr__("model_name", "TaGSim")
args.__setattr__("model_path", "model_save/TaGSim/")

trainer = Trainer(args)
trainer.load(args.model_epoch_start)
trainer.cur_epoch = args.model_epoch_start
trainer.model.eval()


def pre_ged(g1, g2):
    if g1['n'] > g2['n']:
        g1, g2 = g2, g1
    data = trainer.get_new_data(g1, g2)
    score, pre_ged = trainer.model(data)
    return pre_ged


def compute_pre_geds(graph1, graph2s):
    g1 = convert_graph_data(graph1)
    g2s = [convert_graph_data(graph2) for graph2 in graph2s]
    return [pre_ged(g1, g2) for g2 in g2s]


def compute_pre_ged(graph1, graph2):
    g1 = convert_graph_data(graph1)
    g2 = convert_graph_data(graph2)
    return pre_ged(g1, g2)

