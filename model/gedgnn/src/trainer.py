# 基于图匹配的图编辑路径计算模型

import random
import sys
import time

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

from model.gedgnn.src.greedy_algo import hungarian
from model.gedgnn.src.kbest_matching_with_lb import KBestMSolver
from model.gedgnn.src.models import GPN, SimGNN, GedGNN, TaGSim
from model.gedgnn.src.utils import load_labels, one_hot_encode_labels


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.use_gpu = False
        self.device = torch.device('cpu')
        self.load_global_labels()
        self.setup_model()

    def load_global_labels(self):
        self.global_labels = load_labels(self.args.abs_path, self.args.dataset)
        self.number_of_labels = len(self.global_labels)

    def setup_model(self):
        if self.args.model_name == 'GPN':
            self.model = GPN(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "SimGNN":
            self.args.filters_1 = 64
            self.args.filters_2 = 32
            self.args.filters_3 = 16
            self.args.histogram = True
            self.args.target_mode = 'exp'
            self.model = SimGNN(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "GedGNN":
            if self.args.dataset in ["AIDS", "Linux"]:
                self.args.loss_weight = 10.0
            else:
                self.args.loss_weight = 1.0
            self.args.gtmap = True
            self.model = GedGNN(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "TaGSim":
            self.args.target_mode = 'exp'
            self.model = TaGSim(self.args, self.number_of_labels).to(self.device)
        else:
            assert False

    def get_new_data(self, g1, g2):
        new_data = dict()
        feature1, feature2 = one_hot_encode_labels(g1, g2)
        new_data["features_1"] = torch.tensor(feature1).float().to(self.device)
        new_data["features_2"] = torch.tensor(feature2).float().to(self.device)
        new_data["edge_index_1"] = self.get_edge_index(g1)
        new_data["edge_index_2"] = self.get_edge_index(g2)

        n1, n2, m1, m2 = g1['n'], g2['n'], g1['m'], g2['m']
        new_data["n1"] = n1
        new_data["n2"] = n2
        new_data["hb"] = max(n1, n2) + max(m1, m2)
        new_data["avg_v"] = n1 + n2 / 2
        return new_data

    def get_edge_index(self, g):
        edge = g['graph']
        edge = edge + [[y, x] for x, y in edge]
        edge = edge + [[x, x] for x in range(g['n'])]
        edge = torch.tensor(edge).t().long().to(self.device)
        return edge

    @staticmethod
    def delta_graph(g, f, device):
        new_data = dict()

        n = g['n']
        permute = list(range(n))
        random.shuffle(permute)
        mapping = torch.sparse_coo_tensor((list(range(n)), permute), [1.0] * n, (n, n)).to_dense().to(device)

        edge = g['graph']
        edge_set = set()
        for x, y in edge:
            edge_set.add((x, y))
            edge_set.add((y, x))

        random.shuffle(edge)
        m = len(edge)
        ged = random.randint(1, 5) if n <= 20 else random.randint(1, 10)
        del_num = min(m, random.randint(0, ged))
        edge = edge[:(m - del_num)]  # the last del_num edges in edge are removed
        add_num = ged - del_num
        if (add_num + m) * 2 > n * (n - 1):
            add_num = n * (n - 1) // 2 - m
        cnt = 0
        while cnt < add_num:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if (x != y) and (x, y) not in edge_set:
                edge_set.add((x, y))
                edge_set.add((y, x))
                cnt += 1
                edge.append([x, y])
        assert len(edge) == m - del_num + add_num
        new_data["n"] = n
        new_data["m"] = len(edge)

        new_edge = [[permute[x], permute[y]] for x, y in edge]
        new_edge = new_edge + [[y, x] for x, y in new_edge]  # add reverse edges
        new_edge = new_edge + [[x, x] for x in range(n)]  # add self-loops

        new_edge = torch.tensor(new_edge).t().long().to(device)

        feature2 = torch.zeros(f.shape).to(device)
        for x, y in enumerate(permute):
            feature2[y] = f[x]

        new_data["permute"] = permute
        new_data["mapping"] = mapping
        ged = del_num + add_num
        new_data["ta_ged"] = (ged, 0, 0, ged)
        new_data["edge_index"] = new_edge
        new_data["features"] = feature2
        return new_data

    @staticmethod
    def cal_pk(num, pre, gt):
        if num >= len(pre):
            return -1.0
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num

    @staticmethod
    def gen_edit_path(data, permute):
        n1, n2 = data["n1"], data["n2"]
        raw_edges_1, raw_edges_2 = data["edge_index_1"].t().tolist(), data["edge_index_2"].t().tolist()
        raw_f1, raw_f2 = data["features_1"].tolist(), data["features_2"].tolist()
        assert len(permute) == n1
        assert len(raw_f1) == n1 and len(raw_f2) == n2 and len(raw_f1[0]) == len(raw_f2[0])

        edges_1 = set()
        for (u, v) in raw_edges_1:
            pu, pv = permute[u], permute[v]
            if pu <= pv:
                edges_1.add((pu, pv))

        edges_2 = set()
        for (u, v) in raw_edges_2:
            if u <= v:
                edges_2.add((u, v))

        edit_edges = edges_1 ^ edges_2

        f1 = []
        num_label = len(raw_f1[0])
        for f in raw_f1:
            for j in range(num_label):
                if f[j] > 0:
                    f1.append(j)
                    break
        f2 = []
        for f in raw_f2:
            for j in range(num_label):
                if f[j] > 0:
                    f2.append(j)
                    break

        relabel_nodes = set()
        for (u, v) in enumerate(permute):
            if f1[u] != f2[v]:
                relabel_nodes.add((v, f1[u]))

        return edit_edges, relabel_nodes

    def path_score(self, testing_graph_set='test', test_k=None):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()

        num = 0  # total testing number
        time_usage = []
        mae = []  # ged mae

        rate = []  # node matching rate
        recall = []  # path recall
        precision = []  # path precision
        f1 = []  # path f1 score
        sim = []  # path similarity

        for pair_type, i, j_list in tqdm(testing_graphs[10:], file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                if gt_ged == 0:
                    continue
                if test_k is None:
                    model_out = self.model(data)
                    prediction, pre_ged = model_out[0], model_out[1]
                elif test_k == 0:
                    model_out = self.test_noah(data)
                    pre_permute = model_out[2]
                    pre_edit_edges, pre_relabel_nodes = self.gen_edit_path(data, pre_permute)
                    prediction, pre_ged = model_out[0], model_out[1]
                    pre_ged = len(pre_edit_edges) + len(pre_relabel_nodes)
                elif test_k > 0:
                    model_out = self.test_matching(data, test_k)
                    pre_permute = model_out[2]
                    pre_edit_edges, pre_relabel_nodes = self.gen_edit_path(data, pre_permute)
                    prediction, pre_ged = model_out[0], model_out[1]
                else:
                    assert False

                round_pre_ged = round(pre_ged)

                num += 1
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae.append(abs(round_pre_ged - gt_ged))

                assert len(pre_edit_edges) + len(pre_relabel_nodes) == round_pre_ged

                best_rate = 0.
                best_recall = 0.
                best_precision = 0.
                best_f1 = 0.
                best_sim = 0.

                # enumerate groundtruth path
                for permute in data["permute"]:
                    tmp = 0
                    for (v1, v2) in zip(permute, pre_permute):
                        if v1 == v2:
                            tmp += 1
                    best_rate = max(best_rate, tmp / data["n1"])

                    edit_edges, relabel_nodes = self.gen_edit_path(data, permute)
                    assert len(edit_edges) + len(relabel_nodes) == gt_ged
                    num_overlap = len(pre_edit_edges & edit_edges) + len(pre_relabel_nodes & relabel_nodes)

                    best_recall = max(best_recall, num_overlap / gt_ged)
                    best_precision = max(best_precision, num_overlap / round_pre_ged)
                    best_f1 = max(best_f1, 2.0 * num_overlap / (gt_ged + round_pre_ged))
                    best_sim = max(best_sim, num_overlap / (gt_ged + round_pre_ged - num_overlap))

                rate.append(best_rate)
                recall.append(best_recall)
                precision.append(best_precision)
                f1.append(best_f1)
                sim.append(best_sim)

            t2 = time.time()
            time_usage.append(t2 - t1)

        time_usage = round(np.mean(time_usage), 3)
        mae = round(np.mean(mae), 3)
        rate = round(np.mean(rate), 3)
        recall = round(np.mean(recall), 3)
        precision = round(np.mean(precision), 3)
        f1 = round(np.mean(f1), 3)
        sim = round(np.mean(sim), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/100p)', 'mae',
             'recall', 'precision', 'f1'))
        self.results.append(
            (self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mae, recall, precision, f1))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path + 'results.txt', 'a') as f:
            print("## Post-processing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    def score(self, testing_graph_set='test', test_k=None):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()

        num = 0  # total testing number
        time_usage = []
        mse = []  # score mse
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                if test_k is None:
                    model_out = self.model(data)
                elif test_k == 0:
                    model_out = self.test_noah(data)
                elif test_k > 0:
                    model_out = self.test_matching(data, test_k)
                else:
                    assert False
                prediction, pre_ged = model_out[0], model_out[1]
                round_pre_ged = round(pre_ged)

                num += 1
                if prediction is None:
                    mse.append(-0.001)
                elif prediction.shape[0] == 1:
                    mse.append((prediction.item() - target) ** 2)
                else:  # TaGSim
                    mse.append(F.mse_loss(prediction, data["ta_ged"]).item())
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
            t2 = time.time()
            time_usage.append(t2 - t1)

            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            if rho[-1] != rho[-1]:
                rho[-1] = 0.
            if tau[-1] != tau[-1]:
                tau[-1] = 0.
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)

        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/pair)', 'mse', 'mae', 'acc',
             'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                             fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path + 'results.txt', 'a') as f:
            print("## Testing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    def batch_score(self, testing_graph_set='test', test_k=100):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()

        batch_results = []
        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            res = []
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                gt_ged = data["ged"]
                time_list, pre_ged_list = self.test_matching(data, test_k, batch_mode=True)
                res.append((gt_ged, pre_ged_list, time_list))
            batch_results.append(res)

        batch_num = len(batch_results[0][0][1])  # len(pre_ged_list)
        for i in range(batch_num):
            time_usage = []
            num = 0  # total testing number
            mse = []  # score mse
            mae = []  # ged mae
            num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
            num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
            num_better = 0
            ged_better = 0.
            rho = []
            tau = []
            pk10 = []
            pk20 = []

            for res_id, res in enumerate(batch_results):
                pre = []
                gt = []
                for gt_ged, pre_ged_list, time_list in res:
                    time_usage.append(time_list[i])
                    pre_ged = pre_ged_list[i]
                    round_pre_ged = round(pre_ged)

                    num += 1
                    mse.append(-0.001)
                    pre.append(pre_ged)
                    gt.append(gt_ged)

                    mae.append(abs(round_pre_ged - gt_ged))
                    if round_pre_ged == gt_ged:
                        num_acc += 1
                        num_fea += 1
                    elif round_pre_ged > gt_ged:
                        num_fea += 1
                    else:
                        num_better += 1
                        ged_better += (gt_ged - round_pre_ged)
                        # print("\nres_id:", res_id, "batch_id:", i, gt_ged, round_pre_ged)
                rho.append(spearmanr(pre, gt)[0])
                tau.append(kendalltau(pre, gt)[0])
                pk10.append(self.cal_pk(10, pre, gt))
                pk20.append(self.cal_pk(20, pre, gt))

            time_usage = round(np.mean(time_usage), 3)
            mse = round(np.mean(mse) * 1000, 3)
            mae = round(np.mean(mae), 3)
            acc = round(num_acc / num, 3)
            fea = round(num_fea / num, 3)
            rho = round(np.mean(rho), 3)
            tau = round(np.mean(tau), 3)
            pk10 = round(np.mean(pk10), 3)
            pk20 = round(np.mean(pk20), 3)
            if num_better > 0:
                avg_ged_better = round(ged_better / num_better, 3)
            else:
                avg_ged_better = None
            self.results.append(
                (self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                 fea, rho, tau, pk10, pk20, num_better, avg_ged_better))

            print(*self.results[-1], sep='\t')
            with open(self.args.abs_path + self.args.result_path + 'results.txt', 'a') as f:
                print(*self.results[-1], sep='\t', file=f)

    def print_results(self):
        for r in self.results:
            print(*r, sep='\t')

        with open(self.args.abs_path + self.args.result_path + 'results.txt', 'a') as f:
            for r in self.results:
                print(*r, sep='\t', file=f)

    @staticmethod
    def data_to_nx(edges, features):
        edges = edges.t().tolist()

        nx_g = nx.Graph()
        n, num_label = features.shape

        if num_label == 1:
            labels = [-1 for i in range(n)]
        else:
            labels = [-1] * n
            for i in range(n):
                for j in range(num_label):
                    if features[i][j] > 0.5:
                        labels[i] = j
                        break

        for i, label in enumerate(labels):
            nx_g.add_node(i, label=label)

        for u, v in edges:
            if u < v:
                nx_g.add_edge(u, v)
        return nx_g

    def test_matching(self, data, test_k, batch_mode=False):
        if self.args.greedy:
            # the Hungarian algorithm, use greedy matching matrix
            pre_ged = None
            soft_matrix = hungarian(data) + 1.0
        else:
            # use the matching matrix generated by GedGNN
            _, pre_ged, soft_matrix = self.model(data)
            m = torch.nn.Softmax(dim=1)
            soft_matrix = (m(soft_matrix) * 1e9 + 1).round()

        n1, n2 = soft_matrix.shape
        g1 = dgl.graph((data["edge_index_1"][0], data["edge_index_1"][1]), num_nodes=n1)
        g2 = dgl.graph((data["edge_index_2"][0], data["edge_index_2"][1]), num_nodes=n2)
        g1.ndata['f'] = data["features_1"]
        g2.ndata['f'] = data["features_2"]

        if batch_mode:
            t1 = time.time()
            solver = KBestMSolver(soft_matrix, g1, g2)
            res = []
            time_usage = []
            for i in [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                if i > test_k:
                    break
                if i == 0:
                    min_res = solver.subspaces[0].ged
                else:
                    solver.get_matching(i)
                    min_res = solver.min_ged
                t2 = time.time()
                time_usage.append(t2 - t1)
                res.append(min_res)
                if pre_ged is not None:
                    time_usage.append(t2 - t1)
                    res.append(min(pre_ged, min_res))
            return time_usage, res
        else:
            solver = KBestMSolver(soft_matrix, g1, g2)
            solver.get_matching(test_k)
            min_res = solver.min_ged
            best_matching = solver.best_matching()
            return None, min_res, best_matching

    def prediction_analysis(self, values, info_str=''):
        """
        Analyze the performance of value prediction.
        :param values: an array of (pre_ged - gt_ged); Note that there is no abs function.
        """
        if not self.args.prediction_analysis:
            return
        neg_num = 0
        pos_num = 0
        pos_error = 0.
        neg_error = 0.
        for v in values:
            if v >= 0:
                pos_num += 1
                pos_error += v
            else:
                neg_num += 1
                neg_error += v

        tot_num = neg_num + pos_num
        tot_error = pos_error - neg_error

        pos_error = round(pos_error / pos_num, 3) if pos_num > 0 else None
        neg_error = round(neg_error / neg_num, 3) if neg_num > 0 else None
        tot_error = round(tot_error / tot_num, 3) if tot_num > 0 else None

        with open(self.args.abs_path + self.args.result_path + self.args.dataset + '.txt', 'a') as f:
            print("prediction_analysis", info_str, sep='\t', file=f)
            print("num", pos_num, neg_num, tot_num, sep='\t', file=f)
            print("err", pos_error, neg_error, tot_error, sep='\t', file=f)
            print("--------------------", file=f)


    def load(self, epoch):
        self.model.load_state_dict(
            torch.load(self.args.abs_path + self.args.model_path + self.args.dataset + '_' + str(epoch)))
