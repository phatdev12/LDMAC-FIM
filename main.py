import pandas as pd
import networkx as nx
import pickle
import leidenalg
import igraph as ig
import numpy as np
import heapq
import random
import time
from time import strftime, localtime
import math
from utils import live_edge_to_edgelist, gradient_estimate_all_nodes, make_multilinear_objective_samples_group, multi_to_set, greedy, sample_live_icm, indicator, valoracle_to_single
from evolutionary import pop_init

K_SEEDS = 40
POP_SIZE = 10
MAX_GEN = 150
P_CROSSOVER = 0.6
P_MUTATION = 0.1
LAMBDA_VAL = 0.5
PROPAGATION_PROB = 0.01
MC_SIMULATIONS = 1000

NUM_RUNS = 1

graphname = "rice_subset"
attributes = ["color"]

with open(f"dataset/{graphname}.pickle", 'rb') as f:
  G = pickle.load(f)
G = nx.convert_node_labels_to_integers(G, label_attribute='pid')

list_nodes = list(G.nodes())
ngIndex = { list_nodes[ni]: ni for ni in range(len(list_nodes)) }

for u, v in G.edges():
  G[u][v]['p'] = PROPAGATION_PROB

fair_vals_attr = np.zeros((NUM_RUNS, len(attributes)))
greedy_vals_attr = np.zeros((NUM_RUNS, len(attributes)))

include_total = False
group_size = {}
group_size[graphname] = {}
group_size[graphname] = {
  attribute: np.zeros((NUM_RUNS, len(np.unique([G.nodes[v][attribute]])))) for attribute in attributes
}

succession = True
ideal_influences = {}
N = len(G)
alpha = 0.5

for attribute in attributes:
  nvalues = len(np.unique([G.nodes[v][attribute] for v in G.nodes()]))
  group_size[graphname][attribute] = np.zeros((NUM_RUNS, nvalues))

for attribute_index, attribute in enumerate(attributes):
  live_graphs = sample_live_icm(G, MC_SIMULATIONS)
  group_indicator = np.ones((len(G.nodes()), 1))
  print(list(G.nodes()))
  oracle_values = make_multilinear_objective_samples_group(live_graphs, group_indicator, list(G.nodes()), list(G.nodes()), np.ones(len(G)))
  def f_multi(x):
      return oracle_values(x, 1000).sum()
  f_set = multi_to_set(f_multi, n=N, g_nodes=G.nodes())
  violation_0 = []
  violation_1 = []
  min_fraction_0 = []
  min_fraction_1 = []
  pof_0 = []
  time_0 = []
  time_1 = []
  
  for run in range(NUM_RUNS):
      print(strftime("%Y-%m-%d %H:%M:%S"))
      start_time1 = time.perf_counter()
      S, obj = greedy(list(range(len(G))), K_SEEDS, f_set)
      end_time1 = time.perf_counter()
      runningtime1 = end_time1 - start_time1
      
      start_time = time.perf_counter()
      values = np.unique([G.nodes[v][attribute] for v in G.nodes()])
      node_attributes = {}
      for vidx, val in enumerate(values):
          node_attributes[val] = [v for v in G.nodes() if G.nodes[v][attribute] == val]
          group_size[graphname][attribute][run, vidx] = len(node_attributes[val])
      
      opj_succession = {}
      if succession:
          for vidx, val in enumerate(values):
              h = nx.subgraph(G, node_attributes[val])
              h = nx.convert_node_labels_to_integers(h)
              live_graphs_h = sample_live_icm(h, 1000)
              group_indicator = np.ones((len(h.nodes()), 1))
              oracle_values = multi_to_set(valoracle_to_single(
                  make_multilinear_objective_samples_group(live_graphs_h, group_indicator, list(h.nodes()), list(h.nodes()), np.ones(len(h))),
                  0
              ), len(h))
              S_succession, opj_succession[val] = greedy(list(h.nodes()), math.ceil(len(node_attributes[val])*K_SEEDS/len(G)), oracle_values)
      
      if include_total:
          group_indicator = np.zeros((len(G.nodes()), len(values)+1))
          for value_index, value in enumerate(values):
              group_indicator[node_attributes[value], value_index] = 1
          group_indicator[:, -1] = 1
      else:
          group_indicator = np.zeros((len(G.nodes()), len(values)))
          for value_index, value in enumerate(values):
              group_indicator[node_attributes[value], value_index] = 1
      
      oracle_values = make_multilinear_objective_samples_group(
          live_graphs, group_indicator, list(G.nodes()), list(G.nodes()), np.ones(len(G))
      )
          
      f_attribute = {}
      f_multi_attribute = {}
      
      for vidx, value in enumerate(values):
          node_attributes[value] = [v for v in G.nodes() if G.nodes[v][attribute] == value]
          f_multi_attribute[value] = valoracle_to_single(oracle_values, vidx)
          f_attribute[value] = multi_to_set(f_multi_attribute[value], n=N, g_nodes=G.nodes())
      S_attribute = {}
      opt_attribute = {}
      if not succession:
          for value in values:
              S_attribute[value], opt_attribute[value] = greedy(
                  list(range(len(G))),
                  int(len(node_attributes[val])/len(G)*K_SEEDS)
              ), f_attribute[value]
  
      if succession:
          opt_attr = opj_succession
      all_opt = np.array([opt_attr[val] for val in values])
      
      def Eval(X):
          S = [ngIndex[int(i)] for i in X]
          fitness = 0
          x = np.zeros(len(G.nodes))
          x[list(S)] = 1
          
          vals = oracle_values(x, 1000)
          coverage_min = (vals / group_size[graphname][attribute][run]).min()
          violation = np.clip(all_opt - vals, 0, np.inf) / all_opt

          fitness += alpha * coverage_min
          fitness -= (1-alpha) * violation.sum() / len(values)

          return fitness     
      pop = 10
      muttation = 0.1
      crossover = 0.6
      maxgen = 150
      
      graph_ig = ig.Graph.from_networkx(G)
      
      partition = leidenalg.find_partition(graph_ig, leidenalg.ModularityVertexPartition)
      community_all_label = list(set(partition.membership))
      communties = {}
      
      # print("Label of all communities:", community_all_label)
      for community in community_all_label:
          communties[community] = [i for i, x in enumerate(partition.membership) if x == community]
      
      print("communities:", communties)
      community_label = []
      for community in list(communties.keys()):
          for cc in communties[community]:
              print(G.nodes[ngIndex[int(cc)]])
          community_label.append(list(
              set(
                  [G.nodes[ngIndex[int(cc)]][attribute] for cc in communties[community]]
              )
          ))
      pagerank = nx.pagerank(G)
      print(community_label)
      p = pop_init(POP_SIZE, K_SEEDS, communties, values, community_label, node_attributes, pagerank)
      for i in range(MAX_GEN):
          p = sorted(p, key= lambda x: Eval(x), reverse=True)
          
      