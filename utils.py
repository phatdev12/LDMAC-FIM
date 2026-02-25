from numba import jit
import random
import heapq
import numpy as np
import networkx as nx

def indicator(S, n):
    x = np.zeros(n)
    print(n)
    print(list(S))
    x[list(S)] = 1
    return x

def multi_to_set(f, n = None, g_nodes = None):
    if n == None:
        if g_nodes is not None:
            n = len(g_nodes)
        else:
            raise ValueError("Either n or g_nodes must be provided")
    def f_set(S):
        return f(indicator(S, n))
    return f_set

def valoracle_to_single(f, i):
    def f_single(x):
        return f(x, 1000)[i]
    return f_single

def greedy(items, mc, f):
    if mc >= len(items):
        S = set(items)
        return S, f(S)
    for u in items:
        print("set u", set([u]))
    upper_bounds = [(-f(set([u])), u) for u in items]    
    heapq.heapify(upper_bounds)
    starting_objective = f(set())
    S  = set()
    
    while len(S) < mc:
        val, u = heapq.heappop(upper_bounds)
        new_total = f(S.union(set([u])))
        new_val =  new_total - starting_objective
        if new_val >= -upper_bounds[0][0] - 0.01:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S, starting_objective

def sample_live_icm(g, num_graphs):
    live_edge_graphs = []
    for _ in range(num_graphs):
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        for u,v in g.edges():
            if random.random() < g[u][v]['p']:
                h.add_edge(u,v)
        live_edge_graphs.append(h)
    return live_edge_graphs
  
def live_edge_to_edgelist(live_edge_graphs, target_nodes, p_attend):
    Gs = []
    start_poses = []
    ws = []
    Ps = []
    for h in live_edge_graphs:
        g = nx.DiGraph(h)
        # print("egdes:", g.number_of_edges())
        G = np.zeros((g.number_of_edges(), 2), dtype=np.int32)
        P = np.zeros(g.number_of_edges())
        w = np.zeros(len(g))
        print(len(list(target_nodes)))
        w[list(target_nodes)] = 1
        start_pos = np.zeros(len(g)+1, dtype=np.int32)
        curr_pos = 0
        for v in g:
            start_pos[v] = curr_pos
            for u in g.predecessors(v):
                G[curr_pos] = [u, v]
                P[curr_pos] = p_attend[u]
                curr_pos += 1
        start_pos[-1] = g.number_of_edges()
        Gs.append(G)
        Ps.append(P)
        start_poses.append(start_pos)
        ws.append(w)
    return Gs, Ps, start_poses, ws

def make_multilinear_gradient_group(live_graphs, group_indicator, target_nodes, selectable_nodes, p_attend):
    Gs, Ps, start_poses, ws = live_edge_to_edgelist(live_graphs, target_nodes, p_attend)    
    def gradient(x, batch_size):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        grad = gradient_estimate_all_nodes(x, Gs, Ps, start_poses, ws, batch_size) @ group_indicator
        return grad[selectable_nodes, :]
    return gradient

def make_multilinear_objective_samples_group(live_graphs, group_indicator, target_nodes, selectable_nodes, p_attend):
    Gs, Ps, start_poses, ws = live_edge_to_edgelist(live_graphs, target_nodes, p_attend)
    def f_all(x, batch_size):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        pr_reached = np.zeros(len(x_expand))
        for b in range(batch_size):
            i = random.randint(0, len(Gs)-1)
            pr_reached += (1./batch_size)*marginal_coverage_edgelist(x_expand, Gs[i], Ps[i], start_poses[i], ws[i])
        return pr_reached @ group_indicator
    return f_all  

@jit
def gradient_estimate_all_nodes(x, Gs, Ps, start_poses, ws, B):
    '''
    Returns a stochastic estimate of the gradient of the probability that 
    every node is influenced wrt every selectable node. 
    '''
    n = start_poses[0].shape[0] - 1
    grad_est = np.zeros((x.shape[0], n))
    for b in range(B):
        v = random.randint(0, n-1)
        idx = random.randint(0, len(Gs)-1)
        grad_est[:, v] += (n/float(B)) * gradient_coverage_single_edgelist(x, Gs[idx], Ps[idx], start_poses[idx], ws[idx], v)
    return grad_est

@jit
def marginal_coverage_edgelist(x, G, P, start_pos, w):
    probs = np.ones(x.shape[0])
    
    for v in range(x.shape[0]):
        for j in range(start_pos[v], start_pos[v+1]):
            src = G[j][0]
            probs[v] *= 1 - x[src] * P[j]
    probs *= 1 - x
    probs = 1 - probs
    return probs