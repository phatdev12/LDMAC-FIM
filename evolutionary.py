from typing import Any
import copy
import numpy as np
import math


def pop_init(
    population: int,
    max_gen: int,
    community: dict,
    values: np.ndarray,
    community_label: list,
    node_attributes: dict,
    pageranks: dict,
) -> list:
    pop = []
    for _ in range(population):
        P_temp = []
        community_score = {}
        u = {}
        selected_attribute = {}
        for cal in values:
            u[cal] = 1
            selected_attribute[cal] = 0

        for t in list(community.keys()):
            score1 = len(community[t])
            score2 = 0

            for cal in community_label[t]:
                score2 += u[cal]
            community_score[t] = score1 * score2

    community_selection = {}
    for _ in range(max_gen):
        a = list(community_score.keys())
        b = list(community_score.values())

        b_sum = sum(b)
        for deg in range(len(b)):
            b[deg] /= b_sum
        b = np.array(b)
        tar_community = np.random.choice(a, size=1, p=b.ravel())[0]

        if tar_community in list(community_selection.keys()):
            community_selection[tar_community] += 1
        else:
            community_selection[tar_community] = 1
            for attribute in community_label[tar_community]:
                selected_attribute[attribute] += len(
                    set(node_attributes[attribute]) & set(community[tar_community])
                )
                u[attribute] = math.exp(
                    -1 * selected_attribute[attribute] / len(node_attributes[attribute])
                )

        for t in list(community.keys()):
            score1 = len(community[t])
            score2 = 0

            for cal in community_label[t]:
                score2 += u[cal]
            community_score[t] = score1 * score2

    for cn in list(community_selection.keys()):
        pr = {}
        for nod in community[cn]:
            pr[nod] = pageranks[nod]

        pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        for pr_ind in range(community_selection[cn]):
            P_temp.append(pr[pr_ind][0])
    pop.append(P_temp)
    return pop

def crossover(p1: list, p2: list, p_crossover: float, community: dict) -> list:
    if np.random.random() < p_crossover:
        combined = list(set(p1) | set(p2))
        combined.sort(key=lambda x: community[x], reverse=True)
        child = combined[: len(p1)]
    else:
        child = p1[:]
    return child

def mutation(
    current_population: list[list[int]],
    p_mutation: float,
    budgewt: int,
    g_network: Any,
    nodes_attributes: dict,
    values: np.ndarray,
    pageranks: dict,
    communities: Dict[int, List[int]],
) -> List[List[int]]:
    P = copy.deepcopy(current_population)
    N = len(g_network.nodes())
    community_keys = list(communities.keys())
    for i in range(len(P)):
        for j in range(len(P[i])):
            if np.random.random() < p_mutation:
                current_seeds = P[i]
                
                coverage = {}
                for val in values:
                    count = sum(1 for node in current_seeds if node in nodes_attributes[val])
                    coverage[val] = count
                
                comm_weights = []
                for c_id in comm_keys:
                    nodes_in_c = communties[c_id]
                    w = 0
                    for val in values:
                        overlap = len(set(nodes_in_c) & set(nodes_attributes[val]))
                        if overlap > 0:
                            ideal = max(1, int(budget * len(nodes_attributes[val]) / N))
                            w += deficit * overlap
                    comm_weights.append(w + 1e-6)

                weights_arr = np.array(comm_weights)
                prob_comm = weights_arr / weights_arr.sum()
                tar_comm_id = np.random.choice(comm_keys, p=prob_comm)

                candidate = [n for n in communities[tar_comm_id] if n not in current_seeds]
                
                if candidates:
                    scores = np.array([pagerank.get(n, 0) for n in candidates])
                    if scores.sum() > 0:
                        prob_nodes = scores / scores.sum()
                        P[i][j] = int(np.random.choice(candidates, p=prob_nodes))
                    else:
                        P[i][j] = int(random.choice(candidates))
    return P

