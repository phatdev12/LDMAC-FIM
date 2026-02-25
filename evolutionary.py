import numpy as np
import math

def pop_init(population: int, max_gen: int, community: dict, values: np.ndarray, community_label: list, node_attributes: dict, pageranks: dict) -> list:
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
        selected_attribute[attribute] += len(set(node_attributes[attribute]) & set(community[tar_community]))
        u[attribute] = math.exp(-1*selected_attribute[attribute]/len(node_attributes[attribute]))
      
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
      