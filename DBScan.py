import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class LabelEnum:
    undefined = -2
    noise = -1

def find_all_distances(ds,distance_type='euclidean'):
    D = distance.cdist(ds, ds, 'euclidean')
    return D
def dbscan(ds, eps, minPoints):
    D = find_all_distances(ds)
    n = ds.shape[0]
    labels = np.full(n, LabelEnum.undefined)
    core_neighbors = []
    cores = []
    cluster_number =0 
    for i in range(n):
        if labels[i]!= LabelEnum.undefined:
            continue
        neighbors = list(np.argwhere(D[i,:]<eps).flatten())

        if len(neighbors)<= minPoints:
            labels[i] = LabelEnum.noise
            continue
       
        s = neighbors.copy()
        for q in s:
            if labels[q] == LabelEnum.noise:
                labels[q] = cluster_number
            elif labels[q] !=LabelEnum.undefined:
                continue
            labels[q] = cluster_number
            neighbors.append(q)
            q_neighbors = list(np.argwhere(D[q,:]<eps).flatten())
            if len(q_neighbors)>= minPoints:
                s.extend(q_neighbors)

        cores.append(i)
        cluster_number+=1
        core_neighbors.append(neighbors)
    return cores, core_neighbors, labels
