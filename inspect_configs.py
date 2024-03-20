import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from tqdm import tqdm
import networkx as nx
import scipy

def get_neighbors(i, j, k, N):
    iprev = (i-1) % N
    jprev = (j-1) % N
    kprev = (k-1) % N

    inext = (i+1) % N
    jnext = (j+1) % N
    knext = (k+1) % N

    return [
            (iprev, j, k), (inext, j, k),
            (i, jprev, k), (i, jnext, k),
            (i, j, kprev), (i, j, knext)
            ]

    #f = k + N * (j + iprev * N)
    #b = k + N * (j + inext * N)

    #u = k + N * (jprev + i * N)
    #d = k + N * (jnext + i * N)

    #l = kprev + N * (j + i * N)
    #r = knext + N * (j + i * N)


    #for i, n1 in enumerate([f, b, u, d, l, r]):
    #    for j, n2 in enumerate([f, b, u, d, l, r]):
    #        if i != j:
    #            assert(n1 != n2)

    #return np.array([u, d, l, r, f, b], dtype=np.int32)

def build_graph(config, L): # t=1 or t=2 
    g = nx.Graph()
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if config[i, j, k] == 1.:
                    g.add_node((i, j, k), node_color='red')
                elif config[i, j, k] == 2.:
                    g.add_node((i, j, k), node_color='blue')
                else: continue

    for i in range(L):
        for j in range(L):
            for k in range(L):
                for (ni, nj, nk) in get_neighbors(i, j, k, L):
                    if config[ni, nj, nk] > 0:
                        g.add_edge((ni, nj, nk), (i, j, k))
    return g




fname = str(argv[1])
L = int(argv[2])
data = np.load(fname)

num_threads = data.shape[0]
assert data[0][0].shape == (L, L, L)
num_epochs = data.shape[1]



# first half is diffusion-based heuristic; second half is optimization
opt = 0
m = False
for l in tqdm(range(num_epochs)): 
    config = data[3][l].astype(float)

    """
    TODO: fix graph drawing ie. project into 2d/3d to get some insight on structure
    graph = build_graph(config, L)
    pos = nx.spring_layout(graph, k=.15, iterations=10) 
    plt.figure()
    nx.draw(graph, pos, with_labels=False, )
    plt.axis('off')
    plt.savefig(f"graph_{l}.png")
    """


    mask = config == 0.
    config[mask] = np.nan
    if m:
        mask = config == 0.
        config[mask] = np.nan
        mask = config == 1.
        config[mask] = np.nan

    #fig, axs = plt.subplots(4, L//4, figsize=(15, 5))
    #for j in range(4):

    fig, axs = plt.subplots(2, L//(2 * 4), figsize=(30, 10)) 
    for j in range(2):
        for i in range(L//(2*4)):
            idx = i * 4 + j
            axs[j, i].imshow(config[:, :, idx], )
            #colors = np.empty(config[:, :, idx].shape + (4,))
            #colors[config[:, :, idx] == 1] = [1, 0, 0, 1]
            #colors[config[:, :, idx] == 2] = [0, 0, 1, 1]
            #axs[j, i].imshow(config[:, :, idx],) 
            #axs[j, i].set_title(f'slice {idx}')
            axs[j, i].axis('off')
    if l > num_epochs // 2:
        fig.suptitle(f"optimization step {2**opt}")
        opt += 1
    else:
        fig.suptitle(f"diffusion step {2**l}")

    #plt.savefig(f"slices/config_slices_epoch{l}_type2.png", dpi=350)
    plt.show()

