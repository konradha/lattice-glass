import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from tqdm import tqdm
import networkx as nx
import scipy
from sklearn.manifold import TSNE


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
    last_config = config

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

    if L < 16:
        fig, axs = plt.subplots(2, L//2, figsize=(30, 10))
        for j in range(2):
            for i in range(L//2): 
                axs[j, i].imshow(config[:, :, i], )
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

    else:
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



def get_nn(i, j, k, L):
    iprev, inext = (i - 1) % L, (i + 1) % L
    jprev, jnext = (j - 1) % L, (j + 1) % L
    kprev, knext = (k - 1) % L, (k + 1) % L

    u, d = k + L * (j + iprev * L), k + L * (j + inext * L)
    f, b = k + L * (jprev + i * L), k + L * (jnext + i * L)
    l, r = kprev + L * (j + i * L), knext + L * (j + i * L)

    return [u, d, f, b, l, r]


def generate_nn_list(L):
    d = {}
    for i in range(L):
        for j in range(L):
            for k in range(L):
                idx = k + L * (j + i * L)
                d[idx] = get_nn(i, j, k, L)
    return d

def energy(lattice, nn_list, L):
    e = 0
    for i in range(L ** 3):
        e += local_e(lattice, i, nn_list, L)
    return int(e)

def energy2(lattice, nn_list, L):
    e = 0
    for i in range(L ** 3):
        e += local_e(lattice, i, nn_list, L) ** 2
    return int(e)

def local_e(lattice, idx, nn_list, L):
    #nn = get_nn(i, j, k, L)
    nn = nn_list[idx]
    e = 0.
    ty = lattice[idx]
    if ty == 0: return 0
    if ty > 2: raise Exception("OH NO, bad particle type")
    pref = 3 if ty == 1 else 5
    for n in nn:
        e += int(lattice[n] > 0)
    return (pref - e) ** 2

def nn_e(lattice, idx, nn_list, L):
    e = local_e(lattice, idx, nn_list, L)
    nn = nn_list[idx]
    for n in nn:
        e += local_e(lattice, n, nn_list, L)
    return e


"""
nn = generate_nn_list(L)
for t in range(num_threads):
    last_config = data[t][-1] 
    c = last_config.reshape(L*L*L)
    print(energy(c, nn, L))

MAX = 1e9
adj = np.ones((L*L*L, L*L*L)) * MAX 
for n in nn:
    adj[n, n] = 1
    for nns in nn[n]:
        adj[n, nns] = 1

tsne = TSNE(n_components=2)
similarities = 1 / adj
embedded = tsne.fit_transform(similarities)

node_types = [0,1,2]
for node_type in node_types:
    node_indices = [i for i, t in enumerate(node_types) if t == node_type]
    if node_type == 0:
        marker = 'o'  # Use circles for node type 0
        color = 'gray'
    elif node_type == 1:
        marker = 's'  # Use squares for node type 1
        color = 'red'
    else:
        marker = '^'  # Use triangles for node type 2
        color = 'blue'
    plt.scatter(embedded[node_indices, 0], embedded[node_indices, 1], c=color,)
plt.show()
"""
