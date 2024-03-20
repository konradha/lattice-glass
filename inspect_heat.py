import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



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



L = int(argv[1])
num_threads = int(argv[2])

nbetas = 60
ntries = 2
betas = np.linspace(1., 6.9, nbetas) 
fnames = [f"heat_data_longer/out_data_L{L}_{beta:.1f}_{t}.npy" for beta in betas for t in range(1, ntries + 1)]
betas_extra = np.array([[b, b] for b in betas])
betas_extra = betas_extra.reshape(ntries * nbetas)

logger.info("Reading in data")

gs_configs = []
for i, f in tqdm(enumerate(fnames)):
    configs = np.load(f) 
    for t in range(num_threads):
        gs_configs.append(configs[t][-1])


# TODO: check if we _actually_ get the correct ordering like that
gs_configs = np.array(gs_configs)
gs_configs = gs_configs.reshape((nbetas, ntries * num_threads, L, L, L))

logger.info("Calculating E, E²")
nn = generate_nn_list(L)
e_collection, e2_collection = [], []
for b, beta in tqdm(enumerate(betas)):
    ess, e2ss = [], []
    for t in range(ntries * num_threads):
        config = gs_configs[b][t]
        config = config.reshape(L*L*L)
        e  = energy(config, nn, L)
        #es = energy2(config, nn, L)
        ess.append(e)
        e2ss.append(e ** 2)
    e_collection.append(ess)
    e2_collection.append(e2ss)

logger.info("Calculating specific heat")
e = np.array(e_collection)
e2 = np.array(e2_collection)
c_v = []
e_avg_all = []
e2_avg_all = []
for b, beta in enumerate(betas):
    e_avg = np.mean(e[b])
    e2_avg = np.mean(e2[b])
    c_v.append(beta ** 2 * (e2_avg - e_avg ** 2))
    e_avg_all.append(e_avg)
    e2_avg_all.append(e_avg ** 2)

plt.plot(1/betas, np.gradient(e_avg_all, 1/betas), label="dE")
plt.legend()
plt.show()

plt.plot(betas, e2_avg_all, label="E")
plt.legend()
plt.show()


c_v = np.array(c_v)
plt.plot(1./betas, c_v, label="c_v")
plt.legend()
plt.show()
