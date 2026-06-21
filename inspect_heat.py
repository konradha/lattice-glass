import logging
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from analysis_utils import energy, generate_nn_list

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(args: list[str]) -> int:
    if len(args) != 3:
        raise SystemExit("usage: python inspect_heat.py L num_threads")

    L = int(args[1])
    num_threads = int(args[2])

    nbetas = 60
    ntries = 2
    betas = np.linspace(1.0, 6.9, nbetas)
    fnames = [
        f"heat_data_longer/out_data_L{L}_{beta:.1f}_{trial}.npy"
        for beta in betas
        for trial in range(1, ntries + 1)
    ]

    logger.info("Reading in data")

    eq_configs = []
    for fname in tqdm(fnames):
        configs = np.load(fname)
        for thread in range(num_threads):
            eq_configs.append(configs[thread][-1])

    # TODO: check if we actually get the correct ordering like that.
    eq_configs = np.array(eq_configs)
    eq_configs = eq_configs.reshape((nbetas, ntries * num_threads, L, L, L))

    logger.info("Calculating E, E²")
    nn = generate_nn_list(L)
    e_collection, e2_collection = [], []
    for beta_index, _beta in tqdm(enumerate(betas), total=len(betas)):
        ess, e2ss = [], []
        for thread in range(ntries * num_threads):
            config = eq_configs[beta_index][thread]
            config = config.reshape(L * L * L)
            e = energy(config, nn, L)
            ess.append(e)
            e2ss.append(e**2)
        e_collection.append(ess)
        e2_collection.append(e2ss)

    logger.info("Calculating specific heat")
    e = np.array(e_collection)
    e2 = np.array(e2_collection)
    c_v = []
    e_avg_all = []
    e2_avg_all = []
    for beta_index, beta in enumerate(betas):
        e_avg = np.mean(e[beta_index])
        e2_avg = np.mean(e2[beta_index])
        c_v.append(beta**2 * (e2_avg - e_avg**2))
        e_avg_all.append(e_avg)
        e2_avg_all.append(e_avg**2)

    plt.plot(1 / betas, np.gradient(e_avg_all, 1 / betas), label="dE")
    plt.legend()
    plt.show()

    plt.plot(betas, e2_avg_all, label="E")
    plt.legend()
    plt.show()

    c_v = np.array(c_v)
    plt.plot(1.0 / betas, c_v, label="c_v")
    plt.legend()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv))