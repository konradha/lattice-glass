from sys import argv

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from analysis_utils import get_neighbors


def build_graph(config: np.ndarray, L: int) -> nx.Graph:
    graph = nx.Graph()
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if config[i, j, k] == 1.0:
                    graph.add_node((i, j, k), node_color="red")
                elif config[i, j, k] == 2.0:
                    graph.add_node((i, j, k), node_color="blue")

    for i in range(L):
        for j in range(L):
            for k in range(L):
                if not config[i, j, k] > 0:
                    continue
                for neighbor in get_neighbors(i, j, k, L):
                    if config[neighbor] > 0:
                        graph.add_edge(neighbor, (i, j, k))
    return graph


def plot_config_slices(config: np.ndarray, L: int, title: str) -> None:
    if L < 16:
        fig, axs = plt.subplots(2, L // 2, figsize=(30, 10))
        for row in range(2):
            for col in range(L // 2):
                axs[row, col].imshow(config[:, :, col])
                axs[row, col].axis("off")
    else:
        fig, axs = plt.subplots(2, L // 8, figsize=(30, 10))
        for row in range(2):
            for col in range(L // 8):
                idx = col * 4 + row
                axs[row, col].imshow(config[:, :, idx])
                axs[row, col].axis("off")

    fig.suptitle(title)
    plt.show()


def main(args: list[str]) -> int:
    if len(args) != 3:
        raise SystemExit("usage: python inspect_configs.py output.npy L")

    fname = str(args[1])
    L = int(args[2])
    data = np.load(fname)

    num_threads = data.shape[0]
    assert data[0][0].shape == (L, L, L)
    num_epochs = data.shape[1]
    thread_index = min(3, num_threads - 1)

    opt = 0
    hide_type_one = False

    for epoch in tqdm(range(num_epochs)):
        config = data[thread_index][epoch].astype(float)

        config[config == 0.0] = np.nan
        if hide_type_one:
            config[config == 1.0] = np.nan

        if epoch > num_epochs // 2:
            title = f"optimization step {2**opt}"
            opt += 1
        else:
            title = f"diffusion step {2**epoch}"
        plot_config_slices(config, L, title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv))
