from __future__ import annotations

from collections.abc import Mapping, Sequence

Site3 = tuple[int, int, int]


def get_neighbors(i: int, j: int, k: int, L: int) -> list[Site3]:
    return [
        ((i - 1) % L, j, k),
        ((i + 1) % L, j, k),
        (i, (j - 1) % L, k),
        (i, (j + 1) % L, k),
        (i, j, (k - 1) % L),
        (i, j, (k + 1) % L),
    ]


def get_nn(i: int, j: int, k: int, L: int) -> list[int]:
    iprev, inext = (i - 1) % L, (i + 1) % L
    jprev, jnext = (j - 1) % L, (j + 1) % L
    kprev, knext = (k - 1) % L, (k + 1) % L

    u = k + L * (j + iprev * L)
    d = k + L * (j + inext * L)
    f = k + L * (jprev + i * L)
    b = k + L * (jnext + i * L)
    left = kprev + L * (j + i * L)
    right = knext + L * (j + i * L)
    return [u, d, f, b, left, right]


def generate_nn_list(L: int) -> dict[int, list[int]]:
    neighbors: dict[int, list[int]] = {}
    for i in range(L):
        for j in range(L):
            for k in range(L):
                idx = k + L * (j + i * L)
                neighbors[idx] = get_nn(i, j, k, L)
    return neighbors


def local_e(
    lattice: Sequence[int],
    idx: int,
    nn_list: Mapping[int, Sequence[int]],
    L: int | None = None,
) -> float:
    ty = lattice[idx]
    if ty == 0:
        return 0.0
    if ty > 2:
        raise ValueError(f"Bad particle type {ty!r} at site {idx}")

    preferred_connections = 3 if ty == 1 else 5
    occupied_neighbors = 0
    for neighbor in nn_list[idx]:
        occupied_neighbors += int(lattice[neighbor] > 0)
    delta = preferred_connections - occupied_neighbors
    return float(delta * delta)


def energy(lattice: Sequence[int], nn_list: Mapping[int, Sequence[int]], L: int) -> int:
    return int(sum(local_e(lattice, idx, nn_list) for idx in range(L**3)))


def energy2(lattice: Sequence[int], nn_list: Mapping[int, Sequence[int]], L: int) -> int:
    return int(sum(local_e(lattice, idx, nn_list) ** 2 for idx in range(L**3)))


def nn_e(
    lattice: Sequence[int],
    idx: int,
    nn_list: Mapping[int, Sequence[int]],
    L: int | None = None,
) -> float:
    return local_e(lattice, idx, nn_list) + sum(
        local_e(lattice, neighbor, nn_list) for neighbor in nn_list[idx]
    )
