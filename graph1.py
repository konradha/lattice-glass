import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def build_graph(k=6):
    gs = []
    for _ in range(k):
        gs.append(nx.complete_graph(k))

    g = nx.Graph()
    g.add_node(0)
    for i, g_ in enumerate(gs):
        g = nx.disjoint_union(g, g_)
        g.add_edge(0, (i+1) * k)
    return g

def build_graph2(k=6):
    gs = []
    for _ in range(k):
        gs.append(nx.complete_graph(k))

    g = nx.Graph()
    g.add_node(0)
    for i, g_ in enumerate(gs):
        g = nx.disjoint_union(g, g_)
        g.add_edge(0, (i+1) * k)
 
    for i in range(1, k):
        g.add_edge(i*k + 1, (i+1)*k + 2)
        g.add_edge(i*k + 2, (i+1)*k + 1)

    return g

#for i in range(3, 9):
#    plt.cla()
#    g = build_graph3(i)
#    nx.draw(g)
#    plt.show()

def build_graph3(k=6):
    g = nx.Graph()
    gs = []
    for _ in range(k):
        g_ = nx.complete_graph(k)
        for i in range(k-1):
            g_.remove_edge(i, i+1)
        gs.append(g_)

    for i, g_ in enumerate(gs):
        g = nx.disjoint_union(g, g_)
        g.add_edge(0, (i+1) * k)
    return g

def build_graph3(k=6): 
    def build_cycle(k):
        g = nx.Graph()
        for i in range(k):
            g.add_edge(i, (i+1) % k)
        return g

    g = nx.Graph()
    g.add_node(0)
    gs = []
    for _ in range(k):
        g_ = build_cycle(k)
        gs.append(g_)

    for i, g_ in enumerate(gs):
        g = nx.disjoint_union(g, g_)
        g.add_edge(0, (i+1) * k)
    
    # TODO edges between subgraphs!
    #for i in range(k-1):
    #    g.add_edge(i * k + 2, (i+1) * k + 3)
    #    g.add_edge(i * k + 4, (i+1) * k + 1)
            
    return g

def build_lattice(L):
    g = nx.Graph()
    for i in range(L):
        for j in range(L):
            for k in range(L):
                g.add_node((i, j, k))
    
    for i in range(L):
        for j in range(L):
            for k in range(L):
                g.add_edge((i, j, k), ((i+1)%L, j, k))
                g.add_edge((i, j, k), ((i-1)%L, j, k))
                g.add_edge((i, j, k), (i, (j+1)%L, k))
                g.add_edge((i, j, k), (i, (j-1)%L, k))
                g.add_edge((i, j, k), (i, j, (k+1)%L))
                g.add_edge((i, j, k), (i, j, (k-1)%L))
    return g

def ass_color(g, rho):
    to_color_num = int(len(g.nodes()) * rho)
    to_color = random.sample(list(g.nodes()), to_color_num)
    return to_color


g = build_lattice(4)
#g = build_graph3(6)

to_color_nodes = ass_color(g, .3)

nx.draw(g, node_color=["red"if node in to_color_nodes else "blue" for node in g.nodes() ])
plt.show()

#top = nx.bipartite.sets(g)[0]
#pos = nx.bipartite_layout(g, top)
#nx.draw(g, pos)
#plt.show()
    
    



