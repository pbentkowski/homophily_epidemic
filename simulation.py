#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the simulation of a paper:

Impact of homophily in adherence to anti-epidemic measures on 
the spread of infectious diseases in social networks

Piotr Bentkowski, Tomasz Gubiec
https://www.mdpi.com/1099-4300/27/10/1071

"""
import copy
import json
import pickle
import sys
from datetime import datetime

import igraph as ig
import numpy as np
from joblib import Parallel, delayed


def SBMgenerator(N=(50, 50), asymetry=1.0, mean_degree=4):
    """
    Creates a stochastic block model (SBM) graph:
        https://en.wikipedia.org/wiki/Stochastic_block_model
    with two groups of individuals. Inter-group connectivity depends on the
    assymetry parameter.


    Parameters
    ----------
    N : 2-element tuple of int, optional
        The number of nodes in the vaccinated and unvaccinated
        groups. The default is (50, 50).
    asymmetry : float, optional
        A ratio of asymmetry between the number of connections
        within a group to the number of connections between groups. If it's
        1 this is a uniform graph, if it's 0 there are no connection between
        vaccinated and unvaccinated groups. The default is 1.0.
    mean_degree : int, optional
        The mean vertex's degree (number of its connections is 2*E/V).
        Where E - number of edges, V - number of vertices
        The default is 4.

    Raises
    ------
    ValueError
        No graph was created.

    Returns
    -------
    Igraph
        An IGraph object (a graph)

    """
    density = mean_degree / (sum(N) - 1)
    N = np.array(N, dtype=int)
    if len(N) != 2:
        raise ValueError("provide two sizes of the groups")
    p11 = (
        density
        * (sum(N) * (sum(N) - 1) - 2 * asymetry * N[0] * N[1])
        / (N[0] * (N[0] - 1) + N[1] * (N[1] - 1))
    )
    if p11 > 1:
        raise ValueError(
            "Cannot generate graph for given density and asymmetry")
    p12 = asymetry * density
    pref_matrix = [[p11, p12], [p12, p11]]
    return ig.Graph.SBM(sum(N), pref_matrix, list(N)), "SBM_graph"


def TCTCgenerator(N=(50, 50), asymetry=1.0, mean_degree=4, Nrewires=200000,
                  Ntermalization=200000):
    """
    Creates a custom graph implementing the Triadic Closure concept:
        https://en.wikipedia.org/wiki/Triadic_closure
    with two groups of individuals. Inter-group connectivity depends on the
    assymetry parameter.

    Parameters
    ----------
    N : 2-element tuple of int, optional
        The number of nodes in the vaccinated and unvaccinated
        groups. The default is (50, 50).
    asymmetry : float, optional
        A ratio of asymmetry between the number of connections
        within a group to the number of connections between groups. If it's
        1 this is a uniform graph, if it's 0 there are no connection between
        vaccinated and unvaccinated groups. The default is 1.0.
    mean_degree : int, optional
        Mean vertex's degree (number of its connections is 2*E/V). Where
        E - number of edges, V - number of vertices
        The default is 4.
    Nrewires : int, optional
        Maximal number of rewrites
    Ntermalization : int, optional
        Number of Termalization

    Raises
    ------
    ValueError
        No graph was created.

    Returns
    -------
    Igraph
        An IGraph object (a graph)
    """
    membership = [0] * N[0] + [1] * N[1]
    n = np.sum(N)
    # start with ER random graph
    g = ig.Graph.Erdos_Renyi(n, m=mean_degree * n // 2)

    # Termalization
    for ind in range(Ntermalization):
        edge = g.es[np.random.randint(g.ecount())]  # random edge
        # random node of this edge
        i = np.random.choice([edge.source, edge.target])
        nei1 = g.neighborhood(i, order=1, mindist=1)  # first neighbours
        nei2 = g.neighborhood(i, order=2, mindist=2)  # second neighbours
        if nei1 and nei2:  # if nei1 and nei2 are not empty
            j = np.random.choice(nei1)  # random first neighbour
            k = np.random.choice(nei2)  # random second neighbour
            if g.degree(j) > 1:  # only if j will not be alone
                # remove connection to the first neighbour
                g.delete_edges([(i, j)])
                g.add_edges([(i, k)])  # add connection to the second neighbour
    # Main loop
    for ind in range(Nrewires):
        edge = g.es[np.random.randint(g.ecount())]  # random edge
        # random node of this edge
        i = np.random.choice([edge.source, edge.target])
        nei1 = g.neighborhood(i, order=1, mindist=1)  # first neighbours
        nei2 = g.neighborhood(i, order=2, mindist=2)  # second neighbours
        if nei1 and nei2:  # if nei1 and nei2 are not empty
            j = np.random.choice(nei1)  # random first neighbour
            k = np.random.choice(nei2)  # random second neighbour
            # only if j will not be alone and j in from the opposite group
            if g.degree(j) > 1 and membership[i] != membership[j]:
                # remove connection to the first neighbour
                g.delete_edges([(i, j)])
                g.add_edges([(i, k)])  # add connection to the second neighbour
        # Dla 1 graf jednorodny, 0 brak połączeń między grupami zaszczepionych
        if 1 - 2 * ig.Graph.modularity(g, membership) < asymetry:
            break  # break if target asymetry (modularity) is reached
    return g, "TC-TC_graph"


def GraphShuffle(g, N=(50, 50)):
    """
    Takes a a Igraph's graph generated by SBMgenerator(), TCgenerator() or
    TCTCgenerator() and shuffles the vertices preserving the number of
    connections (including the in- and out-group connections) in that way
    saving time on generating new Igraph objects.
    """
    perm = np.concatenate((np.random.permutation(N[0]),
                           np.random.permutation(N[1]) + N[0]))
    return ig.Graph(n=N[0] + N[1],
                    edges=[(perm[a], perm[b]) for (a, b) in g.get_edgelist()])


def SIR(
    graph,
    fn_trim, a_trim, ii,
    N=(50, 50),
    pSI=[[0.005, 0.05], [0.01, 0.1]],
    pIR=(0.1, 0.1),
    MAXsteps=1000,
    Ninfected=(5, 5),
):
    """
    0 - S
    3 - I
    5 - R

    Parameters
    ----------
    graph : Igraph object
        A graph generated by SBMgenerator() function
    N : 2-element tuple, optional
        Number of nodes in the vaccinated and unvaccinated groups.
        The default is (50, 50).
    pSI : list (matrix), optional
        Probability of moving from S to I. It is a 2x2 matrix as we have two
        classes of individuals: vaccinated and unvaccinated.
        The default is [[0.005, 0.05], [0.01, 0.1]].
    pIR : tuple, optional
        Probability of moving from I to R. Separately for vaccinated and
        unvaccinated. The default is (0.1, 0.1).
    MAXsteps : int, optional
        Maximal number of iterations of the simulation. The default is 1000.
    Ninfected : tuple, optional
        Initial number of infected in the vaccinated and unvaccinated groups.
        The default is (5, 5).

    Raises
    ------
    ValueError
        When graph size and groups sizes do not match

    Returns
    -------
    list
        List of states of all the individuals

    """
    if graph.vcount() != sum(N):
        raise ValueError("graph size and group sizes do not match")
    # Everybody S
    state = np.zeros(sum(N), dtype=int)
    # Infect parts of the groups
    state[: Ninfected[0]] = 3
    state[N[0]: N[0] + Ninfected[1]] = 3
    # just a copy
    state_new = state[:]
    # get list of edges
    edges = np.array(graph.get_edgelist())
    # shuffle edges sequence
    half = len(edges) // 2
    temp = edges[:half, 0].copy()
    edges[:half, 0] = edges[:half, 1]
    edges[:half, 1] = temp
    np.random.shuffle(edges)
    # identifies the group of the index: 0 - Compliant, 1 - non-Compliant
    # contagion counter
    cont_counter = np.zeros((2, 2))
    # record array: row indicies correspond to node number
    # column are: [0] infecting node, [1] time step to I, [2] time step to R
    records = np.zeros((sum(N), 3), int)

    def group(index):
        return int(bool(int(index // N[0])))

    def one_step(state, state_new, cont_counter, step):
        # global cont_counter
        # contagion thru edges
        for (i, j) in edges:  # loop over edges
            if state[i] == 3 and state[j] == 0:
                if np.random.rand() < pSI[group(i)][group(j)]:
                    state_new[j] = 3
                    cont_counter[group(i)][group(j)] += 1
                    records[j][0] = i
                    records[j][1] = step
            if state[j] == 3 and state[i] == 0:
                if np.random.rand() < pSI[group(j)][group(i)]:
                    state_new[i] = 3
                    cont_counter[group(j)][group(i)] += 1
                    records[i][0] = j
                    records[i][1] = step
        # recovery of vertices
        for i in range(sum(N)):
            if state[i] == 3:
                if np.random.rand() < pIR[group(i)]:
                    state_new[i] = 5
                    records[i][2] = step
        # przepisywanie
        state = state_new[:]
        # cont_counter = copy.deepcopy(cont_counter)

    for i in range(MAXsteps):
        # stop simulation if there are no infected
        if 3 not in state:
            break
        one_step(state, state_new, cont_counter, i)
    rfn = fn_trim + "_" + str(a_trim) + "_" + ii + "_ev.csv"
    np.savetxt(rfn, records, delimiter=',', header="node,t2I,t2R", fmt='%d')
    return state, copy.deepcopy(cont_counter)


def countStates(state, N=(50, 50)):
    """
    Used when one wants the number of individuals in a specific state in each group.

    Parameters
    ----------
    state : Vector{Int}
        Array of states for all individuals.
    N : Tuple{Int, Int}
        Sizes of the two groups (compliant and non-compliant).

    Returns
    -------
    Dict
        Dictionary with counts for states "S", "I", and "R" in each group.
    s"""
    compl = state[: N[0]]
    noncpl = state[N[0]:]
    counts = {}
    counts["S"] = (np.count_nonzero(compl == 0), np.count_nonzero(noncpl == 0))
    counts["I"] = (np.count_nonzero(compl == 3), np.count_nonzero(noncpl == 3))
    counts["R"] = (np.count_nonzero(compl == 5), np.count_nonzero(noncpl == 5))
    return counts


def pSI_parametrised(eff_self, eff_othr, prob_trans=0.1, rnd=20):
    """
    Generates parametrised matrix of infection probabilities between groups
    being given a transmisibility and contraction factors. As well as the
    probability of infection between two naive (un-vaccinated) individuals.

    Parameters
    ----------
    eff_self : float
        Scaling factor for contraction of the disease (eta).
    eff_othr : float
        Scaling factor for transmitting of the disease (delta).
    prob_trans : float, optional
        Probability of infection between two naive hosts (un-vaccinated).
        The default is 0.95.
    rnd : int, optional
        Rounding of the final probilities. The default is 4 digits.

    Returns
    -------
    list
        A list of transmission probalilities between hosts of different groups.

    """
    pSI_11 = np.round(prob_trans, rnd)
    pSI_10 = np.round(prob_trans * eff_self, rnd)
    pSI_01 = np.round(prob_trans * eff_othr, rnd)
    pSI_00 = np.round(prob_trans * eff_self * eff_othr, rnd)
    return [[pSI_00, pSI_01], [pSI_10, pSI_11]]


def generate_pSI_selected_eff(eff_self_list, eff_othr_list, prob_trans=0.1):
    """
    Generates all the combinations of matrices of infection probabilities
    between groups of individuals.
    """
    pSI_big = []
    i = 0
    for eff_self in eff_self_list:
        eff_s = np.round(eff_self, 10)
        for eff_othr in eff_othr_list:
            eff_o = np.round(eff_othr, 10)
            pSI_big.append(
                (i, (eff_s, eff_o), pSI_parametrised(eff_s, eff_o, prob_trans))
            )
            i += 1
    return pSI_big


def dump_run_params_to_json(
        graph_type, NN_v, NN_s, repeats, asym, mean_degr, pIR, MAXsteps,
        Ninfected, p_base, cores, tag, run_start_time, run_end_time,
        graph_reusage_numbr):
    """Saves simulations parametrisation to a JSON file"""
    file_name = "simParams_" + str(tag) + ".json"
    prm_dict = {}
    prm_dict["graph_type"] = str(graph_type)
    prm_dict["N_compliant"] = int(NN_v)
    prm_dict["N_non_compl"] = int(NN_s)
    prm_dict["repeats"] = int(repeats)
    prm_dict["assymetry"] = list(asym)
    prm_dict["mean_degree"] = int(mean_degr)
    prm_dict["pIR"] = list(pIR)
    prm_dict["max_steps"] = int(MAXsteps)
    prm_dict["N_init_infected"] = list(Ninfected)
    prm_dict["p_nn"] = float(p_base)
    prm_dict["N_of_cores"] = int(cores)
    prm_dict["start_time"] = str(run_start_time)
    prm_dict["end_time"] = str(run_end_time)
    prm_dict["graph_reshuffling_number"] = int(graph_reusage_numbr)
    with open(file_name, "w") as fp:
        json.dump(prm_dict, fp, sort_keys=True, indent=4)
    return prm_dict


def load_params_from_json(json_file):
    """Load params from json file"""
    with open(json_file) as jf:
        prm_dict = json.load(jf)
    return prm_dict


def run_bundle_sims(
        graph_generator, NN_v, NN_s, repeats, asym, mean_degr, pSI, pIR,
        MAXsteps, Ninfected, fig_nr, eff, graph_reusage_numb=30):
    """Runs a bundle of simulation for a given parametrisation"""
    graph_reusage_numb = int(graph_reusage_numb)
    R_of_a = []
    for a in asym:
        sample = []
        gg, g_type = graph_generator((NN_v, NN_s), a, mean_degr)
        for i in range(repeats):
            if i % graph_reusage_numb == 0:
                gg, _ = graph_generator((NN_v, NN_s), a, mean_degr)
            else:
                gg = GraphShuffle(gg, (NN_v, NN_s))
            aa = str(np.around(a, 6))
            a_trim = aa + (6 - len(aa)) * '0'
            fn_trim = str(fig_nr).zfill(5)
            ii = str(i).zfill(5)
            st8, cnt = SIR(gg, fn_trim, a_trim, ii,
                           (NN_v, NN_s), pSI, pIR, MAXsteps, Ninfected)
            d = countStates(st8, (NN_v, NN_s))
            sample.append(d["R"])
            edges = np.array(gg.get_edgelist(), dtype=int)
            np.savetxt(fn_trim + "_" + str(a_trim) + "_" + ii + "_gr.csv",
                       edges, delimiter=',', header="node_1,node_2", fmt='%d')
        sample = np.array(sample).T
        R_of_a.append((np.mean(sample[0]), np.mean(sample[1])))
    R_of_a = np.array(R_of_a).T
    return fig_nr, eff, pSI, R_of_a, g_type


def main():
    """ """
    if len(sys.argv) <= 1:
        sys.exit("Give a unique tag string for this simulation bundle")
    tag = str(sys.argv[1])
    start_time = str(datetime.today())
    # Simulation params
    # graph_gen = SBMgenerator
    graph_gen = TCTCgenerator
    graph_type = graph_gen()[1]
    p_base = 0.05
    pIR = (0.2, 0.2)
    NN_v = 2500  # 1000 # 2500
    NN_s = 2500  # 1000 # 2500
    Ninfected = (50, 50)
    asym = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Technical params
    mean_degr = 4
    MAXsteps = 4000
    graph_reusage_numbr = 30  # 30
    # multirun - how many cores of your processor you want to use:
    cores = 30  # 30  # 18
    repeats = 34  # 56 # 34 # 17
    dump_run_params_to_json(
        graph_type, NN_v, NN_s, repeats, asym, mean_degr, pIR, MAXsteps,
        Ninfected, p_base, cores, tag, start_time, "running...",
        graph_reusage_numbr)
    # ============================
    # Here, put you etas and deltas
    # pp = generate_pSI_selected_eff([eta], [delta], p_base)  # needed for budling
    pp = generate_pSI_selected_eff([0.5], [0.05], p_base)     # remove when bundlig
    # ============================
    pSI_big = []
    for ii in range(cores):
        pSI_big.append((ii, pp[0][1], pp[0][2]))
    # ============================
    with open("pSI_big_" + tag + ".pkl", 'wb') as f1:
        pickle.dump(pSI_big, f1)
    elm = Parallel(n_jobs=cores)(
        delayed(run_bundle_sims)(
            graph_gen, NN_v, NN_s, repeats, asym, mean_degr, pSI, pIR,
            MAXsteps, Ninfected, i, eff, graph_reusage_numbr)
        for i, eff, pSI in pSI_big
    )
    print("There was", len(elm), "simulations.")
    # # Single threat just in case
    # for i, pSI in enumerate(pSI_big):
    #     run_bundle_sims(graph_gen, NN_v, NN_s, repeats, asym, mean_degr, pSI,
    #                     pIR, MAXsteps, Ninfected, i, eff, graph_reusage_numbr)
    with open("sim_results_" + tag + ".pkl", 'wb') as f2:
        pickle.dump(elm, f2)
    np.save("sim_asym_" + tag + ".npy", asym, allow_pickle=True)
    # plot_summary(elm, asym, NN_v, NN_s, mean_degr,
    #              p_base, pIR, graph_type, tag)
    dump_run_params_to_json(
        graph_type, NN_v, NN_s, repeats, asym, mean_degr, pIR, MAXsteps,
        Ninfected, p_base, cores, tag, start_time, str(datetime.today()),
        graph_reusage_numbr)


if __name__ == "__main__":
    main()
