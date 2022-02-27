from typing import List, Set, Dict
import matplotlib.pyplot as plt
import numpy as np
import networkx
# from utils.config import CONTAGION, LETHALITY


LETHALITY = 0.15
CONTAGION = 0.8


""" Part B Functions """


def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    """
    The function inputs a netwokx.Graph object, a set of infected persons, and the number of iterations.
    It returns the set of infected persons after t iterations of the Linear threshold model.
    :param graph: a netwokx.Graph object
    :param patients_0: a set of initial infected persons
    :param iterations: number of iterations
    :return: a set of infected persons after t iterations
    """

    total_infected = set(patients_0)

    nodes = set(graph.nodes)
    s_t = set(nodes - total_infected)
    infected_t_minus_2 = set()
    for t in range(iterations):
        infected_in_t = set()
        infected_t_minus_1 = total_infected.copy()
        for v in s_t:
            sum_nc_w = calc_neighbors_weights_sum(graph, infected_t_minus_1, v)
            c_t_minus_1 = calc_concern(graph, infected_t_minus_2, v)
            if is_v_infected(sum_nc_w, c_t_minus_1):
                infected_in_t.add(v)
        infected_t_minus_2.update(infected_t_minus_1)
        total_infected.update(infected_in_t)
        s_t = s_t - total_infected
    return total_infected


def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:
    """
    The function inputs a networkx.Graph object, a set of infected persons, and the number of iterations.
    It returns the set of infected persons and the deceased (Removed) after t iterations of the Independent cascade model.
    :param graph: a networkx.Graph object
    :param patients_0: a set of initial infected persons
    :param iterations: number of iterations
    :return: a set of infected persons and the deceased after t iterations
    """

    total_infected = set(patients_0)
    total_deceased = set()

    t_pre_infected = set()
    t_pre_deceased = set()

    for infected_node in total_infected:
        if check_if_infected_is_deceased():
            total_deceased.add(infected_node)

    total_infected = total_infected - total_deceased
    nodes = set(graph.nodes)
    s_t = nodes - total_infected - total_deceased
    NI_set_in_t_pre = total_infected.copy()

    for t in range(iterations):
        infected_in_t = set()
        deceased_in_t = set()
        for v in s_t:
            NI_v_in_t = calc_NI(graph, NI_set_in_t_pre, v)
            c_t_pre = calc_concern_in_t_ICM(graph, t_pre_infected, t_pre_deceased, v)

            for neighbor in NI_v_in_t:
                prob_to_infect = calc_prob_to_be_infected(graph, c_t_pre, v, neighbor)
                if check_if_healthy_is_infected(prob_to_infect):
                    if check_if_infected_is_deceased():
                        deceased_in_t.add(v)
                    else:
                        infected_in_t.add(v)

        t_pre_infected.update(NI_set_in_t_pre)
        t_pre_deceased.update(total_deceased)

        NI_set_in_t_pre = infected_in_t.copy()
        total_infected.update(infected_in_t)
        total_deceased.update(deceased_in_t)
        total_infected = total_infected - total_deceased

        s_t = s_t - total_infected - total_deceased

    return total_infected, total_deceased


""" Part B Auxiliary Functions """


def calc_neighbors_weights_sum(graph: networkx.Graph, total_infected: set, v: str) -> float:
    # TODO complete documentation
    """
    :param graph:
    :param total_infected:
    :param v:
    :return:
    """
    v_neighbors = set(graph.adj[v])
    healthy_neighbors = set(v_neighbors - total_infected)
    infected_neighbors = v_neighbors - healthy_neighbors
    sum_weights = 0
    for infected in infected_neighbors:
        sum_weights += graph.adj[v][infected]['w']
    return sum_weights


def calc_concern(graph: networkx.Graph, total_infected: set, v: str) -> float:
    # TODO complete documentation
    """
    :param graph:
    :param total_infected:
    :param v:
    :return:
    """
    v_neighbors = set(graph.adj[v])
    num_of_neighbors = len(v_neighbors)
    num_of_infected_neighbors = v_neighbors - set(v_neighbors - set(total_infected))
    concern = len(num_of_infected_neighbors) / num_of_neighbors
    return concern


def is_v_infected(sum_w: float, c_t_pre: float) -> bool:
    # TODO complete documentation
    """
    :param sum_w:
    :param c_t_pre:
    :return:
    """
    if CONTAGION * sum_w >= 1 + c_t_pre:
        return True
    else:
        return False


def calc_NI(graph: networkx.Graph, NI_in_t: set, v: str) -> set:
    # TODO complete documentation
    """
    :param graph:
    :param NI_in_t:
    :param v:
    :return:
    """
    v_neighbors = set(graph.adj[v])
    v_not_NI_neighbors = set(v_neighbors - set(NI_in_t))
    NI = v_neighbors - v_not_NI_neighbors
    return NI


def calc_concern_in_t_ICM(graph: networkx.Graph, t_pre_infected: set, t_pre_deceased: set, v: str) -> float:
    # TODO complete documentation
    """
    :param graph:
    :param t_pre_infected:
    :param t_pre_deceased:
    :param v:
    :return:
    """
    v_neighbors = set(graph.adj[v])
    num_v_neighbors = len(v_neighbors)
    if num_v_neighbors == 0:
        return 0
    neighbors_pre_infected = v_neighbors - set(v_neighbors - set(t_pre_infected))
    neighbors_pre_deceased = v_neighbors - set(v_neighbors - set(t_pre_deceased))
    num_neighbors_pre_infected = len(neighbors_pre_infected)
    num_neighbors_pre_deceased = len(neighbors_pre_deceased)
    c_t = (num_neighbors_pre_infected + 3 * num_neighbors_pre_deceased) / num_v_neighbors
    return min(1, c_t)


def calc_prob_to_be_infected(graph: networkx.Graph, c_t_pre: float, v: str, neighbor: str) -> float:
    # TODO complete documentation
    """
    :param graph:
    :param c_t_pre:
    :param v:
    :param neighbor:
    :return:
    """
    w_v_neighbor = graph.adj[v][neighbor]['w']
    prob_to_infect = (1 - c_t_pre) * CONTAGION * w_v_neighbor
    return min(prob_to_infect, 1)


def check_if_healthy_is_infected(prob_to_be_infected: float) -> bool:
    # TODO complete documentation
    """
    :param prob_to_be_infected:
    :return:
    """
    random_num_norm = np.random.uniform()
    if random_num_norm <= prob_to_be_infected:
        return True
    else:
        return False


def check_if_infected_is_deceased() -> bool:
    # TODO complete documentation
    """
    :return:
    """
    random_num_norm = np.random.uniform()
    if random_num_norm <= LETHALITY:
        return True
    else:
        return False


def compute_lethality_effect(graph: networkx.Graph, t: int) -> [Dict, Dict]:
    """
    The function inputs a networkx.Graph and runs the following:
        for each Lethality value in [0.05, 0.15, 0.3, 0.5, 0.7]:
            - Choose 50 random patients_0
            - Run ICM for the chosen random patients_0 for t iterations

    :param graph: a networkx.Graph object
    :param t: number of iterations
    :return: The function returns two dictionaries, one for the mean infected number and the second, for the mean number
     of deceased.
    """
    global LETHALITY
    mean_deaths = {}
    mean_infected = {}
    nodes_list = np.array(graph.nodes)
    size = len(nodes_list)
    for l in (.05, .15, .3, .5, .7):
        LETHALITY = l
        total_infected = list()
        total_deceased = list()
        for iteration in range(30):
            random_index = np.random.choice(size, 50, replace=False)
            patients_0 = list(nodes_list[random_index])
            # TODO implement your code here
            infected_l, deceaesd_l = ICM(graph, patients_0, t)
            total_infected.append(len(infected_l))
            total_deceased.append(len(deceaesd_l))
        mean_infected[l] = np.mean(total_infected)
        mean_deaths[l] = np.mean(total_deceased)

    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    """
    The function's input is the dictionaries from the compute_lethality_effect function and plots two lines such that
    the x-axis is the Lethality value and the y-axis are the mean number of deaths or the mean number of infected.
    :param mean_deaths: dictionary of mean death number of people
    :param mean_infected: dictionary of mean infected number of people
    """
    # TODO implement your code here
    x_axis = list(mean_deaths.keys())
    y_axis_infected = list(mean_infected.values())
    y_axis_deceased = list(mean_deaths.values())

    plt.plot(x_axis, y_axis_infected, label='Infected', color='b')
    plt.plot(x_axis, y_axis_deceased, label='Deaceased', color='g')

    plt.show()
