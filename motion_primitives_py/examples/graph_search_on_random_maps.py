from motion_primitives_py import *
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('motion_primitives')
pkg_path = f'{pkg_path}/motion_primitives_py/'

def generate_data():
    # get all lattices in directory
    file_prefix = f'{pkg_path}data/lattices/dispersion'
    dispersion_threshholds = list(np.arange(1, 201))
    dispersion_threshholds.reverse()
    mpls = {}
    for dispersion_threshhold in deepcopy(dispersion_threshholds):
        filename = f"{file_prefix}{dispersion_threshhold}"
        print(f"{filename}.json")
        try:
            mpls[dispersion_threshhold] = MotionPrimitiveLattice.load(f"{filename}.json")
        except:
            print("No lattice file")
            dispersion_threshholds.remove(dispersion_threshhold)

    data_array = np.zeros((2, 100, len(dispersion_threshholds)))
    for n in range(1, 101):  # iterate over maps
        bag_name = f'{pkg_path}data/maps/random/trees_long0.4_{n}.png.bag'
        # bag_name = f'data/maps/random/trees_dispersion_0.6_{n}.png.bag'
        print(bag_name)
        occ_map = OccupancyMap.fromVoxelMapBag(bag_name, force_2d=True)
        start_state = np.zeros((4))
        goal_state = np.zeros_like(start_state)
        start_state[0:2] = [2, 6]
        goal_state[0:2] = [48, 6]
        # occ_map.plot()
        # plt.show()

        for i, dispersion_threshhold in enumerate(dispersion_threshholds):  # iterate over lattices
            mpl = mpls[dispersion_threshhold]
            print(f'Dispersion {mpl.dispersion}')
            gs = GraphSearch(mpl, occ_map, start_state[:mpl.n], goal_state[:mpl.n],
                             heuristic='min_time', mp_sampling_step_size=occ_map.resolution/mpl.max_state[1])
            gs.run_graph_search()
            data_array[0, n-1, i] = gs.path_cost
            data_array[1, n-1, i] = gs.nodes_expanded
    np.save('random_data', data_array)


def process_data(data):
    file_prefix = f'{pkg_path}data/lattices/dispersion'
    dispersion_threshholds = list(np.arange(1, 201))
    for dispersion_threshhold in deepcopy(dispersion_threshholds):
        filename = f"{file_prefix}{dispersion_threshhold}"
        print(f"{filename}.json")
        try:
            open(f"{filename}.json")
        except:
            print("No lattice file")
            dispersion_threshholds.remove(dispersion_threshhold)

    path_cost = data[0, :, :]
    nodes_expanded = data[1, :, :]
    average_path_cost = np.average(path_cost, axis=0)
    average_nodes_expanded = np.average(nodes_expanded, axis=0)
    plt.plot(dispersion_threshholds, average_path_cost)
    plt.xlabel("Dispersion")
    plt.ylabel("Cost")
    plt.figure()
    plt.plot(dispersion_threshholds, average_nodes_expanded)
    plt.show()


if __name__ == '__main__':
    generate_data()
    # data = np.load('data/random_data.npy')
    # process_data(data)
