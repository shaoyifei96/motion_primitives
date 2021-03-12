from motion_primitives_py import *
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import rospkg
import os
import time

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

    data_array = np.zeros((3, 100, len(dispersion_threshholds)))
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
            data_array[2, n-1, i] = len(gs.neighbor_nodes)
    np.save('random_data', data_array)


def generate_data_clutteredness():
    # get all lattices in directory
    file_prefix = f'{pkg_path}data/lattices/dispersion'
    dispersion_threshhold = 150
    mpl = MotionPrimitiveLattice.load(f"{file_prefix}{dispersion_threshhold}.json")

    data_dict = {}
    counter = 0
    for root, dirs, files in os.walk('/home/laura/Documents/research/quals/png_maps/'):
        for f in files:
            if "bag" in f:
                counter += 1
                poisson_spacing = float(f.split("_")[1][4:])
                print(f"Poisson Spacing: {poisson_spacing}")
                occ_map = OccupancyMap.fromVoxelMapBag(root+f, force_2d=True)
                start_state = np.zeros((4))
                goal_state = np.zeros_like(start_state)
                start_state[0:2] = [2, 6]
                goal_state[0:2] = [48, 6]
                # occ_map.plot()
                # plt.show()

                gs = GraphSearch(mpl, occ_map, start_state[:mpl.n], goal_state[:mpl.n],
                                 heuristic='min_time', mp_sampling_step_size=occ_map.resolution/mpl.max_state[1], goal_tolerance=[1, 1])
                gs.run_graph_search()
                if gs.path_cost is None:
                    continue
                data_dict[poisson_spacing] = data_dict.get(poisson_spacing, None)
                d = np.array([gs.path_cost, gs.num_collision_checks]).reshape(1, 2)
                if data_dict[poisson_spacing] is None:
                    data_dict[poisson_spacing] = d
                else:
                    data_dict[poisson_spacing] = np.vstack((data_dict[poisson_spacing], d))
                if counter > 1000:
                    break

    data_processed = np.empty((len(data_dict), 5))
    for i, (k, v) in enumerate(data_dict.items()):
        data_processed[i, 0] = k
        data_processed[i, 1] = np.average(v[:, 0])
        data_processed[i, 2] = np.std(v[:, 0])
        data_processed[i, 3] = np.average(v[:, 1])
        data_processed[i, 4] = np.std(v[:, 1])
    np.save("cluttered_data", data_processed)
    plt.show()

def process_data_clutteredness():
    data_processed = np.load("cluttered_data.npy")
    fig, ax = plt.subplots(2, 1, sharex=True)
    print(data_processed[:, 0])
    print(data_processed[:, 1])
    ax[0].plot(data_processed[:, 0], data_processed[:, 1], 'o')
    # ax[0].fill_between(data_processed[:,0], data_processed[:,1]-data_processed[:,2], data_processed[:,1]+data_processed[:,2],
    #                    alpha=0.5)
    ax[1].plot(data_processed[:, 0], data_processed[:, 3], 'o')
    # ax[1].fill_between(data_processed[:,0], data_processed[:,3]-data_processed[:,4], data_processed[:,3]+data_processed[:,4],
    #                    alpha=0.5)
    ax[0].set_ylabel("Cost")
    ax[1].set_ylabel("Nodes considered")
    ax[1].set_xlabel("Poisson Spacing")
    plt.show()


def process_data(data):
    file_prefix = f'{pkg_path}data/lattices/dispersion'
    dispersion_threshholds = list(np.arange(1, 201))
    dispersion_threshholds.remove(100)
    dispersion_threshholds.reverse()
    for dispersion_threshhold in deepcopy(dispersion_threshholds):
        filename = f"{file_prefix}{dispersion_threshhold}"
        print(f"{filename}.json")
        try:
            open(f"{filename}.json")
        except:
            print("No lattice file")
            dispersion_threshholds.remove(dispersion_threshhold)
    fig, ax = plt.subplots(2, 1, sharex=True)
    path_cost = data[0, :, :]
    average_path_cost = np.nanmean(path_cost, axis=0)
    ax[0].plot(dispersion_threshholds, average_path_cost)
    ax[0].set_ylabel("Cost")
    error = np.nanstd(path_cost, axis=0)
    ax[0].fill_between(dispersion_threshholds, average_path_cost-error, average_path_cost+error,
                       alpha=0.5)

    nodes_expanded = data[1, :, :]
    # average_nodes_expanded = np.average(nodes_expanded, axis=0)
    # ax[2].plot(dispersion_threshholds, average_nodes_expanded)
    # # ax[1].xlabel("Dispersion")
    # ax[2].set_ylabel("Nodes Expanded")

    nodes_considered = data[2, :, :]
    nodes_considered[nodes_expanded > 999] = np.nan
    average_nodes_considered = np.nanmean(nodes_considered, axis=0)
    ax[1].plot(dispersion_threshholds, average_nodes_considered)
    # ax[2].xlabel("Dispersion")
    print(nodes_considered[:, dispersion_threshholds.index(95)])

    error = np.nanstd(nodes_considered, axis=0)
    ax[1].fill_between(dispersion_threshholds, average_nodes_considered-error, average_nodes_considered+error,
                       alpha=0.5)
    ax[1].set_ylabel("Nodes Considered")

    ax[1].set_xlabel("Dispersion")

    plt.show()


if __name__ == '__main__':
    generate_data_clutteredness()
    process_data_clutteredness()
    # data = np.load('data/random_data_2.npy')
    # process_data(data)
