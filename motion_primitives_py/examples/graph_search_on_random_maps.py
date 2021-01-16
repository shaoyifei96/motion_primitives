from motion_primitives_py import *
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

if __name__ == '__main__':
    # get all lattices in directory
    file_prefix = 'data/lattices/dispersion'
    dispersion_threshholds = list(np.arange(1, 201))
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
        bag_name = f'data/maps/random/trees_dispersion_0.6_{n}.png.bag'
        print(bag_name)
        occ_map = OccupancyMap.fromVoxelMapBag(bag_name, 0)
        start_state = np.zeros((4))
        goal_state = np.zeros_like(start_state)
        start_state[0:2] = [10, 6]
        goal_state[0:2] = [72, 6]

        for i, dispersion_threshhold in enumerate(dispersion_threshholds):  # iterate over lattices
            mpl = mpls[dispersion_threshhold]
            print(f'Dispersion {mpl.dispersion}')
            gs = GraphSearch(mpl, occ_map, start_state[:mpl.n], goal_state[:mpl.n],
                             heuristic='min_time', mp_sampling_step_size=occ_map.resolution/mpl.max_state[1])
            path, sampled_path, path_cost, nodes_expanded = gs.run_graph_search()
            data_array[0, n-1, i] = path_cost
            data_array[1, n-1, i] = nodes_expanded
    np.save('random_data', data_array)
