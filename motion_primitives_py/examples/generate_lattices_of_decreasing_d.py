
#!/usr/bin/python3

import matplotlib
import numpy as np
from copy import deepcopy
import matplotlib.animation as animation
from motion_primitives_py import *
import matplotlib.pyplot as plt
import argparse
import rospkg
"""
Run the dispersion algorithm, and save the lattices at a specified set of desired dispersions (all starting from the same dense set).
Run graph search on the same map with said lattices.
"""
# # # %%
name = ''
motion_primitive_type = OptimizationMotionPrimitive
control_space_q = 2
num_dims = 2
max_state = [1.51, 3, 10, 100]
mp_subclass_specific_data = {'iterative_bvp_dt': .1, 'iterative_bvp_max_t': 5, 'rho': 100}
num_dense_samples = 10
num_output_pts = num_dense_samples
dispersion_threshholds = np.arange(200, 30, -5).tolist()
check_backwards_dispersion = True
costs_list = []
nodes_expanded_list = []
file_prefix = 'data/lattices/dispersion' + name  # TODO don't overwrite every time


def init():
    return


def animation_helper(i,  dts, plot_type='maps'):
    dispersion_threshhold = dts[i]
    filename = f"{file_prefix}{dispersion_threshhold}"
    print(f"{filename}.json")
    try:
        mpl = MotionPrimitiveLattice.load(f"{filename}.json")
    except:
        return

    start_state = np.zeros((mpl.n))
    goal_state = np.zeros_like(start_state)
    occ_map = OccupancyMap.fromVoxelMapBag('data/maps/random/trees_long0.4_1.png.bag', force_2d=True)
    start_state[0:2] = [2, 6]
    goal_state[0:2] = [48, 6]

    gs = GraphSearch(mpl, occ_map, start_state[:mpl.n], goal_state[:mpl.n],
                     heuristic='min_time', mp_sampling_step_size=occ_map.resolution/mpl.max_state[1])

    gs.run_graph_search()
    ax0.clear()
    gs.plot(ax0)
    costs_list.append(gs.path_cost)
    nodes_expanded_list.append(gs.nodes_expanded)


def animation_helper2(i):

    lines[0].set_data(dispersion_threshholds[:i+1], costs_list[:i+1])
    lines[1].set_data(dispersion_threshholds[:i+1], nodes_expanded_list[:i+1])
    return lines


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-gnl", help="Generate new lattices", action='store_true')
    args = parser.parse_args()
    if args.gnl:
        print("Generating new lattices")
        generate_new_lattices = True
    else:
        print("Using lattices in data/lattices dir")
        generate_new_lattices = False
    # fig, ax = plt.subplots(len(dispersion_threshholds),1, sharex=True, sharey=True)
    if generate_new_lattices:
        mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type,
                                     tiling=True, plot=False, mp_subclass_specific_data=mp_subclass_specific_data, saving_file_prefix=file_prefix)
        mpl.compute_min_dispersion_space(
            num_output_pts=num_output_pts, check_backwards_dispersion=check_backwards_dispersion, animate=False, num_dense_samples=num_dense_samples, dispersion_threshhold=deepcopy(dispersion_threshholds))

    for dispersion_threshhold in deepcopy(dispersion_threshholds):
        filename = f"{file_prefix}{dispersion_threshhold}"
        print(f"{filename}.json")
        try:
            open(f"{filename}.json")
        except:
            print("No lattice file")
            dispersion_threshholds.remove(dispersion_threshhold)
    print(dispersion_threshholds)

    f, ax0 = plt.subplots(1, 1)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('motion_primitives')
    occ_map = OccupancyMap.fromVoxelMapBag(f'{pkg_path}/motion_primitives_py/data/maps/trees_dispersion_1.1.bag')
    occ_map.plot(ax=ax0)
    f.tight_layout()

    normal_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    ani = animation.FuncAnimation(
        f, animation_helper, len(dispersion_threshholds), interval=1000, fargs=(deepcopy(dispersion_threshholds),), repeat=False, init_func=init)
    ani.save('data/videos/planning_with_decreasing_dispersion.mp4', dpi=800)
    print("done saving")

    f2, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Dispersion')
    ax1.set_ylabel('Cost', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim(max(dispersion_threshholds), 0)
    ax1.set_ylim(0, max(costs_list)*1.1)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim(0, max(nodes_expanded_list)*1.1)
    color = 'tab:blue'
    ax2.set_ylabel('Nodes Expanded', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.invert_xaxis()
    ax2.invert_xaxis()
    costs_line, = ax1.plot([], [], '*--r')
    nodes_expanded_line, = ax2.plot([], [], '*--b')
    lines = [costs_line, nodes_expanded_line]
    ani2 = animation.FuncAnimation(
        f2, animation_helper2, len(costs_list), interval=1000, repeat=False, init_func=init)
    # f.tight_layout()

    ani2.save('data/videos/nodes_expanded_cost_vs_dispersion.mp4', dpi=800)

    # plt.show()
