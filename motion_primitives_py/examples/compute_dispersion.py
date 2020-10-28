from motion_primitives_py import *
import time
import numpy as np
import matplotlib.pyplot as plt
from pycallgraph import PyCallGraph, Config
from pycallgraph.output import GraphvizOutput

"""
Compare fully expanded graphs (up to a certain depth) of lattices vs trees
"""
depth = 1
resolution = [.1, np.inf]
no_sampling_value = 1.5
colorbar_max = None

mpl = MotionPrimitiveLattice.load("data/lattice_test.json")
mpl.plot_config(plot_mps=True)
plt.figure()

# mpl.max_state[0] = 2
# mpl.max_state[1] = 2
mpt = MotionPrimitiveTree(mpl.control_space_q, mpl.num_dims,  [
                          np.inf, np.inf, mpl.max_state[2]], InputsMotionPrimitive, plot=False, mp_subclass_specific_data={'num_u_per_dimension': 5, 'dt': .25})
# print(mpl.dispersion)
# print(sum([1 for mp in np.nditer(mpl.edges, ['refs_ok']) if mp != None])/len(mpl.vertices))
# print(mpl.max_state)

start_state = np.zeros((mpl.n))
goal_state = np.zeros_like(start_state)

om_resolution = 1
origin = [-20, -20]
dims = [40, 40]
data = np.zeros(dims)
data = data.flatten('F')
occ_map = OccupancyMap(om_resolution, origin, dims, data)
# start_state[0:3] = np.array([20, 20, 0])*resolution
goal_state[0:3] = np.array([19, 19, 0])*om_resolution

# occ_map = OccupancyMap.fromVoxelMapBag('trees_dispersion_1.1.bag', 0)
# start_state[0:2] = [10, 6]
# goal_state[0:2] = [22, 6]

goal_tolerance = np.ones_like(start_state)*occ_map.resolution*0

print("Motion Primitive Tree")
# print(start_state)
gs = GraphSearch(mpt, occ_map, start_state[:mpl.n], goal_state[:mpl.n], goal_tolerance,
                 heuristic='zero', mp_sampling_step_size=1)
nodes = gs.expand_all_nodes(depth)
vertices = np.array([v.state for v in nodes])
print(vertices.shape)
mpl.mp_subclass_specific_data['iterative_bvp_dt'] = .2
mpl.mp_subclass_specific_data['iterative_bvp_max_t'] = 2
dispersion = mpl.compute_dispersion_from_graph(vertices, resolution, no_sampling_value=no_sampling_value,  colorbar_max=colorbar_max)


# print("Motion Primitive Lattice")
mpl = MotionPrimitiveLattice.load("data/lattice_test.json")
mpl.plot = False
gs = GraphSearch(mpl, occ_map, start_state[:mpl.n], goal_state[:mpl.n], goal_tolerance,
                 heuristic='zero', mp_sampling_step_size=1)
plt.figure()
nodes = gs.expand_all_nodes(depth)
vertices = np.array([v.state for v in nodes])
print(vertices.shape)
# vertices = mpl.vertices
mpl.mp_subclass_specific_data['iterative_bvp_dt'] = .2
mpl.mp_subclass_specific_data['iterative_bvp_max_t'] = 2
# dispersion = mpl.compute_dispersion_from_graph(vertices, [.25, np.inf])
dispersion = mpl.compute_dispersion_from_graph(vertices, resolution, no_sampling_value=no_sampling_value, colorbar_max=colorbar_max)

plt.show()
