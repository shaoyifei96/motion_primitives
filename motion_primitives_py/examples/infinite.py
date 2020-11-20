from motion_primitives_py import *
import numpy as np
import matplotlib.pyplot as plt
import ujson as json

animate = False
mp_subclass_specific_data = {}

# define parameters
name = 'RS'
motion_primitive_type = ReedsSheppMotionPrimitive
control_space_q = 2
num_dims = 2
max_state = [3, 2*np.pi]
num_dense_samples = 200
num_output_pts = num_dense_samples
dispersion_threshhold = 1.05
check_backwards_dispersion = False

# # # %%
# name = 'poly'
# motion_primitive_type = PolynomialMotionPrimitive
# control_space_q = 2
# num_dims = 2
# max_state = [5.51, 1.51, 15, 100]
# mp_subclass_specific_data = {'iterative_bvp_dt': .1, 'iterative_bvp_max_t': 5, 'rho': 1}
# num_dense_samples = 100
# num_output_pts = num_dense_samples
# dispersion_threshhold = 100
check_backwards_dispersion = True


fig, ax = plt.subplots()

mpl_not_tiled = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, False, False, mp_subclass_specific_data)
mpl_not_tiled.compute_min_dispersion_space(
    num_output_pts=num_output_pts, check_backwards_dispersion=check_backwards_dispersion, animate=animate, num_dense_samples=num_dense_samples, dispersion_threshhold=dispersion_threshhold)
mpl_not_tiled.limit_connections(2*mpl_not_tiled.dispersion)

max_state[0] = max_state[0]/3
mpl_tiled = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, True, False, mp_subclass_specific_data)
mpl_tiled.compute_min_dispersion_space(
    num_output_pts=num_output_pts, check_backwards_dispersion=check_backwards_dispersion, animate=animate, num_dense_samples=num_dense_samples, dispersion_threshhold=dispersion_threshhold)
mpl_tiled.limit_connections(2*mpl_tiled.dispersion)

# create grid ... this is so ugly
plt.plot([max_state[0], max_state[0]], [-7*max_state[0], 7*max_state[0]], 'k--', zorder=7)
plt.plot([-max_state[0], -max_state[0]], [-7*max_state[0], 7*max_state[0]], 'k--', zorder=7)
plt.plot([-7*max_state[0], 7*max_state[0]], [-max_state[0], -max_state[0]], 'k--', zorder=7)
plt.plot([-7*max_state[0], 7*max_state[0]], [max_state[0], max_state[0]], 'k--', zorder=7)
plt.plot([-7*max_state[0], 7*max_state[0]], [7*max_state[0], 7*max_state[0]], 'k--', zorder=7)
plt.plot([-7*max_state[0], 7*max_state[0]], [-7*max_state[0], -7*max_state[0]], 'k--', zorder=7)
plt.plot([7*max_state[0], 7*max_state[0]], [-7*max_state[0], 7*max_state[0]], 'k--', zorder=7)
plt.plot([-7*max_state[0], -7*max_state[0]], [-7*max_state[0], 7*max_state[0]], 'k--', zorder=7)
plt.plot([3 * max_state[0], 3 * max_state[0]], [-7*max_state[0], 7*max_state[0]], 'k--', zorder=7)
plt.plot([-3*max_state[0], -3*max_state[0]], [-7*max_state[0], 7*max_state[0]], 'k--', zorder=7)
plt.plot([-7*max_state[0], 7*max_state[0]], [-3 * max_state[0], -3 * max_state[0]], 'k--', zorder=7)
plt.plot([-7*max_state[0], 7*max_state[0]], [3 * max_state[0], 3 * max_state[0]], 'k--', zorder=7)
plt.plot([5 * max_state[0], 5 * max_state[0]], [-7*max_state[0], 7*max_state[0]], 'k--', zorder=7)
plt.plot([-5*max_state[0], -5*max_state[0]], [-7*max_state[0], 7*max_state[0]], 'k--', zorder=7)
plt.plot([-7*max_state[0], 7*max_state[0]], [-5 * max_state[0], -5 * max_state[0]], 'k--', zorder=7)
plt.plot([-7*max_state[0], 7*max_state[0]], [5 * max_state[0], 5 * max_state[0]], 'k--', zorder=7)

offsets = np.linspace(-3, 3, 7)
for i in offsets:
    for j in offsets:
        mpl_tiled.plot_config(plot_mps=True, ax=ax, start_position_override=[2 * i * max_state[0], 2 * j * max_state[0], 0], style1= 'om', style2='om', linewidth=.5)
    for j in offsets:
        if abs(i) <= 1 and abs(j) <= 1:
            mpl_tiled.plot_config(plot_mps=True, ax=ax, start_position_override=[2 * i * max_state[0], 2 * j * max_state[0], 0], style1= 'ob', style2='om', linewidth=.5)
    for j in offsets:
        if abs(i) == 0 and abs(j) == 0:
            mpl_tiled.plot_config(plot_mps=True, ax=ax, start_position_override=[2 * i * max_state[0], 2 * j * max_state[0], 0], style1= 'og', style2='ob', linewidth=.5)

ax.set_xlim(7.5 * np.array([-max_state[0], max_state[0]]))
ax.set_ylim(7.5 * np.array([-max_state[0], max_state[0]]))
ax.set_aspect('equal', 'box')
plt.draw()

filename = f"plots/1_vs_9/{name}_nds{num_dense_samples}_dt{dispersion_threshhold}"
plt.savefig(f"{filename}.png", dpi=1200, bbox_inches='tight')

data_dict = {}
data_dict['not_tiled_num_vertices'] = len(mpl_not_tiled.vertices)
data_dict['not_tiled_num_edges'] = len([mp for mp in list(mpl_not_tiled.edges.flatten()) if mp != None])
data_dict['not_tiled_edges_per_vertex'] = len([mp for mp in list(mpl_not_tiled.edges.flatten()) if mp != None])/len(mpl_not_tiled.vertices)
data_dict['tiled_num_vertices'] = len(mpl_tiled.vertices)
data_dict['tiled_num_edges'] = len([mp for mp in list(mpl_tiled.edges.flatten()) if mp != None])
data_dict['tiled_edges_per_vertex'] = len([mp for mp in list(mpl_tiled.edges.flatten()) if mp != None])/len(mpl_tiled.vertices)


mpl_tiled.save(f"{filename}_tiled_lattice.json")
mpl_not_tiled.save(f"{filename}_untiled_lattice.json")
with open(f"{filename}_data.json", "w") as output_file:
    json.dump(data_dict, output_file, indent=4)

# plt.show()
