
from motion_primitives_py import *
import matplotlib.pyplot as plt
import numpy as np

# # # %%
name = 'poly'
motion_primitive_type = PolynomialMotionPrimitive
control_space_q = 2
num_dims = 2
max_state = [1.51, 3, 10, 100]
mp_subclass_specific_data = {'iterative_bvp_dt': .1, 'iterative_bvp_max_t': 5, 'rho': 10}
num_dense_samples = 1000
num_output_pts = num_dense_samples
dispersion_threshholds = [200, 100, 50, 30]
check_backwards_dispersion = True

fig, ax = plt.subplots(len(dispersion_threshholds), 1, sharex=True, sharey=True)
for i, dispersion_threshhold in enumerate(dispersion_threshholds):
    mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type,
                                 tiling=True, plot=False, mp_subclass_specific_data=mp_subclass_specific_data)
    mpl.compute_min_dispersion_space(
        num_output_pts=num_output_pts, check_backwards_dispersion=check_backwards_dispersion, animate=False, num_dense_samples=num_dense_samples, dispersion_threshhold=dispersion_threshhold)
    mpl.limit_connections(2*mpl.dispersion)

    filename = f"plots/anecdotal_example/{name}_nds{num_dense_samples}_dt{dispersion_threshhold}"

    mpl.save(f"{filename}.json")

    mpl = MotionPrimitiveLattice.load(f"{filename}.json")

    gs = GraphSearch.from_yaml("data/corridor.yaml", mpl, heuristic='min_time', goal_tolerance=np.empty(0))
    path, sampled_path, path_cost = gs.run_graph_search()
    gs.plot_path(path, sampled_path, path_cost, ax[i])
plt.savefig(f"{filename}_gs.png", dpi=1200, bbox_inches='tight')
