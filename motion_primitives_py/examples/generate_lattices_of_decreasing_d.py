
from motion_primitives_py import *
# # # %%
name = 'poly'
motion_primitive_type = PolynomialMotionPrimitive
control_space_q = 2
num_dims = 2
max_state = [1.51, 3, 10, 100]
mp_subclass_specific_data = {'iterative_bvp_dt': .1, 'iterative_bvp_max_t': 5, 'rho': 10}
num_dense_samples = 100
num_output_pts = num_dense_samples
dispersion_threshholds = [50]
check_backwards_dispersion = True

for dispersion_threshhold in dispersion_threshholds:
    mpl_tiled = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type,
                                       tiling=True, plot=False, mp_subclass_specific_data=mp_subclass_specific_data)
    mpl_tiled.compute_min_dispersion_space(
        num_output_pts=num_output_pts, check_backwards_dispersion=check_backwards_dispersion, animate=False, num_dense_samples=num_dense_samples, dispersion_threshhold=dispersion_threshhold)
    mpl_tiled.limit_connections(2*mpl_tiled.dispersion)

    filename = f"plots/anecdotal_example/{name}_nds{num_dense_samples}_dt{dispersion_threshhold}"

    mpl_tiled.save(f"{filename}.json")
