# Examine the change in average degree and dispersion as more samples are added.
# Plot average degree vs number of samples and dispersion vs number of sample.

# Notes:
# -- Hypothesize that the graph degree should roughly stabilizes at some value
#    after a minimum number of points, and before the test points start getting exhausted.
# -- Hypothesize that sharp cliffs in dispersion/degree and the runaway increase
#    in graph degree as the test points get exhausted are artifacts of the grid
#    initialization, and go away with fuzzing.
# -- Questions to answer:
#   -- How does number of dimensions affects average degree growth?
#   -- How does random/dithered test points affect average degree growth?
# -- For some reason, this script is super slow and not exercising multiple cores.


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from motion_primitives_py.euclidean_motion_primitive import EuclideanMotionPrimitive
from motion_primitives_py.motion_primitive_lattice import MotionPrimitiveLattice
from motion_primitives_py.reduce_graph_degree import reduce_graph_degree

if __name__ == '__main__':

    # Euclidean Graph in 2D
    motion_primitive_type = EuclideanMotionPrimitive
    control_space_q = 1
    num_dims = 2
    max_state = 1 * np.ones(num_dims*control_space_q)
    mp_subclass_specific_data = {}
    resolution = list(0.1 * np.ones(control_space_q))
    fuzz_factor = 0.25 # Useful values include {0 (no fuzz), 0.25 (medium fuzz)}.
    tiling = False
    check_backwards_dispersion = False
    basename = f"euclidean_lattice_{num_dims}D_{resolution[0]}res_{fuzz_factor}fuzz_exhaustive"

    # Euclidean Graph in 6D
    motion_primitive_type = EuclideanMotionPrimitive
    control_space_q = 1
    num_dims = 4
    max_state = 1 * np.ones((control_space_q+1,))
    mp_subclass_specific_data = {}
    resolution = list(0.5 * np.ones(control_space_q+1)) # Not sure about this.
    fuzz_factor = 0.25 # Useful values include {0 (no fuzz), 0.25 (medium fuzz)}.
    tiling = False
    check_backwards_dispersion = False
    basename = f"euclidean_lattice_{num_dims}D_{resolution[0]}res_{fuzz_factor}fuzz_exhaustive"

    # Either load or create the specified lattice. Lattice should not have limited edges yet.
    if Path(f'{basename}.json').exists():
        mpl = MotionPrimitiveLattice.load(f'{basename}.json', plot=False)
    else:
        mpl = MotionPrimitiveLattice(
            control_space_q,
            num_dims,
            max_state,
            motion_primitive_type,
            tiling,
            False,
            mp_subclass_specific_data)
        # Compute the number of tests points used.
        potential_sample_pts, _ = mpl.uniform_state_set(
            mpl.max_state[:mpl.control_space_q], resolution[:mpl.control_space_q])
        print(potential_sample_pts)
        n_test_points = potential_sample_pts.shape[0]
        print(f'Used {n_test_points} test points.')
        # Compute exhaustive dispersion sequence.
        mpl.compute_min_dispersion_space(
            num_output_pts=n_test_points-1, # This -1 is needed, but shouldn't be.
            resolution=resolution,
            check_backwards_dispersion=check_backwards_dispersion,
            fuzz_factor=fuzz_factor)
        mpl.save(f'{basename}.json')

    # Compute number of tests points used.
    potential_sample_pts, _ = mpl.uniform_state_set(
        mpl.max_state[:mpl.control_space_q], resolution[:mpl.control_space_q])
    n_test_points = potential_sample_pts.shape[0]
    print(f'Used {n_test_points} test points.')

    # Build complete cost matrix.
    costs = np.zeros(mpl.edges.shape)
    for i in range(costs.shape[0]):
        for j in range(costs.shape[1]):
            costs[i,j] = mpl.edges[i,j].cost

    # Calculate average degree over time given changing 2*dispersion limit.
    edge_counts = np.zeros(mpl.edges.shape[0])
    for i in range(mpl.edges.shape[0]):
        edge_counts[i] = np.count_nonzero(costs[:i,:i] <= 2 * mpl.dispersion_list[i])
    average_degree = edge_counts / (1+np.arange(edge_counts.size))

    # Plot average degree vs number of samples.
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.plot(range(mpl.edges.shape[0]), average_degree, color='black')
    ax.set_xlabel('number samples')
    ax.set_ylabel('average degree')
    # Plot dispersion vs number of samples (on same axes).
    ax2 = ax.twinx()
    color = 'lightgrey'
    ax2.set_ylabel('dispersion', color=color)
    ax2.plot(range(mpl.edges.shape[0]), mpl.dispersion_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.savefig(f'{basename}.pdf')

    # Plot the configuration with final dispersion value.
    if num_dims <= 3:
      mpl.limit_connections(2*mpl.dispersion)
      mpl.plot_config()

    plt.show()
