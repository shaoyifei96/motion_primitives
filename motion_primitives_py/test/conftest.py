from motion_primitives_py import *
import numpy as np
import pytest


@pytest.fixture(
    scope="module", 
    params=[PolynomialMotionPrimitive, 
            ReedsSheppMotionPrimitive, 
            InputsMotionPrimitive, 
            JerksMotionPrimitive])
def mp_fixture(request):
    # initialize to defaults
    subclass_specific_data = {}
    control_space_q = 3
    num_dims = 2
    
    # parameter specific logic
    if request.param == ReedsSheppMotionPrimitive:
        control_space_q = 1.5 # corresponds to a 3 dimensional state
    elif request.param == InputsMotionPrimitive:
        subclass_specific_data={"u": np.array([1, -1]), "dt": 1}
    elif request.param == JerksMotionPrimitive:
        subclass_specific_data["supress_redirector"] = True

    # build motion primitive
    n = int(num_dims * control_space_q)
    max_state = 100 * np.ones(n,)
    start_state = np.zeros(n,)
    end_state = np.zeros(n,)
    end_state[:num_dims] = np.ones(num_dims,)
    yield request.param(start_state, end_state, num_dims, max_state, 
                        subclass_specific_data)

@pytest.fixture(scope="module")
def om_fixture():
    resolution = 1
    origin = [0, 0]
    dims = [10, 20]
    data = np.zeros(dims)
    data[5:10, 10:15] = 100
    data = data.flatten('F')
    yield OccupancyMap(resolution, origin, dims, data)

@pytest.fixture(scope="module")
def lattice_fixture():
    # lattice parameters
    control_space_q = 2
    num_dims = 2
    num_output_pts = 5
    max_state = [1, 2*np.pi, 2*np.pi, 100, 1, 1]
    resolution = [.2, .2, np.inf, 25, 1, 1]

    # build lattice
    mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state,
                                 ReedsSheppMotionPrimitive)
    mpl.compute_min_dispersion_space(num_output_pts, resolution)
    mpl.limit_connections(2 * mpl.dispersion)
    yield mpl


@pytest.fixture(scope="module")
def search_fixture(om_fixture, lattice_fixture):
    # define parameters for a graph search
    start_state = [8, 2, 0]
    goal_state = [8, 18, 0]
    goal_tol = np.ones_like(goal_state) * lattice_fixture.dispersion
            
    # build graph search
    yield GraphSearch(lattice_fixture, om_fixture, start_state, goal_state,
                      goal_tol, heuristic='euclidean')
    