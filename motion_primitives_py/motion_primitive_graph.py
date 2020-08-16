#!/usr/bin/python3
from motion_primitives_py import PolynomialMotionPrimitive, InputsMotionPrimitive, ReedsSheppMotionPrimitive
import numpy as np
import matplotlib.pyplot as plt

class MotionPrimitiveGraph():
    """
    Compute motion primitive graphs for quadrotors over different size state spaces

    Attributes:
        control_space_q, 
            derivative of configuration which is the control input.
        num_dims, 
            dimension of configuration space
        max_state, 
            list of max values of position space and its derivatives
        motion_primitive_type,
            class that the motion primitive edges belong to. Must be a subclass of MotionPrimitive
        plot, 
            boolean of whether to create/show plots
        vertices, (M, N) 
            minimum dispersion set of M points sampled in N dimensions, 
            the vertices of the graph
        edges, (M, M) 
            adjacency matrix of MotionPrimitive objects representing edges of 
            the graph, with each element (x,y) of the matrix corresponding to a
            trajectory from state vertices(x) to state vertices(y).  
    """

    def __init__(self, control_space_q, num_dims,  max_state, motion_primitive_type, tiling=True, plot=False):
        """
        Input:
            control_space_q, derivative of configuration which is the control input.
            num_dims,        dimension of configuration space
            max_state, list of max values of position space and its derivatives
            plot, boolean of whether to create/show plots
        """
        self.control_space_q = control_space_q
        self.num_dims = num_dims
        self.max_state = np.array(max_state)
        self.plot = plot
        self.motion_primitive_type = motion_primitive_type
        self.dispersion_distance_fn = self.dispersion_distance_fn_simple_norm  # TODO pass as param/input?
        self.n = (self.control_space_q)*self.num_dims  # dimension of state space
        self.mp_subclass_specific_data = {}

        # Setup specific to motion primitive being used TODO move elsewhere
        if self.motion_primitive_type == PolynomialMotionPrimitive:
            self.mp_subclass_specific_data['x_derivs'] = self.motion_primitive_type.get_dynamics_polynomials(self.control_space_q)
        elif self.motion_primitive_type == InputsMotionPrimitive:
            self.mp_subclass_specific_data['dynamics'] = self.motion_primitive_type.get_dynamics_polynomials(
                self.control_space_q, self.num_dims)
        elif self.motion_primitive_type == ReedsSheppMotionPrimitive:
            self.n = 3

        # TODO only really for lattice, maybe should move there
        if tiling:
            self.num_tiles = 3 ** self.num_dims
        else:
            self.num_tiles = 1

        # TODO update Tree subclass to use latest data structures
        self.motion_primitives_list = []
        self.dispersion = None
        self.dispersion_list = []

        if self.plot:
            fig, self.ax = plt.subplots()
            fig_3d, ax_3d = plt.subplots()
            self.ax_3d = fig_3d.add_subplot(111, projection='3d')

    def uniform_state_set(self, bounds, resolution, random=False):
        """
        Return a uniform Cartesian sampling over vector bounds with vector resolution.
        Input:
            bounds, (N, 2) bounds over N dimensions
            resolution, (N,) resolution over N dimensions
        Output:
            pts, (M,N) set of M points sampled in N dimensions
        """
        assert len(bounds) == len(resolution)
        independent = []
        bounds = np.asarray(bounds)
        for (a, b, r) in zip(bounds[:, 0], bounds[:, 1], resolution):
            for _ in range(self.num_dims):
                if random:
                    independent.append(a + np.random.rand((np.ceil((b-a)/r+1).astype(int)))*(b-a))
                else:
                    if r != np.inf:
                        independent.append(np.arange(a, b+.00001, r))
                    else:
                        independent.append(0)  # if the requested resolution is infinity, just return 0
        if self.motion_primitive_type == ReedsSheppMotionPrimitive:  # hack
            independent.pop()
            self.n = 3
        joint = np.meshgrid(*independent)
        pts = np.stack([j.ravel() for j in joint], axis=-1)
        return pts

    def dispersion_distance_fn_simple_norm(self, start_pts, end_pts):
        score = np.linalg.norm(start_pts[:, np.newaxis]-end_pts, axis=2)
        return score, None

    def compute_min_dispersion_points(self, num_output_pts, potential_sample_pts, starting_score, starting_output_sample_index):
        actual_sample_indices = np.zeros((num_output_pts)).astype(int)
        actual_sample_indices[0] = starting_output_sample_index

        # distances of potential sample points to closest chosen output MP node # bottleneck
        min_score = np.ones((starting_score.shape[0], 2))*np.inf
        min_score[:, 0] = np.amin(starting_score, axis=1)
        for mp_num in range(1, num_output_pts):  # start at 1 because we already chose the closest point as a motion primitive
            # distances of potential sample points to closest chosen output MP node
            min_score[:, 0] = np.amin(min_score, axis=1)
            # take the new point with the maximum distance to its closest node
            index = np.argmax(min_score[:, 0])
            result_pt = potential_sample_pts[index, :]
            actual_sample_indices[mp_num] = np.array((index))
            min_score[index, 0] = - np.inf  # give nodes we have already chosen low score
            min_score[:, 1] = self.dispersion_distance_fn(potential_sample_pts, result_pt)  # new point's score
        actual_sample_pts = potential_sample_pts[actual_sample_indices]
        return actual_sample_pts, actual_sample_indices

if __name__ == "__main__":
    control_space_q = 3
    num_dims = 2
    max_state = [3, 1, 1, 100, 1, 1]
    mpg = MotionPrimitiveGraph(control_space_q, num_dims, max_state, True)
