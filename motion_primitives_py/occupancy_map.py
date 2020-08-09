import rosbag
import numpy as np
import matplotlib.pyplot as plt

#from motion_primitives_py.msg import VoxelMap


class OccupancyMap():
    def __init__(self, resolution, origin, dims, data, margin=0):
        self.resolution = resolution
        self.origin = origin
        self.dims = dims
        # distance margin to inflate obstacles by
        self.margin = margin

        if self.dims[2] == 1:
            self.dims = self.dims[:2]
            self.origin = self.origin[:2]
            self.is_free = self._is_free_2D
        else:
            self.is_free = self._is_free_3D
        self.voxels = data.reshape(self.dims, order='F')

        print(self.dims)

    @classmethod
    def fromVoxelMapBag(cls, filename, margin=0):
        # load messages from bagfile
        bag = rosbag.Bag(filename)
        # msgs = [msg for _, msg, _ in bag.read_messages(topics=['/fla/voxel_map'])]
        msgs = [msg for _, msg, _ in bag.read_messages(topics=['/voxel_map'])]
        bag.close()
        # save the first VoxelMap message
        resolution = msgs[0].resolution
        dims = np.array([msgs[0].dim.x, msgs[0].dim.y, msgs[0].dim.z]).astype(int)
        origin = np.array([msgs[0].origin.x, msgs[0].origin.y, msgs[0].origin.z])
        return cls(resolution, origin, dims, np.array(msgs[0].data), margin)

    def get_indices(self, point):
        return np.floor((point - self.origin) / self.resolution).astype(int)

    def get_voxel_center(self, indices):
        return self.resolution * (indices + .5) + self.origin

    def get_bounds(self):
        """
        return an array representing the bounds of map 

        Output:
            bounds, array in form [x_min, y_min, [z_min], x_max, y_max, [z_max]]
        """
        return np.hstack((np.zeros(len(self.dims)), self.resolution * self.dims))

    def is_in_bounds(self, indices):
        if any(indices < 0) or any((self.dims - indices) < 0):
            return False
        else:
            return True

    def is_valid_position(self, point):
        min = self.get_indices(point - self.margin)
        max = self.get_indices(point + self.margin)
        if not self.is_in_bounds(min) or not self.is_in_bounds(max) \
                or not self.is_free(min, max):
            return False
        else:
            return True

    def collision_free(self, mp, offset, plot=False):
        """
        Function to check if there is a collision between a motion primitive
        trajectory and the occupancy map

        Input:
            mp, a MotionPrimitive object to be checked
            offset, offset for starting point

        Output:
            collision, boolean that is True if there were no collisions
        """
        # TODO make number of points a parameter to pass in here
        _, samples, _, _, _, = mp.get_sampled_states()
        for sample in samples.T + offset[:len(self.dims)]:
            # TODO this is inefficient take out plotting stuff after its tested
            if plot:
                indices = self.get_indices(sample)
            if not self.is_valid_position(sample):
                if plot:
                    plt.plot(indices[0], indices[1], 'or')
                    self.get_voxel_center(indices)
                return False
            if plot:
                plt.plot(indices[0], indices[1], 'og')
        return True

    def plot(self, bounds=None):
        if len(self.dims) == 2:
            if bounds:
                upper_l = self.get_indices(np.array([bounds[0], bounds[3]]))
                lower_r = self.get_indices(np.array([bounds[1], bounds[2]]))
                im = self.voxels[upper_l[0]:lower_r[0], upper_l[1]:lower_r[1]].T
            else:
                im = self.voxels.T
            plt.imshow(im, cmap=plt.cm.gray_r, extent=bounds)

    # ----------------------  PRIVATE FUNCTIONS -----------------------------

    def _is_free_2D(self, min_indices, max_indices):
        if (self.voxels[min_indices[0]:max_indices[0],
                        min_indices[1]:max_indices[1]] == 0).all():
            return True
        else:
            return False

    def _is_free_3D(self, min_indices, max_indices):
        if (self.voxels[min_indices[0]:max_indices[0],
                        min_indices[1]:max_indices[1],
                        min_indices[2]:max_indices[2]] == 0).all():
            return True
        else:
            return False


if __name__ == "__main__":
    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup occupancy map
    occ_map = OccupancyMap.fromVoxelMapBag('trees_dispersion_0.6_1.bag', 1)
    occ_map.plot()

    # setup sample motion primitive
    start_position = occ_map.get_voxel_center(np.array([300, 100]))
    start_state = np.zeros((num_dims * control_space_q,))
    start_state[:num_dims] = start_position[:num_dims]
    end_position = occ_map.get_voxel_center(np.array([400, 150]))
    end_state = np.zeros((num_dims * control_space_q,))
    end_state[:num_dims] = end_position[:num_dims]
    max_state = 100 * np.ones((num_dims * control_space_q,))
    from motion_primitives_py.polynomial_motion_primitive import PolynomialMotionPrimitive
    mp = PolynomialMotionPrimitive(start_state, end_state, num_dims, max_state)

    # check collision
    if mp.is_valid:
        print(occ_map.collision_free(mp, np.array([0, 0, 0]), plot=True))

    plt.show()
