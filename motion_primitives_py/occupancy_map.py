import rosbag
import numpy as np
import matplotlib.pyplot as plt

# from motion_primitives_py.msg import VoxelMap


class OccupancyMap():
    def __init__(self, resolution, origin, dims, data, margin=0):
        self.resolution = resolution
        self.voxels = np.squeeze(data.reshape(dims, order='F'))
        self.dims = np.array(self.voxels.shape)
        self.origin = origin[:len(self.dims)]
        map_size = self.dims*self.resolution
        map_min = self.origin
        map_max = map_size + self.origin
        self.extent = [map_min[0], map_max[0], map_min[1], map_max[1]]
        # distance margin to inflate obstacles by
        self.margin = margin

    @classmethod
    def fromVoxelMapBag(cls, filename, topic=None, margin=0):
        # load messages from bagfile
        bag = rosbag.Bag(filename)
        msgs = [msg for _, msg, _ in bag.read_messages(topics=topic)]
        bag.close()
        resolution = msgs[0].resolution
        dims = np.array([msgs[0].dim.x, msgs[0].dim.y, msgs[0].dim.z]).astype(int)
        origin = np.array([msgs[0].origin.x, msgs[0].origin.y, msgs[0].origin.z])
        return cls(resolution, origin, dims, np.array(msgs[0].data), margin)

    def get_indices_from_position(self, point):
        return np.floor((point - self.origin) / self.resolution).astype(int)

    def get_voxel_center_from_indices(self, indices):
        return self.resolution * (indices + .5) + self.origin

    def is_valid_indices(self, indices):
        for i in range(len(self.dims)):
            if indices[i] < 0 or (self.dims[i] - indices[i]) <= 0:
                return False
        else:
            return True

    def is_valid_position(self, position):
        return self.is_valid_indices(self.get_indices_from_position(position))

    def is_free_and_valid_indices(self, indices):
        if not self.is_valid_indices(indices) or not self.voxels[tuple(indices)] == 0:
            return False
        else:
            return True

    def is_free_and_valid_position(self, position):
        indices = self.get_indices_from_position(position)
        return self.is_free_and_valid_indices(indices)

    def is_mp_collision_free(self, mp, step_size=0.1, offset=None):
        """
        Function to check if there is a collision between a motion primitive
        trajectory and the occupancy map

        Input:
            mp, a MotionPrimitive object to be checked
            offset, offset for starting point

        Output:
            collision, boolean that is True if there were no collisions
        """
        if not mp.is_valid:
            return False
        if offset is None:
            offset = mp.start_state
        _, samples = mp.get_sampled_position(step_size)
        for sample in samples.T + offset[:mp.num_dims] - mp.start_state[:mp.num_dims]:
            if not self.is_free_and_valid_position(sample):
                return False
        return True

    def plot(self, ax=None):
        if len(self.dims) == 2:
            if ax == None:
                fig, self.ax = plt.subplots()
                ax = self.ax
                # if len(self.dims) == 3:
                #     fig_3d, ax_3d = plt.subplots()
            im = self.voxels.T
            ax.pcolormesh(np.arange(self.voxels.shape[0]+1)*self.resolution, np.arange(self.voxels.shape[1]+1)
                          * self.resolution, self.voxels.T, cmap='Greys', zorder=1)
            ax.add_patch(plt.Rectangle(self.origin, self.extent[1], self.extent[3], ec='r', fill=False))
            ax.set_aspect('equal')


if __name__ == "__main__":
    from motion_primitives_py import *

    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup occupancy map
    occ_map = OccupancyMap.fromVoxelMapBag('trees_dispersion_0.6_1.bag', 0)
    occ_map.plot()
    print(occ_map.extent)

    # setup sample motion primitive
    start_state = np.zeros((num_dims * control_space_q,))
    start_state[:2] = [18, 5]
    end_state = np.zeros((num_dims * control_space_q,))
    end_state[:2] = [10, 8]
    max_state = 1000 * np.ones((num_dims * control_space_q,))
    mp = PolynomialMotionPrimitive(start_state, end_state, num_dims, max_state)
    mp.plot(position_only=True, ax=occ_map.ax)

    print(occ_map.is_free_and_valid_position([70, 5]))
    plt.show()
