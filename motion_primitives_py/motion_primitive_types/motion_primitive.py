import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class MotionPrimitive():
    """
    A motion primitive that defines a trajectory from a over a time T. 
    Put functions that all MPs should have in this base class. 
    If the implementation is specific to the subclass, raise a NotImplementedError
    """

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={'dynamics': None}):
        """
        """
        self.start_state = np.array(start_state).astype(float)
        self.end_state = np.array(end_state).astype(float)
        self.num_dims = np.array(num_dims).astype(int)
        self.max_state = np.array(max_state).astype(float)
        # TODO probably cleaner to do this with kwargs
        self.subclass_specific_data = deepcopy(subclass_specific_data)
        self.n = self.start_state.shape[0]
        self.control_space_q = int(self.n/num_dims)
        self.is_valid = False
        self.cost = None
        self.traj_time = None
        #TODO error out if start/end state violate max_state

    @classmethod
    def from_dict(cls, dict, num_dims, max_state):
        """
        load a motion primitive from a dictionary
        """
        if dict:
            mp = cls.__new__(cls)
            MotionPrimitive.__init__(mp, np.array(dict["start_state"]),
                                     np.array(dict["end_state"]),
                                     num_dims, max_state)
            mp.cost = dict["cost"]
            mp.traj_time = dict["traj_time"]
            mp.is_valid = True
        else:
            mp = None
        return mp

    def to_dict(self):
        """
        Write important attributes of motion primitive to a dictionary
        """
        if self.is_valid:
            dict = {"cost": self.cost,
                    "traj_time": self.traj_time,
                    "start_state": self.start_state.tolist(),
                    "end_state": self.end_state.tolist(),
                    }
        else:
            dict = {}
        return dict

    def get_state(self, t):
        """
        Given a time t, return the state of the motion primitive at that time. 
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def get_sampled_states(self, step_size=0.1):
        """
        Return an array consisting of sample times and a sampling of the trajectory for plotting 
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def get_sampled_position(self, step_size=0.1):
        """
        Return a sampling of only position of trajectory
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def get_input(self, t):
        """
        Given a time t, return the input of the motion primitive at that time. 
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def get_sampled_input(self, step_size=0.1):
        """
        Return a sampling of inputs needed to move along the mp
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def plot_from_sampled_states(self, sampling_array, position_only=False, ax=None, color=None, zorder=1):
        """
        Plot time vs. position, velocity, acceleration, and jerk (input is already sampled)
        """
        # Plot the state over time.
        if color is None:
            color = 'lightgrey'
        if not position_only:
            fig, axes = plt.subplots(4, 1, sharex=True)
            axes[3].set_xlabel('time')
            fig.suptitle('Full State over Time')
            for j in range(4):
                for i in range(self.num_dims):
                    if j*self.num_dims+i+1 >= sampling_array.shape[0]:
                        break
                    axes[j].plot(sampling_array[0,:],sampling_array[j*self.num_dims+i+1,:])

        else:
            if ax is None:
                ax = plt.gca()
            samples = sampling_array[1:1+self.num_dims, :]
            if self.num_dims==2:
                ax.plot(samples[0,:], samples[1,:], color=color, zorder=zorder)
            elif self.num_dims==3:
                ax.plot(samples[0,:], samples[1,:], samples[2,:], color=color, zorder=zorder)

    def plot(self, position_only=False, ax=None, color=None, zorder=1):
        """
        Generate the sampled state and input trajectories and plot them
        """
        sampling_array = self.get_sampled_states()
        if sampling_array is not None:
            self.plot_from_sampled_states(sampling_array, position_only, ax, color, zorder)
        else:
            print("Trajectory was not found")

    def __eq__(self, other):
        if other is None:
            if self.__dict__.keys() == None:
                return True
            else:
                return False
        if self.__dict__.keys() != other.__dict__.keys():
            return False
        return all(np.array_equal(self.__dict__[key], other.__dict__[key]) for key in self.__dict__)

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __ge__(self, other):
        return self.cost >= other.cost