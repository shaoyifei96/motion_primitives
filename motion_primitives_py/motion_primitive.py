import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class MotionPrimitive():
    """
    # WIP
    A motion primitive that defines a trajectory from a over a time T. 
    Put functions that all MPs should have in this base class. 
    If the implementation is specific to the subclass, raise a NotImplementedError
    """

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={'dynamics':None}):
        """
        """
        self.start_state = np.array(start_state)
        self.end_state = np.array(end_state)
        self.num_dims = np.array(num_dims)
        self.max_state = np.array(max_state)
        # TODO probably cleaner to do this with kwargs
        self.subclass_specific_data = deepcopy(subclass_specific_data)
        self.control_space_q = int(self.start_state.shape[0]/num_dims)
        self.is_valid = False
        self.cost = None

    @classmethod
    def from_dict(cls, dict, num_dims, max_state):
        """
        load a motion primitive from a dictionary
        """
        if dict:
            mp = cls.__new__(cls)
            super(cls, mp).__init__(np.array(dict["start_state"]),
                                    np.array(dict["end_state"]),
                                    num_dims, max_state)
            mp.cost = dict["cost"]
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
        Return a sampling of the trajectory for plotting 
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def get_sampled_position(self, step_size=0.1):
        """
        Return a sampling of only position of trajectory for plotting 
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def get_sampled_input(self, step_size=0.1):
        """
        Return a sampling of inputs needed to move along the mp
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def plot_from_sampled_states(self, st, sp, sv, sa, sj, position_only=False, ax=None, start_position_override=None, color=None):
        """
        Plot time vs. position, velocity, acceleration, and jerk (input is already sampled)
        """
        # Plot the state over time.
        if color is None:
            color = 'lightgrey'
        if not position_only:
            fig, axes = plt.subplots(4, 1, sharex=True)
            samples = [sp, sv, sa, sj]
            axes[3].set_xlabel('time')
            fig.suptitle('Full State over Time')
        else:
            if ax == None:
                axes = [plt.gca()]
            else:
                axes = [ax]
            samples = [sp[0, :], sp[1, :]]
        for i in range(sp.shape[0]):
            for ax, s, l in zip(axes, samples, ('pos', 'vel', 'acc', 'jerk')):
                if s is not None:
                    if position_only:
                        if start_position_override is  None:
                            start_position_override = self.start_state
                        ax.plot(samples[0]+start_position_override[0]-self.start_state[0], samples[1]+start_position_override[1]-self.start_state[1],color=color)
                    else:
                        ax.plot(st, s[i, :])
                ax.set_ylabel(l)

    def plot(self, position_only=False, ax=None, start_position_override=None, color=None):
        """
        Generate the sampled state and input trajectories and plot them
        """
        st, sp, sv, sa, sj = self.get_sampled_states()
        if st is not None:
            self.plot_from_sampled_states(st, sp, sv, sa, sj, position_only, ax, start_position_override, color)
        else:
            print("Trajectory was not found")

    def __eq__(self, other):
        if other is None:
            if self.__dict__.keys()  == None:
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
