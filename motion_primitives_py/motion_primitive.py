import numpy as np
import matplotlib.pyplot as plt


class MotionPrimitive():
    """
    # WIP
    A motion primitive that defines a trajectory from a over a time T. 
    Put functions that all MPs should have in this base class. 
    If the implementation is specific to the subclass, raise a NotImplementedError
    """

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        """
        """
        self.start_state = np.array(start_state)
        self.end_state = np.array(end_state)
        self.num_dims = np.array(num_dims)
        self.max_state = np.array(max_state)
        # TODO probably cleaner to do this with kwargs
        self.subclass_specific_data = subclass_specific_data
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

    def get_sampled_states(self):
        """
        Return a sampling of the trajectory for plotting 
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        raise NotImplementedError

    def plot_from_sampled_states(self, st, sp, sv, sa, sj, position_only=False, ax=None):
        """
        Plot time vs. position, velocity, acceleration, and jerk (input is already sampled)
        """
        # Plot the state over time.
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
                        ax.plot(samples[0], samples[1])
                    else:
                        ax.plot(st, s[i, :])
                ax.set_ylabel(l)

    def plot(self, position_only=False, ax=None):
        """
        Generate the sampled state and input trajectories and plot them
        """
        st, sp, sv, sa, sj = self.get_sampled_states()
        if st is not None:
            self.plot_from_sampled_states(st, sp, sv, sa, sj, position_only, ax)
        else:
            print("Trajectory was not found")

    def __eq__(self, other):
        if self.__dict__.keys() != other.__dict__.keys():
            return False
        return all(np.array_equal(self.__dict__[key], other.__dict__[key]) for key in self.__dict__)

    def __lt__(self, other):
        return self.cost < other.cost
    
    def __gt__(self, other):
        return self.cost > other.cost
