from motion_primitives_py import MotionPrimitive
import numpy as np
import matplotlib.pyplot as plt
import reeds_shepp


class ReedsSheppMotionPrimitive(MotionPrimitive):
    """
    Create a new Reeds-Shepp motion primitive
    """

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        assert(num_dims == 2 and start_state.shape[0] == 3), "ReedsShepp takes num_dims 2 and n 3"
        super().__init__(start_state, end_state, num_dims, max_state, subclass_specific_data)  # Run MotionPrimitive's instantiation first
        if self.subclass_specific_data.get('turning_radius') is None:
            self.turning_radius = .5
        else:
            self.turning_radius = self.subclass_specific_data.get('turning_radius')
        self.cost = reeds_shepp.path_length(self.start_state, self.end_state, self.turning_radius)
        self.is_valid = True

    @classmethod
    def from_dict(cls, dict, num_dims, max_state, subclass_specific_data={}):
        """
        Load a Reeds-Shepp motion primitive from a dictionary 
        """
        mp = super().from_dict(dict, num_dims, max_state)
        if mp:
            mp.turning_radius = dict["turning_radius"]
        return mp

    def to_dict(self):
        """
        Write important attributes of motion primitive to a dictionary
        """
        dict = super().to_dict()
        if dict:
            dict["turning_radius"] = self.turning_radius
        return dict

    def get_state(self, t):
        pass

    def get_sampled_states(self, step_size=0.1):
        ps = np.array(reeds_shepp.path_sample(self.start_state, self.end_state, self.turning_radius, step_size)).T
        p = ps[:2, :]
        v = ps[2, :][np.newaxis, :]
        ts = np.linspace(0, self.cost, int(np.ceil(self.cost/step_size)+1))
        return ts, p, v, None, None


if __name__ == "__main__":
    start_state = np.zeros(3,)
    end_state = np.random.rand(3,)
    max_state = np.ones((3,))*100

    mp1 = ReedsSheppMotionPrimitive(start_state, end_state, 2, max_state)
    dict = mp1.to_dict()
    mp1 = ReedsSheppMotionPrimitive.from_dict(dict, 2, max_state)

    st1, sx1, _, _, _ = mp1.get_sampled_states()
    plt.plot(start_state[0], start_state[1], 'go')
    plt.plot(end_state[0], end_state[1], 'ro')
    plt.plot(sx1[0, :], sx1[1, :])
    plt.show()
