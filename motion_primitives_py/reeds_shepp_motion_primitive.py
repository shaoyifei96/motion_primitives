from motion_primitive import MotionPrimitive
import numpy as np
import matplotlib.pyplot as plt
import reeds_shepp


class ReedsSheppMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from polynomial coefficients
    """

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        super().__init__(start_state, end_state, num_dims, max_state, subclass_specific_data)  # Run MotionPrimitive's instantiation first
        self.reeds_shepp_constructor()

    def reeds_shepp_constructor(self):
        if self.subclass_specific_data.get('turning_radius') is None:
            self.turning_radius = .5
        else:
            self.turning_radius = self.subclass_specific_data.get('turning_radius')
        self.cost = reeds_shepp.path_length(self.start_state, self.end_state, self.turning_radius)
        self.is_valid = True

    def get_state(self, t):
        pass

    def get_sampled_states(self):
        step_size = 0.1
        ps = np.array(reeds_shepp.path_sample(self.start_state, self.end_state, self.turning_radius, step_size)).T
        ts = np.arange(0, self.cost, step_size)
        return ts, ps, None, None, None


if __name__ == "__main__":
    start_state = np.zeros(3,)
    end_state = np.random.rand(3,)
    max_state = np.ones((3,))*100

    mp1 = ReedsSheppMotionPrimitive(start_state, end_state, 3, max_state)
    st1, sx1, _, _, _ = mp1.get_sampled_states()
    plt.plot(start_state[0], start_state[1], 'go')
    plt.plot(end_state[0], end_state[1], 'ro')
    plt.plot(sx1[0, :], sx1[1, :])
    plt.show()
