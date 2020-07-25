
from py_opt_control import min_time_bvp
from motion_primitives_py import c_output_redirector
import io
from motion_primitives_py.motion_primitive import *


class JerksMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from a sequence of constant jerks
    """

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        super().__init__(start_state, end_state, num_dims, max_state, subclass_specific_data)
        assert(self.control_space_q == 3), "This function only works for jerk input space (and maybe acceleration input space one day)"

        # start point # ugly but this is faster than using np.split
        p0, v0, a0 = self.start_state[:self.num_dims], self.start_state[self.num_dims:2 *
                                                                        self.num_dims], self.start_state[2*self.num_dims:3*self.num_dims]
        # end point
        p1, v1, a1 = self.end_state[:self.num_dims], self.end_state[self.num_dims:2 *
                                                                    self.num_dims], self.end_state[2*self.num_dims:3*self.num_dims]
        # state and input limits
        v_max, a_max, j_max = self.max_state[1:1+self.control_space_q] + 1e-5  # numerical hack for library seg fault
        v_min, a_min, j_min = -self.max_state[1:1+self.control_space_q] - 1e-5

        f = io.BytesIO()
        with c_output_redirector.stdout_redirector(f):  # Suppress warning/error messages from C library
            self.switch_times, self.jerks = min_time_bvp.min_time_bvp(p0, v0, a0, p1, v1, a1, v_min, v_max, a_min, a_max, j_min, j_max)
        traj_time = np.max(self.switch_times[:, -1])
        self.is_valid = (self.get_state(traj_time) - self.end_state < 1e-5).all()
        if self.is_valid:
            self.cost = traj_time

    @classmethod
    def from_dict(cls, dict, num_dims, max_state):
        """
        load a jerks representation of the motion primitive from a dictionary 
        """
        mp = super().from_dict(dict, num_dims, max_state)
        if mp:
            mp.switch_times = np.array(dict["switch_times"])
            mp.jerks = np.array(dict["jerks"][0])
        return mp

    def to_dict(self):
        """
        Write important attributes of motion primitive to a dictionary
        """
        dict = super().to_dict()
        if dict:
            dict["jerks"] = self.jerks.tolist(),
            dict["switch_times"] = self.switch_times.tolist()
        return dict

    def get_state(self, t):
        """
        Evaluate full state of a trajectory at a given time
        Input:
            t, numpy array of times to sample at
        Return:
            state, a numpy array of size (num_dims x 4), ordered (p, v, a, j)
        """

        # call to optimization library to evaluate at time t
        sj, sa, sv, sp = min_time_bvp.sample(self.start_state[:self.num_dims], self.start_state[self.num_dims:2 *
                                                                                                self.num_dims], self.start_state[2*self.num_dims:3*self.num_dims], self.switch_times, self.jerks, t)
        return np.squeeze(np.concatenate([sp, sv, sa]))  # TODO concatenate may be slow because allocates new memory

    def get_sampled_states(self):
        p0, v0, a0 = np.split(self.start_state, self.control_space_q)
        st, sj, sa, sv, sp = min_time_bvp.uniformly_sample(p0, v0, a0, self.switch_times, self.jerks, dt=0.001)
        return st, sp, sv, sa, sj


if __name__ == "__main__":
    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup problem
    start_state = np.zeros((num_dims * control_space_q,))
    end_state = np.random.rand(num_dims * control_space_q,)
    max_state = np.ones((num_dims * control_space_q,))*100

    # jerks
    mp1 = JerksMotionPrimitive(start_state, end_state, num_dims, max_state)

    # save
    assert(mp1.is_valid)
    dictionary1 = mp1.to_dict()

    # reconstruct
    mp1 = JerksMotionPrimitive.from_dict(dictionary1, num_dims, max_state)

    # plot
    st, sp, sv, sa, sj = mp1.get_sampled_states()
    mp1.plot_from_sampled_states(st, sp, sv, sa, sj)
    plt.show()
