
from motion_primitives_py import MotionPrimitive
import numpy as np
import matplotlib.pyplot as plt
from ruckig import InputParameter, OutputParameter, Result, Ruckig

class RuckigMotionPrimitive(MotionPrimitive):

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        super().__init__(start_state, end_state, num_dims, max_state, subclass_specific_data)
        assert(self.control_space_q == 3), "This function only works for jerk input space"
        self.run_ruckig()

    def run_ruckig(self):
        inp = InputParameter(self.num_dims)
        inp.current_position = self.start_state[:self.num_dims]
        inp.current_velocity = self.start_state[self.num_dims:2 * self.num_dims]
        inp.current_acceleration = self.start_state[2*self.num_dims:3 * self.num_dims]

        inp.target_position = self.end_state[:self.num_dims]
        inp.target_velocity = self.end_state[self.num_dims:2 * self.num_dims]
        inp.target_acceleration = self.end_state[2*self.num_dims:3 * self.num_dims]

        inp.max_velocity = np.repeat(self.max_state[1], self.num_dims)
        inp.max_acceleration = np.repeat(self.max_state[2], self.num_dims)
        inp.max_jerk = np.repeat(self.max_state[3], self.num_dims)

        ruckig = Ruckig(self.num_dims, 0.05)

        out = OutputParameter(self.num_dims)
        first_output = out

        res = ruckig.update(inp, out)

        # print(f'Calculation duration: {first_output.calculation_duration:0.1f} [µs]')
        # print(f'Trajectory duration: {first_output.trajectory.duration:0.4f} [s]')

        self.traj_time = first_output.trajectory.duration
        if self.traj_time < 1e-10:
            self.is_valid = False
        else:
            self.is_valid = True
        self.cost = self.traj_time
        self.ruckig_trajectory = first_output.trajectory

    @classmethod
    def from_dict(cls, dict, num_dims, max_state, subclass_specific_data={}):
        mp = super().from_dict(dict, num_dims, max_state)
        if mp:
            mp.run_ruckig()
        return mp

    def to_dict(self):
        dict = super().to_dict()
        return dict

    def get_state(self, t):
        pos, vel, acc = self.ruckig_trajectory.at_time(t)
        return np.hstack((pos, vel, acc))

    def get_sampled_states(self, step_size=0.1):
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            sampled_array = np.empty((1+self.n, st.shape[0]))
            sampled_array[0, :] = st
            for i, t in enumerate(st):
                sampled_array[1:, i] = self.get_state(t)
            return sampled_array
        return None

    def get_sampled_position(self, step_size=0.1):
        sampled_array = self.get_sampled_states(step_size)
        return sampled_array[0,:], sampled_array[1,:]

    # def get_sampled_input(self, step_size=None):


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup problem
    start_state = np.zeros((num_dims * control_space_q,))
    end_state = np.random.rand(num_dims * control_space_q,)
    max_state = 100 * np.ones((control_space_q+1))

    # jerks
    mp = RuckigMotionPrimitive(start_state, end_state, num_dims, max_state)
    print(mp.get_state(.4))
    # plot
    sampling_array = mp.get_sampled_states()
    mp.plot_from_sampled_states(sampling_array)
    plt.show()
