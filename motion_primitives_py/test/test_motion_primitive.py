import numpy as np
import pytest


similarity_threshold = 10e-5

def test_initial_state(mp_fixture):
    s0 = np.squeeze(mp_fixture.get_state(np.array([0])))
    assert ((s0 - mp_fixture.start_state) < similarity_threshold).all()

def test_final_state(mp_fixture):
    sf = np.squeeze(mp_fixture.get_state(np.array([mp_fixture.cost])))
    assert ((sf - mp_fixture.end_state) < similarity_threshold).all()

"""
def test_max_u(mp_fixture):
    for t in np.arange(0, mp_fixture.cost, .1):
        u = np.sum(mp_fixture.polys * mp_fixture.x_derivs[-2](t),axis=1)
        assert (abs(u) < mp_fixture.max_state[mp_fixture.control_space_q]).all()
"""

def test_save_load(mp_fixture):
    dictionary = mp_fixture.to_dict()
    mp2 = type(mp_fixture).from_dict(dictionary, mp_fixture.num_dims, 
                                     mp_fixture.max_state)
    states1 = mp_fixture.get_sampled_states()
    states2 = mp2.get_sampled_states()
    assert mp_fixture.cost == mp2.cost
    assert mp_fixture.is_valid == mp2.is_valid
    for i in range(int(len(mp_fixture.start_state) / mp_fixture.num_dims + 1)):
        assert (abs(states1[i] - states2[i]) < similarity_threshold).all()


if __name__ == "__main__":
    pytest.main(["-v", "--disable-pytest-warnings", __file__])
