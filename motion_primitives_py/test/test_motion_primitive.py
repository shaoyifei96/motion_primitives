import numpy as np
import pytest


similarity_threshold = 10e-5


def test_initial_state(mp_fixture):
    s0 = np.squeeze(mp_fixture.get_state(np.array([0])))
    assert (abs(s0 - mp_fixture.start_state) < similarity_threshold).all()


def test_final_state(mp_fixture):
    sf = np.squeeze(mp_fixture.get_state(np.array([mp_fixture.traj_time])))
    assert (abs(sf - mp_fixture.end_state) < similarity_threshold).all()


def test_max_u(mp_fixture):
    _, su = mp_fixture.get_sampled_input()
    assert (abs(su) < mp_fixture.max_state[mp_fixture.control_space_q]).all()


def test_max_state(mp_fixture):
    _, _, sv, sa, sj = mp_fixture.get_sampled_states()
    assert (abs(sv) < mp_fixture.max_state[1]).all()
    assert (abs(sa) < mp_fixture.max_state[2]).all()
    assert (abs(sj) < mp_fixture.max_state[3]).all()


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
