import numpy as np
import pytest
from copy import deepcopy

# TODO test graph search termination
def test_search(search_fixture):
    search_fixture.run_graph_search()
    assert search_fixture.succeeded is True
    search_fixture.make_graph_search_animation()


def test_fail_search(fail_search_fixture):
    fail_search_fixture.run_graph_search()
    assert fail_search_fixture.succeeded is False


if __name__ == '__main__':
    pytest.main(["-v", "--disable-pytest-warnings", __file__])
