import numpy as np
import pytest


def test_search(search_fixture):
    path, sampled_path, path_cost = search_fixture.run_graph_search()
    assert search_fixture.queue
    search_fixture.make_graph_search_animation()


if __name__ == '__main__':
    pytest.main(["-v", "--disable-pytest-warnings", __file__])
