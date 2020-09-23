import numpy as np
import pytest
import ompl.base
import ompl.geometric


def test_mp_search(search_fixture):
    path, sampled_path, path_cost = search_fixture.run_graph_search()
    assert search_fixture.queue
    search_fixture.make_graph_search_animation()

def test_ompl(search_fixture):
    # create a Reeds Shepp query
    space = ompl.base.ReedsSheppStateSpace()

    # set lower and upper bounds
    bounds = ompl.base.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)
    # TODO: havent gotten the specific search to work yet so this is 
    # commented out for the time being
    #bounds.setLow(0, float(search_fixture.map.extent[0]))
    #bounds.setLow(1, float(search_fixture.map.extent[2]))
    #bounds.setHigh(0, float(search_fixture.map.extent[1]))
    #bounds.setHigh(1, float(search_fixture.map.extent[3]))
    space.setBounds(bounds)

    # set start and goal locations
    start = ompl.base.State(space)
    start().setX(.5)
    #start().setX(float(search_fixture.start_state[0]))
    #start().setX(float(search_fixture.start_state[1]))
    goal = ompl.base.State(space)
    goal().setX(-.5)
    #goal().setX(float(search_fixture.goal_state[0]))
    #goal().setX(float(search_fixture.goal_state[1]))

    # create function handle to check if state is valid
    def _is_state_valid(state):
        # TODO add input limitations - this is arbitrary
        return state.getX() < .6
    state_validity_checker = ompl.base.StateValidityCheckerFn(_is_state_valid)

    # set up a simple setup object
    ss = ompl.geometric.SimpleSetup(space)
    ss.setStateValidityChecker(state_validity_checker)
    ss.setStartAndGoalStates(start, goal)
    ss.setPlanner(ompl.geometric.RRT(ss.getSpaceInformation()))

    # attempt to solve the planning problem
    solved = ss.solve(1.0)
    if solved:
        ss.simplifySolution()
        print(ss.getSolutionPath())


if __name__ == '__main__':
    pytest.main(["-v", "--disable-pytest-warnings", __file__])
