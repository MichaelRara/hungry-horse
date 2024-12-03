import pytest
import sys
sys.path.append('../hungry_horse')

from chessboard import Chessboard


@pytest.fixture
def chessboard_parameters():
    return {
        "parameters_1": {"width": 2, "height": 2},
        "parameters_2": {"width": 0, "height": 0},
        "parameters_3": {"width": 1, "height": 1, "step_cost": -5}
    }


@pytest.fixture
def expected_chessboards():
    return {
        "parameters_1": {(0, 0): -1, (0, 1): -1, (1, 1): -1, (1, 0): -1},
        "parameters_2": {},
        "parameters_3": {(0, 0): -5}
    }


@pytest.mark.parametrize("param_key", ["parameters_1", "parameters_2", "parameters_3"])
def test_create_rectangle_chessboard_2(chessboard_parameters, expected_chessboards, param_key):
    created_chessboard = Chessboard(**chessboard_parameters[param_key])
    assert created_chessboard.states_rewards == expected_chessboards[param_key]
