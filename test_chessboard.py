import pytest
import sys
sys.path.append('../hungry_horse')

from chessboard import Chessboard


chessboard_parameters_1 = {"width": 2, "height": 2}
chessboard_parameters_2 = {"width": 0, "height": 0}
chessboard_parameters_3 = {"width": 1, "height": 1, "step_cost": -5}
chessboard_1 = Chessboard(8, 8)
chessboard_2 = Chessboard(3, 2)
chessboard_3 = Chessboard(1, 1, -5)


@pytest.mark.parametrize("chessboard_parameters, expected_chessboard",
                         [(chessboard_parameters_1, {(0, 0): -1, (0, 1): -1, (1, 1): -1, (1, 0): -1}),
                          (chessboard_parameters_2, {}),
                          (chessboard_parameters_3, {(0, 0): -5})])
def test_create_rectangle_chessboard(chessboard_parameters, expected_chessboard):
    created_chessboard = Chessboard(**chessboard_parameters)
    assert created_chessboard.states_rewards == expected_chessboard
