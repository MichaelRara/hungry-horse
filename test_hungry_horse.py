import sys
sys.path.append('../hungry_horse')

from hungry_horse import *
import pytest


@pytest.mark.parametrize("width, height, expected",
                         [(0, 0, []),
                          (1, 1, [(0, 0)]),
                          (1, 2, [(0, 0), (0, 1)])])
def test_create_rectangle_chessboard(width, height, expected):
    created_chessboard = create_rectangle_chessboard(width, height)
    assert created_chessboard == expected
