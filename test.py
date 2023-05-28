import unittest

from itertools import combinations_with_replacement
import numpy as np

from irt import iter_step


class TestIrt(unittest.TestCase):

    def test_irt_ones_percent(self):
        one_parts = np.linspace(0,1,11)
        for one_percent in one_parts:
            iter_step(np.random.binomial(1, one_percent, size=(3, 4)))

    def test_irt_input_matrix_shapes(self):
        shape_borders = list(range(100))
        shapes = combinations_with_replacement(shape_borders, 2)
        for shape in shapes:
            iter_step(np.random.binomial(1, 0.5, size=shape))


if __name__ == '__main__':
    unittest.main()