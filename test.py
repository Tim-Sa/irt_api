import unittest

from itertools import combinations_with_replacement
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd

from irt import prepare, predict, estimated_values, irt
from utils import open_xlsx 


class TestIrt(unittest.TestCase):

    def test_irt_ones_percent(self):
        one_parts = np.linspace(0,1,11)
        for one_percent in one_parts:
            predict(np.random.binomial(1, one_percent, size=(3, 4)))

    def test_irt_input_matrix_shapes(self):
        shape_borders = list(range(100))
        shapes = combinations_with_replacement(shape_borders, 2)
        for shape in shapes:
            predict(np.random.binomial(1, 0.5, size=shape))

    def test_irt_pipeline(self):
            true_abilities = np.array([4.72, 3.29, 1.98, 1.98, 1.98, 0.71, -0.45, -2.24, -3.03])
            true_difficult = np.array([-2.85, -2.85, -2.85, -2.85, -1.40, -0.21, 2.40, 1.59, 4.51, 4.51])
            df = open_xlsx('test.xlsx')
            abilities, difficult, err = irt(df, steps = 9)
            try:
                assert_almost_equal(abilities, true_abilities, decimal=1)
                assert_almost_equal(difficult, true_difficult, decimal=1)
            except AssertionError as e:
                self.assert_(False, msg=str(e))
            self.assertLessEqual(err, 0.02)

    def test_irt_estimated_values(self):
        true_ev_matrix = [[0.04742587, 0.01798621, 0.00669285],
                          [0.11920292, 0.04742587, 0.01798621],
                          [0.26894142, 0.11920292, 0.04742587]]
        ability = np.array([1, 2, 3])        
        diff = np.array([4, 5, 6])
        ev_matrix = estimated_values(ability, diff)
        self.assertTrue(np.allclose(ev_matrix, true_ev_matrix))


if __name__ == '__main__':
    unittest.main()