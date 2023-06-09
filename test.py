import unittest

from itertools import combinations_with_replacement
import numpy as np
import pandas as pd

from irt import prepare, predict, estimated_values


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
            df = pd.DataFrame({
                'task1': [1, 1, 0, 0],
                'task2': [0, 0, 1, 0],
                'task3': [1, 0, 1, 0],
                'task4': [0, 0, 0, 0]    
            }, index=['subj1', 'subj2', 'subj3', 'subj4'])
            true_abilities = np.array([0.69314718, 
                                       -0.69314718,  
                                       0.69314718])
            true_difficult = np.array([-0.69314718, 
                                       0.69314718,  
                                       -0.69314718])
            df, _, _ = prepare(df)
            abilities, difficult = predict(df)
            self.assertTrue(np.allclose(abilities, true_abilities))
            self.assertTrue(np.allclose(difficult, true_difficult))

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