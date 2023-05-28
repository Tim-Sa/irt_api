import unittest

from itertools import combinations_with_replacement
import numpy as np
import pandas as pd

from irt import irt, iter_step


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
            
            abilities, difficult = irt(df)
            self.assertTrue(np.allclose(abilities, true_abilities))
            self.assertTrue(np.allclose(difficult, true_difficult))

if __name__ == '__main__':
    unittest.main()