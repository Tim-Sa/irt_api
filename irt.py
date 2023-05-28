import numpy as np
import pandas as pd
from pandas import DataFrame


def irt(df: DataFrame):
    pass


def iter_step(matrix: np.array):
    '''
    Calculate subjects ability and task difficult in test as logits. 
    Accepts test results in binary form as input.

            Parameters:
                    matrix (2d np.array): A matrix with zeros and ones values only.

            Returns:
                    ability (1d np.array): Binary array of subjects ability.
                    difficult (1d np.array): Binary array of tasks difficult.
    '''
    subjects_mean = np.mean(matrix, axis=1)
    ability = np.log(subjects_mean / (1 - subjects_mean))
    
    tasks_mean = np.mean(matrix, axis=0)
    difficult = np.log((1 - tasks_mean) / tasks_mean)

    return ability, difficult

