import numpy as np
import pandas as pd
from pandas import DataFrame


def prepare(df: DataFrame):
    '''
    Remove consisting only of zeros columns and rows.

        Parameters:
                df (DataFrame): A matrix with zeros and ones values only.

        Returns:
                df (DataFrame): Filtered matrix.
                rejected_subjects (list): List of subjects, who only have zero results.
                rejected_tasks (list): List of tasks, who got only zero results.
    '''
    subjects = df.index
    tasks = df.columns
    # Remove subjects with only zeros.
    check_subject = lambda subject: df.loc[subject].sum() == 0
    rejected_subjects = list(filter(check_subject, subjects))
    df = df.drop(index=rejected_subjects)
    # Remove tasks with only zeros.
    check_task = lambda task: df[task].sum() == 0
    rejected_tasks = list(filter(check_task, tasks))
    df = df.loc[:, ~df.columns.isin(rejected_tasks)]

    # TODO: remove Series with ones only.

    return df, rejected_subjects, rejected_tasks    


def irt(df: DataFrame):
    '''
    Calculation of scores for the subjects of their tasks 
    in the form of logits on the IRT model.

        Parameters:
                df (DataFrame): A matrix with zeros and ones values only.
        
        Returns:
                ability (1d np.array): Binary array of subjects ability.
                difficult (1d np.array): Binary array of tasks difficult.
    '''
    # Remove zeros Series from input data.
    df, _, _ = prepare(df)
    matrix = df.to_numpy()
    # Get prediction about test subjects ability and task diffiult.  
    ability, difficult = predict(matrix)
    # Shift vector of difficults to zero mean.
    bias_difficult = difficult - difficult.mean()


    # Build estimated data by ability and difficult logits
    ev = estimated_values(ability, bias_difficult)
    # Dispersion of estimated values.
    ability_err, diff_err = dispersion(ev)
    ability_diff, diff_diff = logits_difference(matrix, ev)
    # Try to minimize this value getting more quality logits.
    err = np.sum(ability_diff ** 2)
    # get new logits.
    ability = ability - (ability_diff / ability_err)
    difficult = difficult - (diff_diff / diff_err)
    bias_difficult = difficult - difficult.mean()

    return predict(ev)


def predict(matrix: np.array):
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


def estimated_values(ability: np.array, bias_difficult: np.array) -> np.array: 
    '''
    Calculate matrix of estimated values, step of Rasch model learning.
        
        Parameters:
                ability (1d np.array): Vector of predicted subjects ability.
                bias_difficult (1d np.array): Vector of predicted task difficult.
                                              The average of this vector must be almost zero.

        Returns: 
                ev_matrix (2d np.array): Matrix of estimated values.
            
    '''   
    ev_matrix = np.array([ability])
    for dif_value in bias_difficult:
        diff_exp = np.exp(ability - dif_value)
        ev_vec = diff_exp / (1 + diff_exp)
        ev_matrix = np.concatenate((ev_matrix, [ev_vec]), axis=0)
    return ev_matrix[1::].T


def dispersion(ev: np.array):
    ev_dispersion = ev * (1-ev)
    ability_err = -1 * np.sum(ev_dispersion, axis=1)
    diff_err = -1 * np.sum(ev_dispersion, axis=0)
    return ability_err, diff_err


def logits_difference(prev_matrix, ev):
    difference = matrix - ev
    ability_diff = np.sum(difference, axis=1)
    diff_diff = np.sum(difference, axis=0)
