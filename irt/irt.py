import numpy as np
import pandas as pd
from pandas import DataFrame
from utils import df_consist_only_of


class IrtResult:
    """Contain logits of tasks and subjects from IRT 0 model.
       Also contain error and rejected units. 
       Rejected units (subjects or tasks) - units wich cannot be used to 
       calculate logits.
       
    Attributes:
            abilities (pandas.Series): Logits of subjects abilities (subjects is index of Series).
            difficult (pandas.Series): Logits of task difficult (tasks is index of Series).
            err (float): Metric of difference between real test results and estimated results by logits.
            rejected_tasks (list): Name of tasks wich cannot be used to calculate difficult logits.
            rejected_subjects (list): Name of subjects wich cannot be used to calculate ability logits.
    """
    def __init__(self, abilities: pd.Series, 
                       difficult: pd.Series, 
                       err: float, 
                       rejected_tasks: list, 
                       rejected_subjects: list):
        self.abilities = abilities
        self.difficult = difficult
        self.err = err
        self.rejected_tasks = rejected_tasks
        self.rejected_subjects = rejected_subjects



def prepare(df: DataFrame):
    '''
    Iterative remove consisting only of zeros or only ones columns and rows.

        Parameters:
                df (DataFrame): A matrix with zeros and ones values only.

        Returns:
                df (DataFrame): Filtered matrix.
                rejected_subjects (list): List of subjects, who only have zero or only one results.
                rejected_tasks (list): List of tasks, who got only zero  or only one results.
    '''
    # Returns True if DataFrame row contains only zeros.
    check_row_zeros = lambda idx: df.loc[idx].sum() == 0
    # Returns True if DataFrame row contains only ones.
    check_row_ones = lambda idx: df.loc[idx].sum() >= len(df.loc[idx])
    # Returns True if DataFrame row contains only same valuses (0 or 1).
    check_subject = lambda subject: check_row_zeros(subject) or check_row_ones(subject)

    # Returns True if DataFrame column contains only zeros.
    check_col_zeros = lambda idx: df[idx].sum() == 0
    # Returns True if DataFrame column contains only ones.
    check_col_ones = lambda idx: df[idx].sum() >= len(df[idx])
    # Returns True if DataFrame column contains only same valuses (0 or 1).
    check_task = lambda task: check_col_zeros(task) or check_col_ones(task)

    rejected_subjects = []
    rejected_tasks = []

    # cycle while found rejected.
    while True:
        subjects = df.index
        tasks = df.columns

        # check for bad test units.
        rej_subjects = list(filter(check_subject, subjects))
        rej_tasks = list(filter(check_task, tasks))

        # stop if bad units not found.
        if len(rej_subjects) == 0 and len(rej_tasks) == 0:
            break
        
        # must return info about all bad units.
        rejected_subjects += rej_subjects
        rejected_tasks += rej_tasks
        # remove bad units from DataFrame.
        df = df.drop(index=rej_subjects)
        df = df.loc[:, ~df.columns.isin(rej_tasks)]

        rej_subjects = []
        rej_tasks = []

    return df, rejected_subjects, rejected_tasks    


def irt(df: DataFrame, steps = None, accept = 0.02):
    '''
    Calculation of scores for the subjects of their tasks 
    in the form of logits on the IRT model.

        Parameters:
                df (DataFrame): A matrix with zeros and ones values only.
        
        Returns:
                result (IrtResult): object with logit vectors, 
                                    rejected subjects and tasks, 
                                    model error.
    '''
    # DataFrame must contain only ones and zeros values.
    if not df_consist_only_of(df, set([0, 1])):
        raise ValueError
    
    # Remove zeros Series from input data.
    df, rejected_subjects, rejected_tasks = prepare(df)
    tasks = df.columns
    subjects = df.index

    matrix = df.to_numpy()
    # Get prediction about test subjects ability and task diffiult.  
    ability, difficult = predict(matrix)
    # Shift vector of difficults to zero mean.
    bias_difficult = difficult - difficult.mean()
    # TODO: save prev result, return it if next iter gets some nans
    # learning
    if steps:
        for _ in range(steps):
            ability, bias_difficult, err = learn_step(matrix, 
                                                      ability, 
                                                      bias_difficult)
    else:
        while True:
            ability, bias_difficult, err = learn_step(matrix, 
                                                      ability, 
                                                      bias_difficult)
            if err <= accept:
                break
    
    # Concatenate logits and units.
    ability = pd.Series(ability, subjects).to_dict()
    bias_difficult = pd.Series(bias_difficult, tasks).to_dict()

    return IrtResult(ability, bias_difficult, 
                     err, rejected_tasks, rejected_subjects)


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


def learn_step(matrix, ability, bias_difficult):
    '''
    Calculate subjects ability and task difficult in test as logits. 
    Accepts test results in binary form as input.

        Parameters:
                matrix (2d np.array): Original test matrix.
                ability (1d np.array): Binary array of subjects ability from previous step.
                difficult (1d np.array): Binary array of tasks difficult from previous step.

        Returns:
                ability (1d np.array): Binary array of subjects ability.
                difficult (1d np.array): Binary array of tasks difficult. 
                err (float): difference metric between estimated values and original matrix.
    '''

    # Build estimated data by ability and difficult logits
    ev = estimated_values(ability, bias_difficult)
    # Dispersion of estimated values.
    ability_err, diff_err = dispersion(ev)
    ability_diff, diff_diff = logits_difference(matrix, ev)
    # Try to minimize this value getting more quality logits.
    err = np.sum(ability_diff * ability_diff)
    # get new logits.
    ability = ability - (ability_diff / ability_err)
    difficult = bias_difficult - (diff_diff / diff_err)
    # Set logits of difficult average to zero.
    bias_difficult = difficult - difficult.mean()

    return ability, bias_difficult, err


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
    difference = prev_matrix - ev
    ability_diff = np.sum(difference, axis=1)
    diff_diff = -1 * np.sum(difference, axis=0)
    return ability_diff, diff_diff

