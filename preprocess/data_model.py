import sys
sys.path.append("./")
from utils import logger

def get_sparse_dict(hadron):
    """ Get dictionary mapping variable names to their respective axis indices in the sparse.
    Args:
        hadron (str): Type of D meson ('Dzero', 'Dplus', 'Ds').
    Returns:
        sparse_dict (dict): Dictionary mapping variable names to axis indices.
    """

    if hadron == 'Dzero':
        return {
            'Path': 'hf-task-dzero/hSparseMass',
            'Mass': 0,
            'Pt': 1,
            'ScoreBkg': 2,
            'ScorePrompt': 3,
            'ScoreFD': 4,
            'Cent': 5,
            'Occ': 6
        }
    elif hadron == 'Dplus':
        return {
            'Path': 'hf-task-dplus/hSparseMass',
            'Mass': 0,
            'Pt': 1,
            'ScoreBkg': 2,
            'ScorePrompt': 3,
            'ScoreFD': 4,
            'Cent': 5,
            'Occ': 6,
        }
    elif hadron == 'Ds':
        return {
            'Path': 'hf-task-ds/Data/hSparseMass',
            'Mass': 0,
            'Pt': 1,
            'Cent': 2,
            'ScoreBkg': 3,
            'ScorePrompt': 4,
            'ScoreFD': 5,
            'Occ': 6
        }
    else:
        logger(f"Sparse dictionary {data_type} not defined for hadron type {hadron}", level='ERROR')
