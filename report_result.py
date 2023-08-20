import matplotlib.pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated
import torch
from copy import deepcopy


markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', # 'w'
]

def info(msg: str):
    print("[INFO]: {}".format(msg))