import torch
import csv
import numpy as np
import pygmo as pg
from botorch.utils.multi_objective.pareto import is_non_dominated

class DesignSpaceExplorationProblem(object):
    def __init__(self, path_dataset: str, num_objectives: int = 2):
        super(DesignSpaceExplorationProblem, self).__init__()
        self.num_objectives = num_objectives
        self._ref_point = torch.Tensor([0 for i in range(num_objectives)])
        self.load_dataset(path_dataset)
        self.n_dim = num_objectives

    def load_dataset(self,path_dataset: str, scale: bool = True):
        dataset = []
        with open(path_dataset, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                dataset.append(row)
        
        data = []
        for d in dataset:
            _data = []
            for i in d[:-4]:
                if i == 'True':
                    _data.append(1)
                elif i == 'False':
                    _data.append(0)
                else:
                    _data.append(int(i))
            for i in d[-4:-2]:
                _data.append(float(i))
            data.append(_data)
        data = np.array(data)

        [max_area, max_perf] = np.max(data, axis=0)[-2:]
        [min_area, min_perf] = np.min(data, axis=0)[-2:]
        self.bias_area = max_area
        self.bias_perf = min_perf
        self.scale_area = max_area - min_area
        self.scale_perf = max_perf - min_perf

        if scale:
            data[:, -2] = (self.bias_area - data[:, -2]) / self.scale_area
            data[:, -1] = (data[:, -1] - self.bias_perf) / self.scale_perf

        self.total_x = torch.Tensor(data[:, :-2])
        self.total_y = torch.Tensor(data[:, -2:])
        self.x = self.total_x.clone().detach()
        self.y = self.total_y.clone().detach()
        self.design_dim = self.x.shape[-1]
    
    def evaluate(self, x: torch.Tensor):
        _, indices = torch.topk(((self.x.t() == x.unsqueeze(-1)).all(dim=1)).int(),1,1)
        value = self.y[indices].to(torch.float32).squeeze()
        sampled = torch.zeros(self.x.size()[0],dtype=torch.bool)[:, np.newaxis]
        mask = sampled.index_fill_(0, indices.squeeze(), True).squeeze()
        self.x = self.x[mask[:] == False]
        self.y = self.y[mask[:] == False]

        return value
    
    def rescale_value(self, data: torch.Tensor):
        _data = data.clone().detach()
        _data[:, -2] = self.bias_area - self.scale_area * _data[:, -2]
        _data[:, -1] = self.bias_perf + self.scale_perf * _data[:, -1]

        return _data

    def remove_sampled(self, x: torch.Tensor):
        sampled = torch.zeros(
            self.x.size()[0],
            dtype=torch.bool
        )[:, np.newaxis]
        _, indices = torch.topk(
            ((self.x.t() == x.unsqueeze(-1)).all(dim=1)).int(),
            1,
            1
        )
        mask = sampled.index_fill_(0, indices.squeeze(), True).squeeze()
        self.x = self.x[mask[:] == False]
        self.y = self.y[mask[:] == False]

    def calc_adrs(self, y: torch.Tensor):
        adrs = 0
        reference = get_pareto_front(self.total_y)
        for omega in reference:
            mini = float('inf')
            for gama in y:
                mini = min(mini, np.linalg.norm(omega - gama))
            adrs += mini
        adrs = adrs / len(reference)
        return adrs

    def calc_hv(self, y: torch.Tensor):
        return pg.core.hypervolume(-1.0*np.array(get_pareto_front(y))).compute(np.array(self._ref_point))

def get_pareto_front(y: torch.Tensor, reverse=False):
    if reverse:
        return y[is_non_dominated(-y)]
    else:
        return y[is_non_dominated(y)]


def create_dse_problem(dataset_path):
    return DesignSpaceExplorationProblem(dataset_path)

