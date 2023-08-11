from dse_problem import DesignSpaceExplorationProblem
from initial_sample import rds_initial_sample
from MAB import MultiArmedBandit  


class Explorer(object):
    def __init__(self, problem: DesignSpaceExplorationProblem, candidate_models: list):
        super(Explorer, self).__init__()
        self.problem = problem
        self.model_selector = MultiArmedBandit(len(candidate_models), reshape_factor=1)
        self.adrs_record = []
        self.hv_record = []
        self.model_record = []

    def initialize(self):
        self.visited_x, self.visited_y = rds_initial_sample(self.problem, 16)
        self.adrs_record.append(self.problem.calc_adrs(self.visited_y))
        self.hv_record.append(self.problem.calc_hv(self.visited_y))
