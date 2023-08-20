import os
import torch
import tqdm
import gpytorch
import numpy as np
from dse_problem import DesignSpaceExplorationProblem
from initial_sample import rds_initial_sample
from MAB import MultiArmedBandit  
from model import initialize_dkl_gp, initialize_rdn_forest
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from qmc_acq import qUpperConfidenceBoundHypervolume, qEHVI4RandomForest
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from dse_problem import get_pareto_front
import pygmo as pg
from report_result import info

class Explorer(object):
    def __init__(self, problem: DesignSpaceExplorationProblem, candidate_models: list = ["GP", "RF"]):
        super(Explorer, self).__init__()
        self.problem = problem
        self.model_selector = MultiArmedBandit(len(candidate_models), reshape_factor=1)
        self.adrs_record = []
        self.hv_record = []
        self.model_record = []

    def initialize(self):
        self.visited_x, self.visited_y = rds_initial_sample(self.problem, 16)
        self.init_y = self.visited_y.clone().detach()
        self.adrs_record.append(self.problem.calc_adrs(self.visited_y))
        self.hv_record.append(self.problem.calc_hv(self.visited_y))

    def fit_and_suggest(self):
        if len(self.hv_record) > 1:
            self.model_selector.update_posterior(self.hv_record[-1] > self.hv_record[-2])
        
        selected_model = self.model_selector.choose_bandit();
        if selected_model == 0:
            self.model_record.append("GP")
            self.fit_dkl_gp()
            return self.qHybrid_suggest_gp()
        else:
            self.model_record.append("RF")
            self.fit_rdn_forest()
            return self.qHybrid_suggest_rf()
        
    def set_optimizer(self) -> torch.optim.Adam:
        parameters = [
            {"params": self.model.mlp.parameters()},
            {"params": self.model.gp.covar_module.parameters()},
            {"params": self.model.gp.mean_module.parameters()},
            {"params": self.model.gp.likelihood.parameters()}
        ]
        return torch.optim.Adam(
            parameters, lr=0.001
        )

    def fit_dkl_gp(self):
        self.model = initialize_dkl_gp(
            self.visited_x,
            self.visited_y,
            6
        )
        self.model.set_train()
        optimizer = self.set_optimizer()

        iterator = tqdm.trange(1000, desc="Training DKL-GP")
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.gp.likelihood,
            self.model.gp
        )
        y = self.model.transform_ylayout(self.visited_y).squeeze(1)
        for i in iterator:
            optimizer.zero_grad()
            _y = self.model.train(self.visited_x)
            loss = -mll(_y, y)
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=loss.item())
        self.model.set_eval()
    
    def fit_rdn_forest(self):
        self.model = initialize_rdn_forest(self.visited_x, self.visited_y)

    def qHybrid_suggest_gp(self, batch: int = 1):
        partitioning = NondominatedPartitioning(
            ref_point=self.problem._ref_point.to(self.model.device),
            Y=self.visited_y.to(self.model.device)
        )

        
        ucb_func = qUpperConfidenceBoundHypervolume(
            model=self.model.gp,
            beta=2.0,
            ref_point=self.problem._ref_point.tolist(),
            partitioning=partitioning
            # constraints= [all_positive]
        ).to(self.model.device)

        ucb_val = ucb_func(
            self.model.forward_mlp(
                self.problem.x.to(torch.float).to(self.model.device)
            ).unsqueeze(1).to(self.model.device)
        ).to(self.model.device)

        ehvi_func = ExpectedHypervolumeImprovement(
            model=self.model.gp,
            ref_point=self.problem._ref_point.tolist(),
            partitioning=partitioning
        ).to(self.model.device)

        ehvi_val = ehvi_func(
            self.model.forward_mlp(
                self.problem.x.to(torch.float).to(self.model.device)
            ).unsqueeze(1).to(self.model.device)
        ).to(self.model.device)

        acq_val = torch.cat((ucb_val.unsqueeze(-1), ehvi_val.unsqueeze(-1)), -1)
        candidate_indices = torch.where(is_non_dominated(acq_val))[0].numpy()
        indices = np.random.choice(candidate_indices, batch)

        new_x = self.problem.x[indices].to(torch.float32).reshape(-1, self.problem.design_dim)
        self.visited_x = torch.cat((self.visited_x, new_x), 0)
        self.visited_y = torch.cat((
                self.visited_y,
                self.problem.evaluate(new_x).unsqueeze(0)
            ),
            0
        )
        self.update_metrics()
        self.problem.remove_sampled(new_x)

    def qHybrid_suggest_rf(self, batch: int = 1):
        partitioning = NondominatedPartitioning(
            ref_point=self.problem._ref_point.to("cpu"),
            Y=self.visited_y.to("cpu")
        )

        y_samples = self.model.sample(self.problem.x.numpy()).unsqueeze(-2)

        ehvi_func = qEHVI4RandomForest(
            samples= y_samples,
            ref_point=self.problem._ref_point.tolist(),
            partitioning=partitioning
        ).to("cpu")

        ehvi_val = ehvi_func(
                self.problem.x.to(torch.float).to("cpu")
            .unsqueeze(1).to("cpu")
        ).to("cpu")

        if torch.any(ehvi_val):
            _, indices = torch.topk(ehvi_val, k=batch)
        else:
            pred_y, std_y = self.model.predict(self.problem.x.numpy(), return_std=True)
            candidate_indices = torch.where(is_non_dominated(pred_y))[0]
            _, tmp_indices = torch.topk(std_y[candidate_indices], k=batch)
            indices = candidate_indices[tmp_indices]

        new_x = self.problem.x[indices].to(torch.float32).reshape(-1, self.problem.design_dim)
        self.visited_x = torch.cat((self.visited_x, new_x), 0)
        self.visited_y = torch.cat((
                self.visited_y,
                self.problem.evaluate(new_x).unsqueeze(0)
            ),
            0
        )
        self.update_metrics()
        self.problem.remove_sampled(new_x)
    
    def update_metrics(self):
        self.adrs_record.append(self.problem.calc_adrs(self.visited_y))
        self.hv_record.append(self.problem.calc_hv(self.visited_y))

    def report_result(self, model_type: str = "GP"):
        gt = get_pareto_front(self.problem.total_y, reverse=False)
        pred = get_pareto_front(self.visited_y, reverse=False)
        # gt_x = self.problem.total_x[is_non_dominated(self.problem.total_y), :]
        hv_gt = pg.core.hypervolume(-1.0 * np.array(gt))
        hv_pred = pg.core.hypervolume(-1.0 * np.array(pred))
        self.adrs_record.append(self.problem.calc_adrs(self.visited_y))
        info("inital ADRS: {}, Hypervolume: {}".format(self.adrs_record[0], self.hv_record[0]))
        info("pareto ADRS: {}, Hypervolume: {} / {}".format(
                # str(gt_x),
                self.adrs_record[-1],
                hv_pred.compute(np.array([0.0,0.0])),
                hv_gt.compute(np.array([0.0,0.0]))
            )
        )
        print(self.hv_record[-1])
        # plot_pareto_set(
        #     rescale_dataset(pred),
        #     gt=rescale_dataset(gt),
        #     init=rescale_dataset(self.init_y),
        #     visited = rescale_dataset(self.visited_y),
        #     design_space=self.problem.configs["dataset"]["path"],
        #     # rd_space = "dataset/dataRS.csv",
        #     output=os.path.join(p, "report.pdf")
        # )
        # write_txt(
        #     os.path.join(
        #         p,
        #         "adrs_" + model_type + ".rpt"
        #     ),
        #     np.array(self.adrs),
        #     fmt="%f"
        # )
        # write_txt(
        #     os.path.join(
        #         p,
        #         "hypervolume_" + model_type + ".rpt"
        #     ),
        #     np.array(self.hv),
        #     fmt="%f"
        # )
        return self.adrs_record[0], self.adrs_record[-1], self.hv_record[0], self.hv_record[-1]

def create_explorer(problem: DesignSpaceExplorationProblem):
    return Explorer(problem)