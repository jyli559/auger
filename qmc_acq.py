from botorch.acquisition.multi_objective.monte_carlo import MultiObjectiveMCAcquisitionFunction
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import combinations
from typing import Any, Callable, List, Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.cached_cholesky import CachedCholeskyMCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.utils import (
    prune_inferior_points_multi_objective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.model import Model
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling.base import MCSampler
from botorch.utils.multi_objective.box_decompositions.box_decomposition_list import (
    BoxDecompositionList,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.utils import (
    _pad_batch_pareto_frontier,
)
from botorch.utils.objective import apply_constraints_nonnegative_soft
from botorch.utils.torch import BufferDict
from botorch.utils.transforms import (
    concatenate_pending_points,
    is_fully_bayesian,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor
import math


class qUpperConfidenceBoundHypervolume(MultiObjectiveMCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: float,
        ref_point: Union[List[float], Tensor],
        partitioning: NondominatedPartitioning,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: Optional[Union[Tensor, float]] = 1e-3,
    ) -> None:
        r"""Upper Confidence Bound Hypervolume

        Args:
            model: A fitted model.
            ref_point: A list or tensor with `m` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the objective values (i.e. after
                applying`objective` to the samples).
            partitioning: A `NondominatedPartitioning` module that provides the non-
                dominated front and a partitioning of the non-dominated space in hyper-
                rectangles. If constraints are present, this partitioning must only
                include feasible points.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using `get_sampler`.
            objective: The MCMultiOutputObjective under which the samples are evaluated.
                Defaults to `IdentityMultiOutputObjective()`.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility. The acqusition function will compute expected feasible
                hypervolume.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into `X` upon forward call. Copied and set
                to have no gradient.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints. In case of a float the
                same eta is used for every constraint in constraints. In case of a
                tensor the length of the tensor must match the number of provided
                constraints. The i-th constraint is then estimated with the i-th
                eta value.
        """
        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
        self.beta_prime = math.sqrt(beta * math.pi / 2)
        self.register_buffer("ref_point", ref_point)
        cell_bounds = partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        self.q_out = -1
        self.q_subset_indices = BufferDict()

    def _cache_q_subset_indices(self, q_out: int) -> None:
        r"""Cache indices corresponding to all subsets of `q_out`.

        This means that consecutive calls to `forward` with the same
        `q_out` will not recompute the indices for all (2^q_out - 1) subsets.

        Note: this will use more memory than regenerating the indices
        for each i and then deleting them, but it will be faster for
        repeated evaluations (e.g. during optimization).

        Args:
            q_out: The batch size of the objectives. This is typically equal
                to the q-batch size of `X`. However, if using a set valued
                objective (e.g., MVaR) that produces `s` objective values for
                each point on the q-batch of `X`, we need to properly account
                for each objective while calculating the hypervolume contributions
                by using `q_out = q * s`.
        """
        if q_out != self.q_out:
            indices = list(range(q_out))
            tkwargs = {"dtype": torch.long, "device": self.ref_point.device}
            self.q_subset_indices = BufferDict(
                {
                    f"q_choose_{i}": torch.tensor(
                        list(combinations(indices, i)), **tkwargs
                    )
                    for i in range(1, q_out + 1)
                }
            )
            self.q_out = q_out

    def _compute_hv(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Compute the expected (feasible) hypervolume improvement given MC samples.

        Args:
            samples: A `n_samples x batch_shape x q' x m`-dim tensor of samples.
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (model_batch_shape)`-dim tensor of expected hypervolume
            improvement for each batch.
        """
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).
        obj = self.objective(samples, X=X)
        q = obj.shape[-2]
        if self.constraints is not None:
            feas_weights = torch.ones(
                obj.shape[:-1], device=obj.device, dtype=obj.dtype
            )
            feas_weights = apply_constraints_nonnegative_soft(
                obj=feas_weights,
                constraints=self.constraints,
                samples=samples,
                eta=self.eta,
            )
        self._cache_q_subset_indices(q_out=q)
        batch_shape = obj.shape[:-2]
        # this is n_samples x input_batch_shape x
        areas_per_segment = torch.zeros(
            *batch_shape,
            self.cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=obj.device,
        )
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *self.cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            self.cell_upper_bounds.shape[-2],
            1,
            self.cell_upper_bounds.shape[-1],
        )
        for i in range(1, self.q_out + 1):
            # TODO: we could use batches to compute (q choose i) and (q choose q-i)
            # simultaneously since subsets of size i and q-i have the same number of
            # elements. This would decrease the number of iterations, but increase
            # memory usage.
            q_choose_i = self.q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape)
            )
            # substract cell lower bounds, clamp min at zero
            lengths_i = (
                overlap_vertices - self.cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            areas_i = lengths_i.prod(dim=-1)
            # if constraints are present, apply a differentiable approximation of
            # the indicator function
            if self.constraints is not None:
                feas_subsets = feas_weights.index_select(
                    dim=-1, index=q_choose_i.view(-1)
                ).view(feas_weights.shape[:-1] + q_choose_i.shape)
                areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)
            areas_i = areas_i.sum(dim=-1)
            # Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            areas_per_segment += (-1) ** (i + 1) * areas_i
        # sum over segments and average over MC samples
        return areas_per_segment.sum(dim=-1)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)
        obj = self._compute_hv(samples=samples, X=X)
        mean = obj.mean(dim=0)
        ucb_samples = mean + self.beta_prime * (obj - mean).abs()
        return ucb_samples.mean(dim=0)
    

class qEHVI4RandomForest(torch.nn.Module):
    def __init__(
        self,
        samples: Tensor,
        # beta: float,
        ref_point: Union[List[float], Tensor],
        partitioning: NondominatedPartitioning,
        # sampler: Optional[MCSampler] = None,
        # objective: Optional[MCMultiOutputObjective] = None,
        # constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        # X_pending: Optional[Tensor] = None,
        # eta: Optional[Union[Tensor, float]] = 1e-3,
    ) -> None:
        r"""Upper Confidence Bound Hypervolume

        Args:
            model: A fitted model.
            ref_point: A list or tensor with `m` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the objective values (i.e. after
                applying`objective` to the samples).
            partitioning: A `NondominatedPartitioning` module that provides the non-
                dominated front and a partitioning of the non-dominated space in hyper-
                rectangles. If constraints are present, this partitioning must only
                include feasible points.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using `get_sampler`.
            objective: The MCMultiOutputObjective under which the samples are evaluated.
                Defaults to `IdentityMultiOutputObjective()`.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility. The acqusition function will compute expected feasible
                hypervolume.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into `X` upon forward call. Copied and set
                to have no gradient.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints. In case of a float the
                same eta is used for every constraint in constraints. In case of a
                tensor the length of the tensor must match the number of provided
                constraints. The i-th constraint is then estimated with the i-th
                eta value.
        """
        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        super().__init__()
        self.samples = samples
        # self.beta_prime = math.sqrt(beta * math.pi / 2)
        self.register_buffer("ref_point", ref_point)
        cell_bounds = partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        self.q_out = -1
        self.q_subset_indices = BufferDict()

    def _cache_q_subset_indices(self, q_out: int) -> None:
        r"""Cache indices corresponding to all subsets of `q_out`.

        This means that consecutive calls to `forward` with the same
        `q_out` will not recompute the indices for all (2^q_out - 1) subsets.

        Note: this will use more memory than regenerating the indices
        for each i and then deleting them, but it will be faster for
        repeated evaluations (e.g. during optimization).

        Args:
            q_out: The batch size of the objectives. This is typically equal
                to the q-batch size of `X`. However, if using a set valued
                objective (e.g., MVaR) that produces `s` objective values for
                each point on the q-batch of `X`, we need to properly account
                for each objective while calculating the hypervolume contributions
                by using `q_out = q * s`.
        """
        if q_out != self.q_out:
            indices = list(range(q_out))
            tkwargs = {"dtype": torch.long, "device": self.ref_point.device}
            self.q_subset_indices = BufferDict(
                {
                    f"q_choose_{i}": torch.tensor(
                        list(combinations(indices, i)), **tkwargs
                    )
                    for i in range(1, q_out + 1)
                }
            )
            self.q_out = q_out

    def _compute_ehvi(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Compute the expected (feasible) hypervolume improvement given MC samples.

        Args:
            samples: A `n_samples x batch_shape x q' x m`-dim tensor of samples.
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (model_batch_shape)`-dim tensor of expected hypervolume
            improvement for each batch.
        """
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).
        obj = samples.clone().detach()
        q = obj.shape[-2]
        # if self.constraints is not None:
        #     feas_weights = torch.ones(
        #         obj.shape[:-1], device=obj.device, dtype=obj.dtype
        #     )
        #     feas_weights = apply_constraints_nonnegative_soft(
        #         obj=feas_weights,
        #         constraints=self.constraints,
        #         samples=samples,
        #         eta=self.eta,
        #     )
        self._cache_q_subset_indices(q_out=q)
        batch_shape = obj.shape[:-2]
        # this is n_samples x input_batch_shape x
        areas_per_segment = torch.zeros(
            *batch_shape,
            self.cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=obj.device,
        )
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *self.cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            self.cell_upper_bounds.shape[-2],
            1,
            self.cell_upper_bounds.shape[-1],
        )
        for i in range(1, self.q_out + 1):
            # TODO: we could use batches to compute (q choose i) and (q choose q-i)
            # simultaneously since subsets of size i and q-i have the same number of
            # elements. This would decrease the number of iterations, but increase
            # memory usage.
            q_choose_i = self.q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape)
            )
            # substract cell lower bounds, clamp min at zero
            lengths_i = (
                overlap_vertices - self.cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            areas_i = lengths_i.prod(dim=-1)
            # if constraints are present, apply a differentiable approximation of
            # the indicator function

            # if self.constraints is not None:
            #     feas_subsets = feas_weights.index_select(
            #         dim=-1, index=q_choose_i.view(-1)
            #     ).view(feas_weights.shape[:-1] + q_choose_i.shape)
            #     areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)

            areas_i = areas_i.sum(dim=-1)
            # Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            areas_per_segment += (-1) ** (i + 1) * areas_i
        # sum over segments and average over MC samples
        return areas_per_segment.sum(dim=-1).mean(dim=0)

    # @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return self._compute_ehvi(samples=self.samples, X=X)