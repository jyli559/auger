# AUGER
A Multi-Objective Design Space Exploration Framework for CGRAs.

## Abstract
Coarse-grained reconﬁgurable architectures (CGRAs) are gaining increasing attention as domain-specific accelerators due to their high flexibility and energy efficiency. CGRAs typically involve numerous design parameters, which results in a vast design space. Therefore, exploring a trade-off between various metrics, such as area and throughput, is challenging. Although many CGRA design space exploration (DSE) frameworks have been proposed to overcome this issue, a scalable and robust DSE framework that can achieve a trade-off between different metrics for a large design space is still lacking. In this paper, we propose AUGER, an open-source multi-objective DSE framework for CGRAs. AUGER incorporates an unsupervised active learning-based sampling algorithm that maximizes both representativeness and diversity of initial samples. To enhance the robustness and scalability of AUGER, we construct a multi-acquisition strategy and a hybrid surrogate model within the framework. Our approach demonstrates superior performance on two Pareto frontier metrics compared to state-of-the-art DSE algorithms on different datasets.

## Dependency
python >= 3.8

torch >= 1.7.1

GPytorch>=1.9.1

BoTorch >= 0.8.3

scikit-learn >= 1.2.2

Numpy >= 1.24.2

pandas >= 2.0.0

## Run and get results
```bash
cd auger
python3 auger.py
```
