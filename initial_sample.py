import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
import random
from dse_problem import DesignSpaceExplorationProblem

class RDSampler(object):
    def __init__(self, problem: DesignSpaceExplorationProblem, num_corner_points: int = 4):
        self.problem = problem
        self.dataset = problem.x.numpy().copy()
        self.num_corner_points = num_corner_points

    def cornerSampling(self, n_sample):
        sizes = self.dataset[:,0] * self.dataset[:,1]
        delays = self.dataset[:,3].copy()
        size_corner_idx = set(np.where(sizes == np.max(sizes))[0].tolist() + np.where(sizes == np.min(sizes))[0].tolist())
        delay_corner_idx = set(np.where(delays == np.max(delays))[0].tolist() + np.where(delays == np.min(delays))[0].tolist())
        corner_candidates = self.dataset[np.array(list(size_corner_idx.union(delay_corner_idx))), :].copy()
        
        corner_clusters = self.clustering(corner_candidates, n_sample, is_strict=True)
        corner_idxes = []
        for i in range(len(corner_clusters)):
            corner_idxes.append(
                np.argmin(np.max(euclidean_distances(corner_clusters[i]), axis=1))
                )
        corner_idxes = np.array(corner_idxes)
        self.corner_points = corner_candidates[corner_idxes, :]

    def clustering(self, points, max_k, max_iter=1000, is_strict=False):
        if(max_k == 1):
            return [points]
        # print(max_k)
        """k-means clustering"""
        best_score = 0
        if is_strict:
            i = max_k
        else:
            i = max_k - 5 if max_k > 6 else 2
        # i = max_k
        while i <= max_k:
            partitioner = KMeans(n_clusters=i, n_init=50, max_iter=max_iter)
            cluster_result = partitioner.fit_predict(points)
            score = metrics.calinski_harabasz_score(points, cluster_result)
            if score > best_score:
                best_cluster_result = cluster_result
                best_score = score
            # print(i, score)
            i += 1
        print(np.max(best_cluster_result)+1)
        return self.gather_groups(points, best_cluster_result, np.max(best_cluster_result)+1)

    def gather_groups(self, dataset, cluster_result, n_cluster):
        clustered_dataset = [[] for i in range(n_cluster)]

        for i in range(len(dataset)):
            clustered_dataset[cluster_result[i]].append(dataset[i])
        for i in range(len(clustered_dataset)):
            clustered_dataset[i] = np.array(clustered_dataset[i])

        return clustered_dataset

    def sample(self, initial_budgets: int, predef_clusters: int = 5, max_iter: int = 1000, with_corner: bool = False):
        is_corner_sample = (initial_budgets > 16) or with_corner
        if is_corner_sample:
            self.cornerSampling(self.num_corner_points)
        
        clustered_dataset = self.clustering(
            self.dataset,
            predef_clusters,
            max_iter=max_iter
        )
        self.clustered_dataset = []
        for points in clustered_dataset:
            # points[:,1] /= 6.0
            self.clustered_dataset += self.clustering(
                points,
                initial_budgets // len(clustered_dataset),
                max_iter,
                True
            )
        
        idx_candidates = []
        for i in range(len(self.clustered_dataset)):
            idx_candidates.append(
                np.argmin(np.max(euclidean_distances(self.clustered_dataset[i]), axis=1))
                )
        history_candIdx = set()
        history_candIdx.add(str(idx_candidates))

        candidates = np.array([self.clustered_dataset[i][idx_candidates[i]] for i in range(len(idx_candidates))])
        cnt = 0
        order_list = list(range(len(candidates)))
        while cnt < 1000:
            random.shuffle(order_list)
            for i in order_list:
                R_score = np.sum(euclidean_distances(self.clustered_dataset[i]), axis=1) / (1.0 * len(self.clustered_dataset[i]))
                if is_corner_sample:
                    D_score = np.min(euclidean_distances(self.clustered_dataset[i], np.concatenate((candidates[:i], candidates[i+1:], self.corner_points))), axis=1)
                else:
                    D_score = np.min(euclidean_distances(self.clustered_dataset[i], np.concatenate((candidates[:i], candidates[i+1:]))), axis=1)
                idx_candidates[i] = np.argmax(D_score - R_score)
                candidates[i] = self.clustered_dataset[i][idx_candidates[i]]
            if str(idx_candidates) not in history_candIdx:
                history_candIdx.add(str(idx_candidates))
            else:
                break
            cnt += 1
        
        if is_corner_sample:
            return np.concatenate((self.corner_points, candidates))
        else:
            return candidates


def rds_initial_sample(problem: DesignSpaceExplorationProblem, initial_budgets: int):
    sampler = RDSampler(problem)
    x = torch.Tensor(sampler.sample(initial_budgets))
    y = problem.evaluate(x)
    return x, y