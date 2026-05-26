"""Evaluation module
This file contains all the methods for evaluating every module in pyjedai.
"""
import cupy as cp
import numpy as np
from privjedai.base.evaluation import  BaseEvaluation

class Evaluation(BaseEvaluation):

    def create_entity_index_from_clusters(
            self,
            clusters: list,
            gpu: bool = False
    ) -> dict | np.ndarray :
        """

        Args:
            clusters: list of the clusters produced
            gpu: Enable GPU

        Returns:

        """

        if gpu:
            self.xp = cp
        else:
            self.xp = np
        return super().create_entity_index_from_clusters(clusters)

