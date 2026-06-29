"""Evaluation module
This file contains all the methods for evaluating every module in pyjedai.
"""
from collections import defaultdict
import numpy as np

from privjedai.base.evaluation import BaseEvaluation


class Evaluation(BaseEvaluation):
    """Evaluation class. Contains multiple methods for all the fitted & predicted data.
    """

    def evaluate_candidate_pairs(self, prediction: dict) -> None:
        """
        Evaluates the candidate pairs based on the predicted matches.     
        Args:
            prediction:  Dict   A dictionary where keys are candidate pair IDs and values are lists of predicted matches. Returns:
        """
        total_matching_pairs = 0
        bounds_offset = self.encoded_data.bounds[0]

        for block in prediction.values():
            total_matching_pairs += len(block)

        true_positives = sum(
            1 for id1, id2 in self.encoded_data.ground_truth.values
            if id1 in prediction and (id2 + bounds_offset) in prediction[id1]
        )
        self.calculate_scores(true_positives, total_matching_pairs)

    def evaluate_blocks(self, blocks_with_keys: np.ndarray, limit_: int):

        unique_keys, inverse_keys = np.unique(blocks_with_keys[:, 0],  return_inverse=True)
        minlength = unique_keys.shape[0]


        entity_1_keys = inverse_keys[blocks_with_keys[:, 1] < limit_]
        entity_2_keys = inverse_keys[blocks_with_keys[:, 1] >= limit_]


        cnt_entity_1_blocks = np.bincount(entity_1_keys, minlength=minlength)
        cnt_entity_2_blocks = np.bincount(entity_2_keys, minlength=minlength)

        blocks_cardinalities = cnt_entity_1_blocks * cnt_entity_2_blocks
        total_matching_blocks = np.sum(blocks_cardinalities)

        # total_matching_blocks = blocks_with_keys.shape[0]
        id_to_keys = defaultdict(set)
        for key, id_ in blocks_with_keys:
            id_to_keys[id_].add(key)

        true_positives = 0
        bounds_offset = self.encoded_data.bounds[0]

        for id1, id2 in self.encoded_data.ground_truth.values:
            if id1 in id_to_keys and (id2 + bounds_offset) in id_to_keys:
                if not id_to_keys[id1].isdisjoint(id_to_keys[id2 + bounds_offset]):
                    true_positives += 1

        self.calculate_scores(true_positives, total_matching_blocks)
