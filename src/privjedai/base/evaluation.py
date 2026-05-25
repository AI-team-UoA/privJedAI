from abc import ABC
from typing import Optional
from dataclasses import dataclass, field
from warnings import warn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from privjedai.datamodel import  EncodedData

@dataclass
class Metrics:
    """Metrics Dataclass"""
    f1: float = 0.0
    recall: float = 0.0
    precision: float = 0.0

@dataclass
class ConfusionMatrix:
    """Confusion Matrix Dataclass"""
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_matching_pairs: int = 0
    num_of_true_duplicates: int = 0

@dataclass
class TPS:
    """TPS Class"""
    tps_found : int = 0
    duplicate_emitted : Optional[dict] = None
    tps_indices: list = field(default_factory=list)

class BaseEvaluation(ABC):

    def __init__(self, encoded_data: EncodedData | None) -> None:
        self.xp = np
        self.metrics : Metrics  = Metrics()
        self.cm : ConfusionMatrix = ConfusionMatrix()
        if encoded_data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")
        self.encoded_data: EncodedData = encoded_data

        if self.encoded_data.skip_ground_truth:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                    "Data object has not been initialized with the ground-truth file")

        self.cm.true_positives = self.cm.true_negatives = \
            self.cm.false_positives = self.cm.false_negatives = 0

        self._tps = TPS()
        self.total_emissions : int = 0
        self.matchers_info : list = []

    def _set_true_positives(self, true_positives) -> None:
        self.cm.true_positives = true_positives
    def _set_total_matching_pairs(self, total_matching_pairs) -> None:
        self.cm.total_matching_pairs = total_matching_pairs

    def calculate_scores(self, true_positives: int = 0, total_matching_pairs: int = 0) -> None:
        """
        Calculate evaluation metrics for duplicate detection.

        Computes precision, recall, F1-score, and other classification metrics
        based on true positives and total matching pairs. Handles edge case where
        no matches are found.

        Args:
            true_positives: Number of correctly identified duplicate pairs
            total_matching_pairs: Total number of pairs identified as duplicates

        Returns:
            None: Updates instance attributes with calculated metrics including:
                - precision, recall, f1
                - true_positives, false_positives, false_negatives, true_negatives
                - num_of_true_duplicates, total_matching_pairs
        """
        self.cm.true_positives = true_positives
        self.cm.total_matching_pairs = total_matching_pairs

        if self.cm.total_matching_pairs == 0:
            warn("Evaluation: No matches found", Warning)
            self.cm.num_of_true_duplicates = self.cm.false_negatives \
                = self.cm.false_positives = self.cm.total_matching_pairs \
                    = self.cm.true_positives = self.cm.true_negatives \
                        = self.metrics.recall = self.metrics.f1 = self.metrics.precision = 0
        else:
            self.cm.num_of_true_duplicates = len(self.encoded_data.ground_truth)
            self.cm.false_negatives = self.cm.num_of_true_duplicates - self.cm.true_positives
            self.cm.false_positives = self.cm.total_matching_pairs - self.cm.true_positives
            cardinality = self.encoded_data.get_cardinality()
            self.cm.true_negatives = cardinality - \
                self.cm.false_negatives - self.cm.num_of_true_duplicates
            self.metrics.precision = self.cm.true_positives / self.cm.total_matching_pairs
            self.metrics.recall = self.cm.true_positives / self.cm.num_of_true_duplicates
            if self.metrics.precision == 0 or self.metrics.recall == 0:
                self.metrics.f1 = 0.0
            else:
                self.metrics.f1 = 2*((self.metrics.precision*self.metrics.recall)/
                                    (self.metrics.precision+self.metrics.recall))

    def report(
            self,
            configuration: dict = None,
            export_to_df=False,
            with_classification_report=False,
            verbose=True
        ) -> dict | pd.DataFrame:
        """
        Generate and display evaluation results report.

        Creates a comprehensive performance report with metrics visualization.
        Supports multiple output formats and verbosity levels.

        Args:
            configuration: Dictionary containing method configuration details
            export_to_df: If True, returns results as pandas DataFrame
            with_classification_report: If True, includes detailed classification metrics
            verbose: If True, prints formatted report to console

        Returns:
            Union[dict, pd.DataFrame]: Results dictionary or DataFrame containing:
                - Precision %, Recall %, F1 %
                - True Positives, False Positives, True Negatives, False Negatives
        """

        results_dict = {
                'Precision %': self.metrics.precision*100,
                'Recall %': self.metrics.recall*100,
                'F1 %': self.metrics.f1*100,
                'True Positives': self.cm.true_positives,
                'False Positives': self.cm.false_positives,
                'True Negatives': self.cm.true_negatives,
                'False Negatives': self.cm.false_negatives
            }

        if verbose:
            if configuration:
                params : dict = configuration['parameters']
                print('*' * 123)
                print(' ' * 40, 'Method: ', configuration['name'])
                print('*' * 123)
                print(
                    "Method name: " + configuration['name'] +
                    "\nParameters: \n" + ''.join([f'\t{k}: {v}\n' for k, v in params.items()]) +
                    f"Runtime: {configuration['runtime']:2.4f} seconds"
                )
            else:
                print(" " + (configuration['name'] if configuration else "") + " Evaluation \n---")


            print('\u2500' * 123)
            print(f"Performance:\n"
                f"\tPrecision: {self.metrics.precision*100:9.2f}% \n"
                f"\tRecall:    {self.metrics.recall*100:9.2f}%\n"
                f"\tF1-score:  {self.metrics.f1*100:9.2f}%"
            )

            print('\u2500' * 123)
            if with_classification_report:
                print(f"Classification report:\n"
                    f"\tTrue positives: {self.cm.true_positives}\n"
                    f"\tFalse positives: {self.cm.false_positives}\n"
                    f"\tTrue negatives: {self.cm.true_negatives}\n"
                    f"\tFalse negatives: {self.cm.false_negatives}\n"
                    f"\tTotal comparisons: {self.cm.total_matching_pairs}"
                )
                print('\u2500' * 123)

        if export_to_df:
            pd.set_option("display.precision", 2)
            results = pd.DataFrame.from_dict(results_dict, orient='index').T
            return results

        return results_dict

    def create_entity_index_from_clusters(
            self,
            clusters: list,
    ) -> np.ndarray | dict:
        """"""

        _limit = self.encoded_data.bounds[0]
        _d2_limit = self.encoded_data.bounds[1]
        entity_index = self.xp.full(_d2_limit, -1, dtype=np.int32)

        if not clusters:
            return entity_index


        cluster_lens = [len(c) for c in clusters]
        cluster_ids_flat = self.xp.repeat(np.arange(len(clusters)), cluster_lens)
        entities_flat = self.xp.concatenate([list(c) for c in clusters])
        is_d1 = entities_flat < _limit
        d1_counts = self.xp.bincount(cluster_ids_flat, weights=is_d1, minlength=len(clusters))
        d2_counts = self.xp.array(cluster_lens) - d1_counts
        self.cm.total_matching_pairs += int(np.sum(d1_counts * d2_counts))
        entity_index[entities_flat] = cluster_ids_flat

        return entity_index

    def confusion_matrix(self):
        """Generates a confusion matrix based on the classification report.
        """
        heatmap = [
            [int(self.cm.true_positives), int(self.cm.false_positives)],
            [int(self.cm.false_negatives), int(self.cm.true_negatives)]
        ]
        # plt.colorbar(heatmap)
        sns.heatmap(
            heatmap,
            annot=True,
            cmap='Blues',
            xticklabels=['Non-Matching', 'Matching'],
            yticklabels=['Non-Matching', 'Matching'],
            fmt='g'
        )
        plt.title("Confusion Matrix", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted pairs", fontsize=10, fontweight='bold')
        plt.ylabel("Real matching pairs", fontsize=10, fontweight='bold')
        plt.show()



