"""Evaluation module
This file contains all the methods for evaluating every module in pyjedai.
"""
import cupy as cp
import numpy as np
from typing import List, Tuple, Dict
from warnings import warn
import random
from dataclasses import dataclass, field


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from privjedai.datamodel import Block, EncodedData
from privjedai.utils import  batch_pairs

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
    duplicate_emitted : dict = None
    tps_indices: list = field(default_factory=list)




class Evaluation:
    """Evaluation class. Contains multiple methods for all the fitted & predicted data.
    """
    def __init__(self, encoded_data: EncodedData) -> None:
        self.metrics : Metrics  = Metrics()
        self.cm : ConfusionMatrix = ConfusionMatrix()
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

    def calculate_scores(self, true_positives=None, total_matching_pairs=None) -> None:
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
        if true_positives is not None:
            self.cm.true_positives = true_positives

        if total_matching_pairs is not None:
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
            if self.metrics.precision == 0.0 or self.metrics.recall == 0.0:
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
        ) -> any:
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
                    f"\tTrue negatives: {self.cm.false_negatives}\n"
                    f"\tFalse negatives: {self.cm.total_matching_pairs}\n"
                    f"\tTotal comparisons: {self.cm.total_matching_pairs}"
                )
                print('\u2500' * 123)

        if export_to_df:
            pd.set_option("display.precision", 2)
            results = pd.DataFrame.from_dict(results_dict, orient='index').T
            return results

        return results_dict


    def _create_entity_index_from_clusters_gpu(
            self,
            clusters: list
    ) -> dict:
        """
        Docstring for _create_entity_index_from_clusters_gpu

        :param self: Description
        :param clusters: Description
        :type clusters: list
        :return: Description
        :rtype: dict
        """

        _limit = self.encoded_data.bounds[0]
        _d2_limit = self.encoded_data.bounds[1]
        # print(clusters)
        # all_gt_ids: set = set(range(_d2_limit))
        entity_index = cp.full(_d2_limit, -1, dtype=cp.int32)

        if not clusters:
            return entity_index


        cluster_lens = [len(c) for c in clusters]
        cluster_ids_flat = cp.repeat(cp.arange(len(clusters)), cluster_lens)
        entities_flat = cp.concatenate([cp.array(list(c)) for c in clusters])
        is_d1 = entities_flat < _limit
        d1_counts = cp.bincount(cluster_ids_flat, weights=is_d1, minlength=len(clusters))
        d2_counts = cp.array(cluster_lens) - d1_counts
        self.cm.total_matching_pairs += int(cp.sum(d1_counts * d2_counts))
        entity_index[entities_flat] = cluster_ids_flat

        return entity_index

    def create_entity_index_from_clusters(
            self,
            clusters: list,
            gpu: bool = False
    ) -> dict:
        """
        Docstring for create_entity_index_from_clusters

        :param self: Description
        :param clusters: Description
        :type clusters: list
        :return: Description
        :rtype: dict
        """

        if gpu:
            return self._create_entity_index_from_clusters_gpu(clusters)
        _limit = self.encoded_data.bounds[0]
        _d2_limit = self.encoded_data.bounds[1]
        # print(clusters)
        # all_gt_ids: set = set(range(_d2_limit))
        entity_index = np.full(_d2_limit, -1, dtype=np.int32)

        if not clusters:
            return entity_index


        cluster_lens = [len(c) for c in clusters]
        cluster_ids_flat = np.repeat(np.arange(len(clusters)), cluster_lens)
        entities_flat = np.concatenate([list(c) for c in clusters])
        is_d1 = entities_flat < _limit
        d1_counts = np.bincount(cluster_ids_flat, weights=is_d1, minlength=len(clusters))
        d2_counts = np.array(cluster_lens) - d1_counts
        self.cm.total_matching_pairs += int(np.sum(d1_counts * d2_counts))
        entity_index[entities_flat] = cluster_ids_flat

        return entity_index

        # entity_index = {}
        # for cluster_id, cluster in enumerate(clusters):
        #     cluster_entities_d1 = 0
        #     cluster_entities_d2 = 0

        #     for entity_id in set(cluster):
        #         entity_index[entity_id] = cluster_id

        #         if entity_id < _limit:
        #             cluster_entities_d1 += 1
        #         else:
        #             cluster_entities_d2 += 1

        #     self.cm.total_matching_pairs += cluster_entities_d1*cluster_entities_d2

        # return entity_index

    def create_entity_index_from_blocks(
            self,
            blocks: Dict[any, Block]
    ) -> dict:
        """Used for evaluating blocking techniques. Evaluating only for 2 datasets"""
        entity_index : Dict[int, set] = {}
        for block_id, block in blocks.items():
            for entity_id in block.entities['D0']:
                entity_index.setdefault(entity_id, set())
                entity_index[entity_id].add(block_id)

            for entity_id in block.entities['D1']:
                entity_index.setdefault(entity_id, set())
                entity_index[entity_id].add(block_id)

            self.cm.total_matching_pairs += len(block.entities['D0'])*len(block.entities['D1'])

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

    def _visualize_roc_for_method(self,
                ax,
                methods_data: List[dict],
                drop_tp_indices=True) -> Tuple[list, list]:
        colors = []
        normalized_aucs = []
        # for each method layout its plot
        for method_data in methods_data:
            cumulative_recall, normalized_auc = self._generate_auc_data(
                total_candidates=method_data['total_emissions'],tp_positions=method_data['tp_idx'])
            if drop_tp_indices:
                del method_data['tp_idx']
            method_name=method_data['name']
            method_data['auc'] = normalized_auc
            method_data['recall'] = cumulative_recall[-1] if len(cumulative_recall) != 0 else 0.0

            x_values = range(len(cumulative_recall))
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
            colors.append(color)
            normalized_aucs.append(normalized_auc)
            # if proportional: sizes = [cr * 100 for cr in cumulative_recall]
            # else: sizes = [10] * len(cumulative_recall)
            ax.scatter(x_values, cumulative_recall,
                    marker='o', s=0.05, color=color, label=method_name)
            ax.plot(x_values, cumulative_recall, color=color)
        return colors, normalized_aucs, len(cumulative_recall)



    def visualize_roc(self, methods_data : List[dict], drop_tp_indices=True) -> None:
        """
        Visualize ROC-like cumulative recall curves for multiple methods.

        Args:
            methods_data (List[dict]):
                A list of dictionaries containing method evaluation data.
                Each dict should include:
                    - 'name' (str): Method name.
                    - 'total_emissions' (int): Total number of emissions or comparisons.
                    - 'tp_positions' (List[int]): Positions of true positives.
                Additional fields like 'tp_idx' may be removed if `drop_tp_indices` is True.
            drop_tp_indices (bool, optional):
                If True, removes 'tp_idx' from each method_data to save memory. Defaults to True.

        Returns:
            None: Displays the ROC-like cumulative recall plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))  # set the size of the plot
        colors, normalized_aucs, len_cum_recall = self. \
                    _visualize_roc_for_method(ax, methods_data, drop_tp_indices)

        ax.set_xlabel('ec*', fontweight='bold', labelpad=10)
        ax.set_ylabel('Cumulative Recall', fontweight='bold', labelpad=10)
        ax.set_xlim(0, len_cum_recall)
        ax.set_ylim(0, 1)

        # add a legend showing the name of each curve and its color
        legend = ax.legend(ncol=2, loc='lower left', title='Methods', bbox_to_anchor=(0, -0.4))
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(12)
        plt.setp(legend.get_lines(), linewidth=4)

        # add AUC score legend
        handles, _ = ax.get_legend_handles_labels()
        auc_legend_labels = [f'AUC: {nauc:.2f}' for nauc in normalized_aucs]
        auc_legend = ax.legend(handles,
                        auc_legend_labels, loc='lower left',
                        bbox_to_anchor=(0.5, -0.4),
                        ncol=2,
                        frameon=True,
                        title='AUC', title_fontsize=12)
        auc_legend.get_title().set_fontweight('bold')
        for i, text in enumerate(auc_legend.get_texts()):
            plt.setp(text, color=colors[i])
        ax.add_artist(legend)

        # set the figure background color to the RGB color of the solarized terminal theme
        fig.patch.set_facecolor((0.909, 0.909, 0.909))

        # adjust the margins of the figure to move the graph to the right
        fig.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)

        plt.show()

    def calculate_ideal_auc(self, pairs_num : int, true_duplicates_num : int) -> float:
        """Calculates the ideal AUC for the given number of candidate pairs
        Args:
            pairs_num (int): Total number of candidate pairs
            true_duplicates_num (int): The number of true duplicates
        Returns:
            float: Ideal AUC
        """
        ideal_auc : float

        if pairs_num == true_duplicates_num:
            ideal_auc = 0.5
        else:
            ideal_auc = (pairs_num % true_duplicates_num) / true_duplicates_num * 0.5
            if pairs_num > true_duplicates_num:
                ideal_auc += (pairs_num - true_duplicates_num) / true_duplicates_num

        return ideal_auc

    def _till_full_tps_emission(self) -> bool:
        """Checks if emission should be stopped once all TPs have been found (TPs dict supplied)
        Returns:
            bool: Stop emission on all TPs found / Emit all pairs
        """
        return self._tps.duplicate_emitted is not None

    def _all_tps_emitted(self) -> bool:
        """Checks if all TPs have been emitted
        (Defaults to False in the case of all pairs emission approach)

        Returns:
            bool: All TPs emitted / not emitted
        """
        if self._till_full_tps_emission():
            return self._tps.tps_found >= len(self._tps.duplicate_emitted)
        return False

    def _update_true_positive_entry(self, entity : int, candidate : int) -> None:
        """Updates the checked status of the given true positive

        Args:
            entity (int): Entity ID
            candidate (int): Candidate ID
        """
        if self._till_full_tps_emission():
            if not self._tps.duplicate_emitted[(entity, candidate)]:
                self._tps.duplicate_emitted[(entity, candidate)] = True
                self._tps.tps_found += 1


    def calculate_tps_indices(self,
            pairs : List[Tuple[float, int, int]],
            duplicate_of : dict = None,
            duplicate_emitted : dict = None,
            batch_size : int  = 1) -> Tuple[List[int], int]:
        """
        Args:
            pairs (List[float, int, int]): Candidate pairs to emit
                    in the form [similarity, first dataframe entity ID, second dataframe entity ID]
            duplicate_of (dict, optional): Dictionary
                of the form [entity ID] -> [IDs of duplicate entities]. Defaults to None.
            duplicate_emitted (dict, optional): Dictionary
                of the form [true positive pair] -> [emission status: emitted/not].
                Defaults to None.
            batch_size (int, optional): Recall update emission rate. Defaults to 1.

        Raises:
            AttributeError: No ground truth has been given
        Returns:
            Tuple[List[int], int]: Indices of true positive
                duplicates within the candidates list and the total emissions
        """

        if duplicate_emitted is not None:
            for pair in duplicate_emitted.keys():
                duplicate_emitted[pair] = False

        if duplicate_of is None:
            raise AttributeError("Can calculate ROC AUC without a ground-truth file. \
                Data object mush have initialized with the ground-truth file")

        self._tps.tps_found = 0
        self._tps.duplicate_emitted = duplicate_emitted
        self._tps.tps_indices = []

        batches = batch_pairs(pairs, batch_size)
        # ideal_auc = self.calculate_ideal_auc(len(pairs), self.cm.num_of_true_duplicates)
        self.total_emissions : int = 0
        for batch in batches:
            for _, entity, candidate in batch:
                if self._all_tps_emitted():
                    break
                if candidate in duplicate_of[entity]:
                    self._update_true_positive_entry(entity, candidate)
                    self._tps.tps_indices.append(self.total_emissions)

            self.total_emissions += 1
            if self._all_tps_emitted():
                break

        # _normalized_auc = 0 if(ideal_auc == 0) else _normalized_auc / ideal_auc
        return self._tps.tps_indices, self.total_emissions


    def _generate_auc_data(self,
            total_candidates : int, tp_positions : List[int]) -> Tuple[List[float], float]:
        """Generates the recall axis containing
        the recall value for each emission and calculates the normalized AUC

        Args:
            total_candidates (int): Total number of pairs emitted
            tp_positions (List[int]): Indices of true positives within the candidate pairs list

        Returns:
            Tuple[List[float], float]: Recall axis and the normalized AUC
        """

        _recall_axis : List[float] = []
        _recall : float = 0.0
        _tp_index : int = 0
        _dataset_total_tps : int = len(self.encoded_data.ground_truth)
        _total_found_tps : int = len(tp_positions)

        for recall_index in range(total_candidates):
            if _tp_index < _total_found_tps:
                if recall_index == tp_positions[_tp_index]:
                    _recall =  (_tp_index + 1.0) / _dataset_total_tps
                    _tp_index += 1
            _recall_axis.append(_recall)

        _normalized_auc : float = sum(_recall_axis) / (total_candidates + 1.0)

        return _recall_axis, _normalized_auc


    def visualize_results_roc(self, results : dict, drop_tp_indices=True) -> None:
        """For each of the executed workflows,
        calculates the cumulative recall and normalized AUC based upon true positive indices.
        Finally, displays the ROC for all of the workflows
        with proper annotation (each workflow gains a unique identifier).

        Args:
            results (dict): Nested dictionary of the form [dataset]
                -> [matcher] -> [executed workflows and their info] / [model] -> [executed -//-]
        """

        workflows_info : List[Tuple[dict]] = []

        for dataset in results:
            matchers = results[dataset]
            for matcher in matchers:
                matcher_info = matchers[matcher]
                if isinstance(matcher_info, list):
                    for workflow_info in matcher_info:
                        workflows_info.append((workflow_info))
                else:
                    for model in matcher_info:
                        for workflow_info in matcher_info[model]:
                            workflows_info.append((workflow_info))

        self.visualize_roc(workflows_info, drop_tp_indices=drop_tp_indices)


    def evaluate_auc_roc(self, matchers : List,
            drop_tp_indices=True) -> None:
        """For each matcher,
        takes its prediction data,
        calculates cumulative recall and auc,
        plots the corresponding ROC curve,
        populates prediction data with performance info

        Args:
            matchers List[ProgressiveMatching]: Progressive Matchers

        Raises:
            AttributeError: No Data object
            AttributeError: No Ground Truth file
        """

        if self.encoded_data is None:
            raise AttributeError("Can not proceed to AUC ROC evaluation without data object.")

        if self.encoded_data.skip_ground_truth:
            raise AttributeError("Can not proceed to AUC ROC" \
            " evaluation without a ground-truth file. " +
            "Data object has not been initialized with the ground-truth file")

        self.matchers_info = []

        for matcher in matchers:
            _tp_indices, _total_emissions  = self.calculate_tps_indices(pairs=matcher.pairs,
                                                duplicate_of=matcher.duplicate_of,
                                                duplicate_emitted=matcher.duplicate_emitted)
            matcher_info = {}
            matcher_info['name'] = generate_unique_identifier()
            matcher_info['total_emissions'] = _total_emissions
            matcher_info['tp_idx'] = _tp_indices
            matcher_info['time'] = matcher.execution_time

            matcher_prediction_data : PredictionData = PredictionData(matcher=matcher,
                                                            matcher_info=matcher_info)
            matcher.set_prediction_data(matcher_prediction_data)
            self.matchers_info.append(matcher_info)

        self.visualize_roc(methods_data=self.matchers_info, drop_tp_indices=drop_tp_indices)
