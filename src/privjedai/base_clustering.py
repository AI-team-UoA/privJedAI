"""
Docstring for pprl.clustering
"""

import numpy as np
from tqdm.autonotebook import tqdm
import pandas as pd
from privjedai.datamodel import PPRLFeature
from privjedai.encoded_data import BloomEncodedData
from privjedai.evaluation import Evaluation

RANDOM_SEED = 42


class EquivalenceCluster(PPRLFeature):
    """
    Docstring for EquivalenceCluster

    :var Args: Description
    :var Returns: Description
    :var Args: Description
    :var Returns: Description
    :var list: Description
    :vartype list: list
    :var Args: Description
    :var Args: Description
    :var Returns: Description
    :var list: Description
    :vartype list: list
    """


    def __init__(self, encoded_data : BloomEncodedData,
                flattened_cluster : list = None) -> None:
        super().__init__()
        self.encoded_data: BloomEncodedData = encoded_data
        self.d1_entities = set()
        self.d2_entities = set()
        if flattened_cluster:
            self.add_entities(flattened_cluster)

    def add_entities(self, entities : list) -> None:
        """
        Docstring for add_entities

        :param self: Description
        :param entities: Description
        :type entities: list
        """
        for entity in entities:
            target_set : set = self.d1_entities if entity < self.encoded_data.bounds[0] \
                else self.d2_entities
            target_set.add(entity)

    def get_entities(self) -> list:
        """
        Docstring for get_entities

        :param self: Description
        :return: Description
        :rtype: list
        """
        return list((self.get_d1_entities() | self.get_d2_entities()))

    def get_d1_entities(self) -> set:
        """
        Docstring for get_D1_entities

        :param self: Description
        :return: Description
        :rtype: set
        """
        return self.d1_entities

    def get_d2_entities(self) -> set:
        """
        Docstring for get_D2_entities

        :param self: Description
        :return: Description
        :rtype: set
        """
        return self.d2_entities

    def has_entities(self) -> bool:
        """
        Docstring for has_entities

        :param self: Description
        :return: Description
        :rtype: bool
        """
        return self.has_d1_entities() or self.has_d2_entities()

    def has_d1_entities(self) -> bool:
        """
        Docstring for has_D1_entities

        :param self: Description
        :return: Description
        :rtype: bool
        """
        return len(self.d1_entities) > 0

    def has_d2_entities(self) -> bool:
        """
        Docstring for has_D2_entities

        :param self: Description
        :return: Description
        :rtype: bool
        """
        return len(self.d2_entities) > 0

    def has_entity(self, entity : int) -> bool:
        """
        Docstring for has_entity

        :param self: Description
        :param entity: Description
        :type entity: int
        :return: Description
        :rtype: bool
        """

        target_set : set = self.d1_entities if entity < self.encoded_data.bounds[0] \
            else self.d2_entities

        return entity in target_set

    def remove_entity(self, entity: int) -> None:
        """
        Docstring for remove_entity

        :param self: Description
        :param entity: Description
        :type entity: int
        """
        target_set : set = self.d1_entities if entity < self.encoded_data.bounds[0] \
            else self.d2_entities
        target_set.remove(entity)

    def remove_entities(self, entities: list) -> None:
        """
        Docstring for remove_entities

        :param self: Description
        :param entities: Description
        :type entities: list
        """
        for entity in entities:
            self.remove_entity(entity)

    def flatten(self) -> list:
        """
        Docstring for flatten

        :param self: Description
        :return: Description
        :rtype: list
        """
        flattened_cluster : list = []

        for d1_entity in self.d1_entities:
            flattened_cluster.append(d1_entity)
        for d2_entity in self.d2_entities:
            flattened_cluster.append(d2_entity)

        return flattened_cluster

    def evaluate(self,
                prediction: dict,
                export_to_df: bool = False,
                with_classification_report: bool = False,
                verbose: bool = True) -> any:
        """
        Docstring for evaluate

        :param self: Description
        :param prediction: Description
        :param export_to_df: Description
        :type export_to_df: bool
        :param export_to_dict: Description
        :type export_to_dict: bool
        :param with_classification_report: Description
        :type with_classification_report: bool
        :param verbose: Description
        :type verbose: bool
        :return: Description
        :rtype: Any
        """

    def _configuration(self) -> dict:
        pass

    def stats(self) -> None:
        """
        Docstring for stats

        :param self: Description
        """

class ExtendedSimilarityEdge(PPRLFeature):
    """
    Docstring for ExtendedSimilarityEdge
    """
    left_node : int
    right_node: int
    similarity : float
    active : bool
    def __init__(self,
                 left_node : int,
                 right_node : int,
                 similarity : float,
                 active : bool = True) -> None:
        super().__init__()
        self.set_left_node(left_node=left_node)
        self.set_right_node(right_node=right_node)
        self.set_similarity(similarity=similarity)
        self.set_active(active=active)

    def set_left_node(self, left_node : int):
        """
        Docstring for set_left_node

        :param self: Description
        :param left_node: Description
        :type left_node: int
        """
        self.left_node : int = left_node

    def set_right_node(self, right_node : int):
        """
        Docstring for set_right_node

        :param self: Description
        :param right_node: Description
        :type right_node: int
        """
        self.right_node : int = right_node

    def set_similarity(self, similarity : float):
        """
        Docstring for set_similarity

        :param self: Description
        :param similarity: Description
        :type similarity: float
        """
        self.similarity : float = similarity

    def set_active(self, active : bool):
        """
        Docstring for set_active

        :param self: Description
        :param active: Description
        :type active: bool
        """
        self.active : bool = active

    def is_active(self):
        """
        Docstring for is_active

        :param self: Description
        """
        return self.active

    def __lt__(self, other):
        return self.similarity < other.similarity

    def __le__(self, other):
        return self.similarity <= other.similarity

    def __eq__(self, other):
        return self.similarity == other.similarity

    def __ne__(self, other):
        return self.similarity != other.similarity

    def __gt__(self, other):
        return self.similarity > other.similarity

    def __ge__(self, other):
        return self.similarity >= other.similarity

    def evaluate(self,
                prediction: dict,
                export_to_df: bool = False,
                with_classification_report: bool = False,
                verbose: bool = True) -> any:
        """
        Docstring for evaluate

        :param self: Description
        :param prediction: Description
        :param export_to_df: Description
        :type export_to_df: bool
        :param export_to_dict: Description
        :type export_to_dict: bool
        :param with_classification_report: Description
        :type with_classification_report: bool
        :param verbose: Description
        :type verbose: bool
        :return: Description
        :rtype: Any
        """

    def _configuration(self) -> dict:
        pass

    def stats(self) -> None:
        """
        Docstring for stats

        :param self: Description
        """


class Vertex(PPRLFeature):
    """
    Docstring for Vertex
    """
    average_weight : float

    def __init__(self,
                 identifier : int,
                 edges : list = None) -> None:
        super().__init__()
        self.set_identifier(identifier=identifier)
        self.set_attached_edges(attached_edges=0)
        self.set_weight_sum(weight_sum=0)
        self.set_edges(edges={})
        if edges is not None:
            self.insert_edges(edges=edges)

    def set_identifier(self, identifier : int) -> None:
        """
        Docstring for set_identifier

        :param self: Description
        :param identifier: Description
        :type identifier: int
        """
        self.identifier : int = identifier

    def set_attached_edges(self, attached_edges : int) -> None:
        """
        Docstring for set_attached_edges

        :param self: Description
        :param attached_edges: Description
        :type attached_edges: int
        """
        self.attached_edges : int = attached_edges

    def set_weight_sum(self, weight_sum : float) -> None:
        """
        Docstring for set_weight_sum

        :param self: Description
        :param weight_sum: Description
        :type weight_sum: float
        """
        self.weight_sum : float = weight_sum

    def set_edges(self, edges : dict) -> None:
        """
        Docstring for set_edges

        :param self: Description
        :param edges: Description
        :type edges: dict
        """
        self.edges : dict = edges

    def set_average_weight(self, average_weight : float) -> None:
        """
        Docstring for set_average_weight

        :param self: Description
        :param average_weight: Description
        :type average_weight: float
        """
        self.average_weight : float = average_weight

    def insert_edges(self, edges : list) -> None:
        """
        Docstring for insert_edges

        :param self: Description
        :param edges: Description
        :type edges: list
        """
        for edge in edges:
            self.insert_edge(edge=edge)

    def insert_edge(self, edge : tuple) -> None:
        """
        Docstring for insert_edge

        :param self: Description
        :param edge: Description
        :type edge: tuple
        """
        vertex, weight = edge
        self.update_weight_sum_by(update_value=weight)
        self.update_attached_edges_by(update_value=1)
        self.edges[vertex] = weight
        self.update_average_weight()

    def remove_edges(self, edges : list) -> None:
        """
        Docstring for remove_edges

        :param self: Description
        :param edges: Description
        :type edges: list
        """
        for edge in edges:
            self.remove_edge(edge=edge)

    def remove_edge(self, edge : int) -> None:
        """
        Docstring for remove_edge

        :param self: Description
        :param edge: Description
        :type edge: int
        """
        weight = self.edges.pop(edge, None)
        if weight is not None:
            self.update_attached_edges_by(update_value=-1)
            self.update_weight_sum_by(update_value=-weight)
            self.update_average_weight()

    def get_attached_edges(self) -> int:
        """
        Docstring for get_attached_edges

        :param self: Description
        :return: Description
        :rtype: int
        """
        return self.attached_edges

    def get_weight_sum(self) -> float:
        """
        Docstring for get_weight_sum

        :param self: Description
        :return: Description
        :rtype: float
        """
        return self.weight_sum

    def get_edges(self) -> list:
        """
        Docstring for get_edges

        :param self: Description
        :return: Description
        :rtype: list
        """
        return self.edges

    def get_identifier(self) -> int:
        """
        Docstring for get_identifier

        :param self: Description
        :return: Description
        :rtype: int
        """
        return self.identifier

    def get_similarity_with(self, entity : int) -> float:
        """
        Docstring for get_similarity_with

        :param self: Description
        :param entity: Description
        :type entity: int
        :return: Description
        :rtype: float
        """
        return self.edges[entity] if entity in self.edges else 0.0

    def update_weight_sum_by(self, update_value : float) -> None:
        """
        Docstring for update_weight_sum_by

        :param self: Description
        :param update_value: Description
        :type update_value: float
        """
        self.set_weight_sum(self.get_weight_sum() + update_value)

    def update_attached_edges_by(self, update_value : float) -> None:
        """
        Docstring for update_attached_edges_by

        :param self: Description
        :param update_value: Description
        :type update_value: float
        """
        self.set_attached_edges(self.get_attached_edges() + update_value)

    def update_average_weight(self, negative = True) -> None:
        """
        Docstring for update_average_weight

        :param self: Description
        :param negative: Description
        """
        _average_weight : float = (self.get_weight_sum() / self.get_attached_edges())
        _average_weight = -_average_weight if negative else _average_weight
        self.set_average_weight(average_weight=_average_weight)

    def has_edges(self):
        """
        Docstring for has_edges

        :param self: Description
        """
        return self.get_attached_edges() > 0

    def __lt__(self, other):
        return self.average_weight < other.average_weight

    def __le__(self, other):
        return self.average_weight <= other.average_weight

    def __eq__(self, other):
        return self.average_weight == other.average_weight

    def __ne__(self, other):
        return self.average_weight != other.average_weight

    def __gt__(self, other):
        return self.average_weight > other.average_weight

    def __ge__(self, other):
        return self.average_weight >= other.average_weight

    def evaluate(self,
                prediction: dict,
                export_to_df: bool = False,
                with_classification_report: bool = False,
                verbose: bool = True) -> any:
        """
        Docstring for evaluate

        :param self: Description
        :param prediction: Description
        :param export_to_df: Description
        :type export_to_df: bool
        :param export_to_dict: Description
        :type export_to_dict: bool
        :param with_classification_report: Description
        :type with_classification_report: bool
        :param verbose: Description
        :type verbose: bool
        :return: Description
        :rtype: Any
        """


    def _configuration(self) -> dict:
        pass

    def stats(self) -> None:
        """
        Docstring for stats

        :param self: Description
        """

class RicochetCluster(PPRLFeature):
    """
    Docstring for RicochetCluster
    """
    def __init__(self,
                 center : int,
                 members : list) -> None:
        super().__init__()
        self.set_center(center=center)
        self.set_members(members=set())
        self.add_members(new_members=members)

    def set_center(self, center : int) -> None:
        """
        Docstring for set_center

        :param self: Description
        :param center: Description
        :type center: int
        """
        self.center : int = center

    def set_members(self, members : set) -> None:
        """
        Docstring for set_members

        :param self: Description
        :param members: Description
        :type members: set
        """
        self.members : set = members

    def add_members(self, new_members : list) -> None:
        """
        Docstring for add_members

        :param self: Description
        :param new_members: Description
        :type new_members: list
        """
        for new_member in new_members:
            self.add_member(new_member)

    def add_member(self, new_member: int) -> None:
        """
        Docstring for add_member

        :param self: Description
        :param new_member: Description
        :type new_member: int
        """
        self.members.add(new_member)

    def remove_member(self, member : int) -> None:
        """
        Docstring for remove_member

        :param self: Description
        :param member: Description
        :type member: int
        """
        self.members.remove(member)

    def get_members(self) -> list:
        """
        Docstring for get_members

        :param self: Description
        :return: Description
        :rtype: list
        """
        return self.members

    def get_center(self) -> int:
        """
        Docstring for get_center

        :param self: Description
        :return: Description
        :rtype: int
        """
        return self.center

    def change_center(self, new_center : int):
        """
        Docstring for change_center

        :param self: Description
        :param new_center: Description
        :type new_center: int
        """
        self.remove_member(member=self.get_center())
        self.add_member(new_member=new_center)
        self.set_center(center=new_center)

    def evaluate(self,
            prediction: dict,
            export_to_df: bool = False,
            with_classification_report: bool = False,
            verbose: bool = True) -> any:
        """
        Docstring for evaluate

        :param self: Description
        :param prediction: Description
        :param export_to_df: Description
        :type export_to_df: bool
        :param export_to_dict: Description
        :type export_to_dict: bool
        :param with_classification_report: Description
        :type with_classification_report: bool
        :param verbose: Description
        :type verbose: bool
        :return: Description
        :rtype: Any
        """

    def _configuration(self) -> dict:
        pass

    def stats(self) -> None:
        """
        Docstring for stats

        :param self: Description
        """

class AbstractClustering(PPRLFeature):
    """
    Docstring for AbstractClustering

    :var Args: Description
    :var Returns: Description
    """
    _method_name: str = "Abstract Clustering"
    _method_short_name: str = "AC"
    _method_info: str = "Abstract Clustering Method"
    blocks: dict
    similarity_threshold : float

    def __init__(self) -> None:
        super().__init__()
        self.encoded_data: BloomEncodedData
        self.similarity_threshold: float = 0.1
        self.execution_time: float = 0.0

    def _get_valid_edges_and_weights(self, graph : tuple) -> list:
        edges, weights = graph
        mask = weights > self.similarity_threshold
        valid_edges = edges[mask]
        valid_weights = weights[mask]

        return valid_edges, valid_weights


    def evaluate(self,
            prediction: dict,
            export_to_df: bool = False,
            with_classification_report: bool = False,
            verbose: bool = True) -> any:
        """
        Docstring for evaluate

        :param self: Description
        :param prediction: Description
        :param export_to_df: Description
        :type export_to_df: bool
        :param export_to_dict: Description
        :type export_to_dict: bool
        :param with_classification_report: Description
        :type with_classification_report: bool
        :param verbose: Description
        :type verbose: bool
        :return: Description
        :rtype: Any
        """

        if prediction is None:
            if self.blocks is None:
                raise AttributeError("Can not proceed to evaluation without build_blocks.")
            eval_blocks = self.blocks
        else:
            eval_blocks = prediction

        if self.encoded_data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.encoded_data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.encoded_data)
        true_positives = 0
        entity_index = eval_obj.create_entity_index_from_clusters(eval_blocks)


        id1_arr = self.encoded_data.ground_truth.iloc[:,0].values
        id2_arr = self.encoded_data.ground_truth.iloc[:,1].values + self.encoded_data.bounds[0]

        clusters_1 = entity_index[id1_arr]
        clusters_2 = entity_index[id2_arr]

        true_positives = int(np.count_nonzero((clusters_1 == clusters_2 ) & (clusters_1 != -1)))

        # true_positives


        # for _, (id1, id2) in self.encoded_data.ground_truth.iterrows():
        #     id2 += self.encoded_data.bounds[0]
        #     if id1 in entity_index and    \
        #         id2 in entity_index and entity_index[id1] == entity_index[id2]:
        #         true_positives += 1
        eval_obj.calculate_scores(true_positives=true_positives)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                with_classification_report,
                                verbose)

    def stats(self) -> None:
        """
        Docstring for stats

        :param self: Description
        """

    def export_to_df(self, prediction: list, tqdm_enable:bool = False) -> pd.DataFrame:
        """Creates a dataframe for the evaluation report.

        Args:
            prediction (list): Predicted clusters.

        Returns:
            pd.DataFrame: Dataframe containing evaluation scores and stats.
        """
        pairs_list = []

        dataset_limit = self.encoded_data.bounds[0]

        for cluster in tqdm(prediction, desc="Exporting to DataFrame", disable=not tqdm_enable):
            lcluster = list(cluster)

            for i1, node1 in enumerate(lcluster):
                for i2, node2 in enumerate(lcluster):
                    if i1 <= i2:
                        continue

                    if node1 < dataset_limit:
                        id1 = node1
                        id2 = node2 - dataset_limit
                    else:
                        id2 = node1 - dataset_limit
                        id1 = node2

                    pairs_list.append((id1, id2))

        pairs_df = pd.DataFrame(pairs_list, columns=['index_1', 'index_2'])

        return pairs_df


    def sorted_indicators(self, first_indicator : int, second_indicator : int):
        """
        Docstring for sorted_indicators

        :param self: Description
        :param first_indicator: Description
        :type first_indicator: int
        :param second_indicator: Description
        :type second_indicator: int
        """
        return (first_indicator, second_indicator) if first_indicator < second_indicator \
            else (second_indicator, first_indicator)

    def id_to_index(self, identifier : int):
        """
        Docstring for id_to_index

        :param self: Description
        :param identifier: Description
        :type identifier: int
        """
        return identifier \
            if identifier < self.encoded_data.dataset_limit \
            else identifier - self.encoded_data.dataset_limit

    def index_to_id(self, index : int, left_dataset : True):
        """
        Docstring for index_to_id

        :param self: Description
        :param index: Description
        :type index: int
        :param left_dataset: Description
        :type left_dataset: True
        """
        return index if left_dataset else index + self.encoded_data.dataset_limit

    def _configuration(self) -> dict:
        return {
            "Similarity Threshold": self.similarity_threshold
        }
