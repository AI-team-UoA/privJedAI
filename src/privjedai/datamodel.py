"""Datamodel of pprl.
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
from ordered_set import OrderedSet
import pandas as pd

class PPRLFeature(ABC):
    """
    Abstract base class for Privacy-Preserving Record Linkage (PPRL) feature methods.

    This class defines the interface and common functionality for all PPRL feature
    extraction and evaluation methods. Concrete subclasses must implement the
    abstract methods for specific feature evaluation techniques.

    Attributes:
        _method_name (str): Formal name of the method.
        _method_info (str): Detailed description of the method.
        _method_short_name (str): Abbreviated name for the method.
        execution_time (float): Time taken to execute the method in seconds.
    """

    _method_name: str
    _method_info: str
    _method_short_name: str

    def __init__(self) -> None:
        super().__init__()
        self.execution_time: float

    @abstractmethod
    def _configuration(self) -> dict:
        pass

    @abstractmethod
    def evaluate(self,
                 prediction: dict,
                 export_to_df: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        """
        Evaluate the PPRL method on given predictions.

        This abstract method must be implemented by subclasses to perform
        the actual evaluation of the PPRL feature method.

        Args:
            prediction: The prediction results to evaluate. Format depends on
                        the specific implementation.
            export_to_df (bool): If True, export results as pandas DataFrame.
            with_classification_report (bool): If True, include detailed
                classification metrics in the output.
            verbose (bool): If True, print progress and results to console.

        Returns:
            any: Evaluation results. The exact return type depends on the
                    export flags and specific implementation. Could be a
                    DataFrame, dictionary, or custom result object.

        Raises:
            NotImplementedError: If not implemented by subclass.
            ValueError: If prediction data is invalid or missing.
        """

    def method_configuration(self) -> dict:
        """Returns configuration details
        """
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }

    def report(self) -> None:
        """Prints method configuration
        """
        _configuration = self._configuration()
        parameters = ("\n" + ''.join([f'\t{k}: {v}\n' for k, v in _configuration.items()])) \
                        if len(self._configuration().items()) != 0 else ' None'
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nParameters: " + parameters +
            f"\nRuntime: {self.execution_time:2.4f} seconds"
        )


class Block:
    """The main module used for storing entities in the blocking steps of pyjedai module. \
        Consists of 2 sets of profile entities 1 for Dirty ER and 2 for Clean-Clean ER.
    """
    def __init__(self) -> None:
        # self.entities = {"D0": OrderedSet, "D1": OrederedSet, ...}
        self.entities : Dict[str, OrderedSet] = defaultdict(OrderedSet)

    def get_cardinality(self) -> int:
        """Returns block cardinality.

        Args:
            is_dirty_er (bool): Dirty or Clean-Clean ER.

        Returns:
            int: Cardinality
        """
        return len(self.entities['D0']) * len(self.entities['D1'])

    def __str__(self):
        return f"{dict(self.entities)}"


class EncodedData(ABC):
    """
    Abstract base class representing encoded data for Privacy-Preserving Record Linkage (PPRL).

    This class provides a common interface for handling encoded or anonymized data
    in PPRL workflows. It stores the encoded representations of entities along with
    metadata and ground truth information for evaluation purposes.

    Attributes:
        encoded_dict (Dict[int, Dict[str, any]]): Dictionary containing encoded entities.
            Keys are entity identifiers, values are dictionaries of encoded attributes.
        bounds (List): List containing boundary information for the encoded data,
            typically used for spatial or range-based encodings.
        skip_ground_truth (bool): Flag indicating whether ground truth data should
            be processed or skipped during operations.
        ground_truth (pd.DataFrame): DataFrame containing the ground truth matching
            pairs for evaluation purposes.
    """
    encoded_dict: Dict[int, Dict[str, any]]
    bounds : List
    skip_ground_truth : bool
    ground_truth: pd.DataFrame

    def get_cardinality(self) -> int:
        """
        Get the number of encoded entities in the dataset.

        Returns:
            int: The cardinality (number of entities) in the encoded dataset.

        """
        return len(self.encoded_dict)

    @abstractmethod
    def to_file(self, filename: str):
        """
        Save the encoded data to a file.

        This abstract method must be implemented by subclasses to provide
        serialization functionality for the encoded data.

        Args:
            filename (str): Path to the output file where encoded data will be saved.

        """

    @classmethod
    @abstractmethod
    def from_file(cls, *kwargs):
        """
        Load encoded data from a file.

        This abstract class method must be implemented by subclasses to provide
        deserialization functionality for encoded data files.

        Args:
            *kwargs: Variable length argument list. Specific arguments depend on
                    the implementation but typically include:
                filename (str): Path to the input file containing encoded data.
                Other implementation-specific parameters.

        Returns:
            EncodedData: An instance of the specific EncodedData subclass loaded
                        with data from the file.
        """
