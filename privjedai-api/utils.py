"""Utils for privJedAI API"""

import tempfile
import time
from typing import List, Literal, Optional, Union
from fastapi import File, Form, UploadFile
from fastapi.exceptions import HTTPException
import pandas as pd
import numpy as np

from pydantic import ValidationError

from schemas import (
    LSHBlockingAuto,
    LSHBlockingManual,
    SchemaBloomFilterConfig,
    LSHWorkflow,
    SchemaBlockFiltering,
    SchemaBlockPurging,
    Combination,
    Metablocking,
    Matching,
    Clustering,
    PrivJedAIWorkflow,
    IVF,
    SchemaFAISSBlocking,
    FAISSWorkflow,
    HNSW,
    Flat,
    BruteForceWorkflow,
)



from privjedai.blocking import BitBlocker, FAISSBlocking
from privjedai.encoder import BloomFilter, BloomFilterConfig
from privjedai.encoded_data import BloomEncodedData
from privjedai.block_filtering import BlockFiltering, BlockPurging
from privjedai.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST,
)
from privjedai.matching import Matcher
from privjedai.clustering import (
    ConnectedComponentsClustering,
    UniqueMappingClustering,
    KiralyMSMApproximateClustering,
)


clustering_methods = {
    "Connected Components Clustering": ConnectedComponentsClustering,
    "Unique Mapping Clustering": UniqueMappingClustering,
    "Kiraly Clustering": KiralyMSMApproximateClustering,
}

cc_methods = {
    "Weighted Edge Pruning": WeightedEdgePruning,
    "Weighted Node Pruning": WeightedNodePruning,
    "Cardinality Edge Pruning": CardinalityEdgePruning,
    "Cardinality Node Pruning": CardinalityNodePruning,
    "BLAST": BLAST,
}


def clean_numpy_types(obj):
    """Recursively converts NumPy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: clean_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_numpy_types(i) for i in obj]
    return obj


def bloom_filter_config_form(
    file: UploadFile = File(...),
    separator: str = Form(...),
    bloom_filter_size: int = Form(...),
    number_of_hashes: int = Form(...),
    hashing_type: Literal[
        "salted_string",
        "salted_qgrams",
        "salted_skipqgrams",
        "salted_metaphone",
        "salted_tokens",
    ] = Form("salted_string"),
    qgrams: Optional[int] = Form(2),
    salt: Optional[str] = Form(""),
    attributes: Optional[List[str]] = Form(None),
) -> SchemaBloomFilterConfig:
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Dependency Check"""
    return SchemaBloomFilterConfig(
        file=file,
        separator=separator,
        bloom_filter_size=bloom_filter_size,
        number_of_hashes=number_of_hashes,
        hashing_type=hashing_type,
        qgrams=qgrams,
        salt=salt,
        attributes=attributes,
    )


# --- 3. Background Task Logic ---


def process_and_save_bloom_filter(
    config: BloomFilterConfig, df: pd.DataFrame, temp_pkl: str
):
    """Executes the heavy encoding outside the main request thread."""
    bloom_generator = BloomFilter(config)

    # Assuming encode returns an object that has the .to_file() method
    encoded_d1 = bloom_generator.encode(df)
    encoded_d1.to_file(temp_pkl)


async def load_encoded_data(encoded_d1: UploadFile, encoded_d2: UploadFile):
    """Function to load the encoded_data"""

    # Initialize paths so the 'finally' block doesn't crash if an error occurs early
    temp_p1_path = None
    temp_p2_path = None

    # 1. Create temporary files on disk
    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_p1,
        tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_p2,
    ):

        temp_p1.write(await encoded_d1.read())
        temp_p2.write(await encoded_d2.read())

        temp_p1_path = temp_p1.name
        temp_p2_path = temp_p2.name

    # 2. Use your custom class with the physical file paths
    bloom_encoded_data = BloomEncodedData.from_file(temp_p1_path, temp_p2_path)

    return bloom_encoded_data, temp_p1_path, temp_p2_path


def execute_lsh(
    blocking_config: Union[LSHBlockingAuto, LSHBlockingManual],
    encoded_data: BloomEncodedData,
):
    """Executes the LSH algorithm."""

    lsh_execution = {}
    lsh_execution["method"] = "LSH Blocking"

    blocks = None

    bit_blocker_config = None
    if isinstance(blocking_config, LSHBlockingAuto):
        bit_blocker_config = BitBlocker.auto_psi_lambda(
            encoded_data, threshold=blocking_config.auto_threshold
        )
    else:
        bit_blocker_config = {}
        bit_blocker_config["psi"] = blocking_config.psi
        bit_blocker_config["lambda_"] = blocking_config.lambda_

    psi = bit_blocker_config["psi"]
    lambda_ = bit_blocker_config["lambda_"]

    start = time.time()

    lsh_blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
    blocks = lsh_blocker.build_blocks(encoded_data)
    end = time.time()

    lsh_execution["runtime"] = end - start
    lsh_execution["configuration"] = bit_blocker_config

    if not encoded_data.skip_ground_truth:
        ev = lsh_blocker.evaluate(verbose=False)
        lsh_execution["evaluation"] = ev

    return lsh_execution, blocks


def execute_filtering(
    filtering_config: Union[SchemaBlockFiltering, SchemaBlockPurging, Combination],
    encoded_data: BloomEncodedData,
):
    """Executes the filtering process."""

    blocks = None

    filtering_execution = {}

    filtering_execution["method"] = ""

    start = time.time()

    filtering_method = None

    if isinstance(filtering_config, (SchemaBlockPurging, Combination)):
        purging = BlockPurging(filtering_config.smoothing_factor)
        blocks = purging.process(encoded_data)

        filtering_method = purging

        filtering_execution["method"] += "Block Purging"

    if isinstance(filtering_config, (SchemaBlockFiltering, Combination)):
        filtering = BlockFiltering(filtering_config.ratio)
        blocks = filtering.process(encoded_data)
        filtering_method = filtering

        if isinstance(filtering_config, Combination):
            filtering_execution["method"] += " + "

        filtering_execution["method"] += "Block Filtering"

    end = time.time()

    filtering_execution["runtime"] = end - start
    filtering_execution["configuration"] = filtering_config.model_dump_json(
        indent=2, exclude_none=True
    )

    if not encoded_data.skip_ground_truth:
        ev = filtering_method.evaluate(verbose=False)
        filtering_execution["evaluation"] = ev

    return filtering_execution, blocks


def execute_metablocking(
    metablocking_config: Metablocking, encoded_data: BloomEncodedData
):
    """Executes the metablocking process."""

    blocks = None
    metablocking_execution = {}

    metablocking_execution["method"] = metablocking_config.method

    start = time.time()

    cc = cc_methods[metablocking_config.method](metablocking_config.weighting_scheme)
    blocks = cc.process(encoded_data)

    end = time.time()

    metablocking_execution["runtime"] = end - start
    metablocking_execution["configuration"] = metablocking_config.model_dump_json(
        indent=2, exclude_none=True
    )

    if not encoded_data.skip_ground_truth:
        ev = cc.evaluate(blocks, verbose=False)
        metablocking_execution["evaluation"] = ev

    return metablocking_execution, blocks


def execute_matching(matching_config: Matching, encoded_data: BloomEncodedData, blocks):
    """Executes the matching process."""

    matching_execution = {}

    matching_execution["method"] = "Matching"

    start = time.time()

    # Perform matching

    matching = Matcher(
        batch_size=matching_config.batch_size,
        threshold=matching_config.threshold,
        metric=matching_config.metric,
    )
    graph = matching.predict(encoded_data, blocks)

    end = time.time()

    matching_execution["runtime"] = end - start
    matching_execution["configuration"] = matching_config.model_dump_json(
        indent=2, exclude_none=True
    )

    if not encoded_data.skip_ground_truth:
        ev = matching.evaluate(graph, verbose=False)
        matching_execution["evaluation"] = ev

    return matching_execution, graph


def execute_clustering(
    clustering_config: Clustering, encoded_data: BloomEncodedData, graph
):
    """Executes the clustering process."""

    clustering_execution = {}

    clustering_execution["method"] = "Clustering"

    start = time.time()

    # Perform clustering
    cl = clustering_methods[clustering_config.method]()
    pairs = cl.process(graph, encoded_data, clustering_config.threshold)

    pairs = [p for p in pairs if len(p) == 2]
    pairs_matrix = np.array(pairs)
    sorted_pairs = np.sort(pairs_matrix, axis=1)
    pairs = list(sorted_pairs)
    end = time.time()

    clustering_execution["runtime"] = end - start
    clustering_execution["configuration"] = clustering_config.model_dump_json(
        indent=2, exclude_none=True
    )

    if not encoded_data.skip_ground_truth:
        ev = cl.evaluate(pairs, verbose=False)
        clustering_execution["evaluation"] = ev

    return clustering_execution, pairs


def execute_lsh_workflow(workflow: LSHWorkflow, encoded_data: BloomEncodedData):
    """Executes the LSH workflow."""

    execution = {}

    blocking_config: Union[LSHBlockingAuto, LSHBlockingManual] = workflow.blocking
    execution["blocking"], blocks = execute_lsh(blocking_config, encoded_data)

    if workflow.filtering_method:
        filtering_config: Union[
            SchemaBlockFiltering, SchemaBlockPurging, Combination
        ] = workflow.filtering_method
        filtering_method, blocks = execute_filtering(filtering_config, encoded_data)
        execution["filtering_method"] = filtering_method
    if workflow.metablocking:
        metablocking_config = workflow.metablocking
        metablocking, blocks = execute_metablocking(metablocking_config, encoded_data)
        execution["metablocking"] = metablocking

    matching, graph = execute_matching(workflow.matching, encoded_data, blocks)
    execution["matching"] = matching
    edges, _ = graph

    pairs_df = pd.DataFrame(edges, columns=["id1", "id2"])

    if workflow.clustering:
        clustering, pairs = execute_clustering(workflow.clustering, encoded_data, graph)
        execution["clustering"] = clustering
        pairs_df = pd.DataFrame(pairs, columns=["id1", "id2"])

    pairs_df["id2"] = pairs_df["id2"] - encoded_data.bounds[0]
    execution = clean_numpy_types(execution)

    return execution, pairs_df


def execute_ivf(ivf_config: IVF, encoded_data: BloomEncodedData):
    """Executes the IVF blocking method."""

    execution = {}

    start = time.time()
    faiss = FAISSBlocking(index_type="ivf")

    faiss.configure_ivf(ivf_config.nlist, ivf_config.nprobe)
    blocks = faiss.build_blocks(encoded_data, top_k=ivf_config.top_k)
    end = time.time()
    execution["method"] = "FAISS IVF"
    execution["runtime"] = end - start
    execution["configuration"] = ivf_config.model_dump_json(indent=2, exclude_none=True)

    if not encoded_data.skip_ground_truth:
        # Convert sets to lists so NumPy can concatenate them during evaluation
        blocks_for_eval = {k: list(v) for k, v in blocks.items()}
        ev = faiss.evaluate(blocks_for_eval, verbose=False)
        execution["evaluation"] = ev

    return execution, blocks


def execute_hnsw(hnsw_config: HNSW, encoded_data: BloomEncodedData):
    """Executes the HNSW blocking method."""

    execution = {}

    start = time.time()
    faiss = FAISSBlocking(index_type="hnsw")

    faiss.configure_hsnw(hnsw_config.hnsw_m)
    blocks = faiss.build_blocks(encoded_data, top_k=hnsw_config.top_k)
    end = time.time()
    execution["method"] = "FAISS HNSW"
    execution["runtime"] = end - start
    execution["configuration"] = hnsw_config.model_dump_json(
        indent=2, exclude_none=True
    )

    if not encoded_data.skip_ground_truth:

        # Convert sets to lists so NumPy can concatenate them during evaluation
        blocks_for_eval = {k: list(v) for k, v in blocks.items()}

        ev = faiss.evaluate(blocks_for_eval, verbose=False)
        execution["evaluation"] = ev

    return execution, blocks


def execute_flat(flat_config: Flat, encoded_data: BloomEncodedData):
    """Executes the IVF blocking method."""

    execution = {}

    start = time.time()
    faiss = FAISSBlocking(index_type="flat")
    blocks = faiss.build_blocks(encoded_data, top_k=flat_config.top_k)

    print(blocks)
    end = time.time()
    execution["method"] = "FAISS Flat"
    execution["runtime"] = end - start
    execution["configuration"] = flat_config.model_dump_json(
        indent=2, exclude_none=True
    )

    if not encoded_data.skip_ground_truth:

        # Convert sets to lists so NumPy can concatenate them during evaluation
        blocks_for_eval = {k: list(v) for k, v in blocks.items()}

        ev = faiss.evaluate(blocks_for_eval, verbose=False)
        execution["evaluation"] = ev

    return execution, blocks


def execute_faiss(blocking_config: SchemaFAISSBlocking, encoded_data: BloomEncodedData):
    """Executes the FAISS blocking method."""

    if isinstance(blocking_config.method, IVF):
        return execute_ivf(blocking_config.method, encoded_data)
    if isinstance(blocking_config.method, HNSW):
        return execute_hnsw(blocking_config.method, encoded_data)

    return execute_flat(blocking_config.method, encoded_data)


def execute_faiss_workflow(workflow: FAISSWorkflow, encoded_data: BloomEncodedData):
    """Executes the FAISS workflow."""

    execution = {}

    blocking_config: SchemaFAISSBlocking = workflow.blocking
    blocking, blocks = execute_faiss(blocking_config, encoded_data)

    execution["blocking"] = blocking

    matching, graph = execute_matching(workflow.matching, encoded_data, blocks)
    execution["matching"] = matching
    edges, _ = graph

    pairs_df = pd.DataFrame(edges, columns=["id1", "id2"])

    if workflow.clustering:
        clustering, pairs = execute_clustering(workflow.clustering, encoded_data, graph)
        execution["clustering"] = clustering
        pairs_df = pd.DataFrame(pairs, columns=["id1", "id2"])

    pairs_df["id2"] = pairs_df["id2"] - encoded_data.bounds[0]
    execution = clean_numpy_types(execution)

    return execution, pairs_df


def execute_brute_force_worfklow(workflow, encoded_data):
    """Executes the brute force workflow."""

    execution = {}

    matching, graph = execute_matching(workflow.matching, encoded_data, None)
    execution["matching"] = matching
    edges, _ = graph

    pairs_df = pd.DataFrame(edges, columns=["id1", "id2"])

    if workflow.clustering:
        clustering, pairs = execute_clustering(workflow.clustering, encoded_data, graph)
        execution["clustering"] = clustering
        pairs_df = pd.DataFrame(pairs, columns=["id1", "id2"])

    pairs_df["id2"] = pairs_df["id2"] - encoded_data.bounds[0]
    execution = clean_numpy_types(execution)

    return execution, pairs_df


def execute_workflow_utils(config, encoded_data):
    """Executes the workflow in the background."""
    workflow = config.workflow_config
    # workflow = config

    if isinstance(workflow, LSHWorkflow):
        return execute_lsh_workflow(workflow, encoded_data)

    if isinstance(workflow, FAISSWorkflow):
        return execute_faiss_workflow(workflow, encoded_data)

    return execute_brute_force_worfklow(workflow, encoded_data)


LSHEXAMPLE = PrivJedAIWorkflow(
    workflow_id="1",
    workflow_name="Test Workflow",
    created_at="2023-01-01T00:00:00Z",
    updated_at="2023-01-01T00:00:00Z",
    user_id="1",
    workflow_config=LSHWorkflow(
        type="lsh",
        blocking=LSHBlockingAuto(type="auto", auto_threshold=0.5),
        filtering_method=SchemaBlockFiltering(type="block_filtering", ratio=0.8),
        metablocking=Metablocking(
            method="Weighted Edge Pruning", weighting_scheme="CN-CBS"
        ),
        matching=Matching(batch_size=10000, threshold=0.5, metric="dice"),
        # clustering=Clustering(method="Kiraly Clustering", threshold=0.5),
    ),
).model_dump_json(indent=2, exclude_none=True)

FAISSEXAMPLE = PrivJedAIWorkflow(
    workflow_id="1",
    workflow_name="Test Workflow",
    created_at="2023-01-01T00:00:00Z",
    updated_at="2023-01-01T00:00:00Z",
    user_id="1",
    workflow_config=FAISSWorkflow(
        type="faiss",
        blocking=SchemaFAISSBlocking(
            method=IVF(
                type="ivf",
                top_k=10,
                nlist=32,
                nprobe=8,
            )
        ),
        matching=Matching(batch_size=10000, threshold=0.5, metric="dice"),
        # clustering=Clustering(method="Kiraly Clustering", threshold=0.5),
    ),
).model_dump_json(indent=2, exclude_none=True)


BRUTEEXAMPLE = PrivJedAIWorkflow(
    workflow_id="1",
    workflow_name="Test Workflow",
    created_at="2023-01-01T00:00:00Z",
    updated_at="2023-01-01T00:00:00Z",
    user_id="1",
    workflow_config=BruteForceWorkflow(
        type="brute_force",
        matching=Matching(batch_size=10000, threshold=0.5, metric="dice"),
        # clustering=Clustering(method="Kiraly Clustering", threshold=0.5),
        #     ),
    ),
).model_dump_json(indent=2, exclude_none=True)


def parse_privjedai_workflow(
    config_json: str = Form(..., examples=[BRUTEEXAMPLE]),
) -> PrivJedAIWorkflow:
    """Dependency for parsing PrivJedAI workflow configurations."""
    try:
        return PrivJedAIWorkflow.model_validate_json(config_json)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors(include_url=False)) from e
