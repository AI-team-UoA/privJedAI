from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError, Field
from typing import List
import pandas as pd
import json
import os
import uuid
from typing import Optional, Literal

from privjedai.encoder import BloomFilterConfig, BloomEncodedData, BloomFilter

from privjedai.blocking import BitBlocker, FAISSBlocking
from privjedai.block_filtering import BlockFiltering, BlockPurging
from privjedai.comparison_cleaning import AbstractComparisonCleaning, WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, CardinalityNodePruning, BLAST
from privjedai.matching import Matcher


app = FastAPI(title="PrivjedAI API", description="API for PrivjedAI", version="1.0.0")

# Ensure the storage directory exists when the server starts
STORAGE_DIR = "encoded_data"
CONFIG_DIR = "config_data"

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)



metablocking_methods = {
    'WEP': WeightedEdgePruning,
    'WNP': WeightedNodePruning,
    'CEP': CardinalityEdgePruning,
    'CNP': CardinalityNodePruning,
    'BLAST': BLAST
}

class BloomFilterConfigRequest(BaseModel):
    size: int
    offset: int = 0
    num_hashes: int
    hashing_type: str = "salted_qgrams"
    salt: str = ""
    attributes: List[str]
    qgrams: int = 2



class LSHConfigModel(BaseModel):
    threshold: float = 0.5
    delta: float = 0.1
    max_lambda: int = 150

class FAISSConfigModel(BaseModel):
    top_k: int = 30
    index_type: Literal['flat', 'hnsw', 'multihash'] = 'flat'
    hnsw_m: Optional[int] = None  # Only relevant if index_type is 'hnsw'

class BlockFilteringConfigModel(BaseModel):
    ratio: float = 0.8

class BlockPurgingConfigModel(BaseModel):
    smoothing_factor : float = 1.025

class MetablockingConfigModel(BaseModel):
    method: Literal['WEP', 'WNP', 'CEP', 'CNP', 'BLAST'] = 'WEP'
    weighting_scheme: Literal['CN-CBS', 'CBS', 'SN-CBS', 'SNC', 'CNC',
                    'SND', 'CND', 'CNJ', 'SNJ', 'COSINE', 'DICE', 'ECBS',
                    'JS', 'EJS', 'X2'] = 'CN-CBS'

class MatchingConfigModel(BaseModel):
    threshold: float = 0.6
    metric: Literal["dice", "scm", 'jaccard', "cosine"] = "dice"

class ClusteringConfigModel(BaseModel):
    method: Literal['CCC', 'UMC'] = 'UMC'
    threshold: float = 0.5

class ConfigureModel(BaseModel):
    config_id: str = Field(..., description="Unique identifier for this configuration profile (e.g., 'job_experiment_1')")
    party_A: str
    party_B: str

    # Blocking
    LSH_config: Optional[LSHConfigModel] = None
    FAISS_config: Optional[FAISSConfigModel] = None

    # Block filtering and purging
    blockfiltering_config: Optional[BlockFilteringConfigModel] = None
    blockpurging_config: Optional[BlockPurgingConfigModel] = None

    # Metablocking
    metablocking_config: Optional[MetablockingConfigModel] = None

    # Matching
    matching_config: MatchingConfigModel  # This can be further detailed based on your matching logic

    # Clustering
    clustering_config: Optional[ClusteringConfigModel] = None


# Helper function to delete temporary files
def cleanup_files(*file_paths):
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)

@app.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint to verify that the API is running.
    """
    return {"status": "ok"}

@app.post("/encode")
async def encode_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Upload dataset as a CSV"),
    config_str: str = Form(..., description="JSON string of the configuration")
):
    # 1. Parse and validate the incoming JSON configuration
    try:
        config_dict = json.loads(config_str)
        valid_config = BloomFilterConfigRequest(**config_dict)
    except (json.JSONDecodeError, ValidationError) as e:
        return {"error": "Invalid configuration", "details": str(e)}

    # 2. Generate unique filenames to prevent users from overwriting each other's data
    job_id = str(uuid.uuid4())
    temp_csv = f"{job_id}_input.csv"
    temp_pkl = f"{job_id}_encoded.pkl"

    try:
        # 3. Save the uploaded CSV to the server
        with open(temp_csv, "wb") as buffer:
            buffer.write(await file.read())

        # 4. Load it into Pandas
        abt = pd.read_csv(temp_csv)

        # 5. Run your Python library logic
        # Use model_dump() to turn the validated Pydantic model back into a dictionary
        config = BloomFilterConfig(**valid_config.model_dump())
        bloom_generator = BloomFilter(config)

        encoded_d1 = bloom_generator.encode(abt)

        # Save the result to disk
        encoded_d1.to_file(temp_pkl)

        # 6. Schedule the cleanup function to run AFTER the response is sent
        background_tasks.add_task(cleanup_files, temp_csv, temp_pkl)

        # 7. Return the .pkl file to the user for download
        return FileResponse(
            path=temp_pkl,
            filename="encoded_dataset.pkl",
            media_type="application/octet-stream"
        )

    except Exception as e:
        # Clean up the CSV if something goes wrong before completion
        cleanup_files(temp_csv, temp_pkl)
        return {"error": "Processing failed", "details": str(e)}


@app.post("/upload")
async def upload_encoded_dataset(

    party_id: str = Form(..., description="Unique ID for the party (e.g., party_A)"),
    file: UploadFile = File(..., description="The encoded .pkl dataset")
):
    # Security check: ensure they are only uploading pickle files
    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Only .pkl files are allowed")

    # Save the file to our storage directory, named after the party
    file_path = os.path.join(STORAGE_DIR, f"{party_id}.pkl")

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        return {
            "status": "success",
            "message": f"Encoded dataset securely saved for {party_id}"
        }
    except Exception as e:
        return {"error": "Failed to save file", "details": str(e)}


@app.post("/configure", summary="Save or update a TTP matching configuration")
def save_configuration(config: ConfigureModel):
    """
    Saves a configuration profile on the TTP server so it can be
    reused across multiple matching runs without resending parameters.
    """
    config_path = os.path.join(CONFIG_DIR, f"{config.config_id}.json")

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=4)

        return {
            "status": "success",
            "message": f"Configuration '{config.config_id}' saved successfully.",
            "config": config.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist configuration: {str(e)}")

@app.post("/run_job", summary="Run a matching job based on a saved configuration")
def run_job(config_id: str):
    """
    Runs a matching job using a previously saved configuration.
    """
    config_path = os.path.join(CONFIG_DIR, f"{config_id}.json")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config = ConfigureModel(**config_data)


        party_a_pkl = os.path.join(STORAGE_DIR, f"{config.party_A}.pkl")
        party_b_pkl = os.path.join(STORAGE_DIR, f"{config.party_B}.pkl")

        encoded_data = BloomEncodedData.from_file(party_a_pkl, party_b_pkl)


        if config.LSH_config:
            lsh_config = config.LSH_config.model_dump()
            bit_blocker_config = BitBlocker.auto_psi_lambda(encoded_data, **lsh_config)
            psi = bit_blocker_config['psi']
            lambda_ = bit_blocker_config['lambda_']
            lsh_blocker = BitBlocker(psi=psi, lambda_=lambda_, seed=42)
            _ = lsh_blocker.build_blocks(encoded_data)

            if config.blockpurging_config:
                block_purging_config = config.blockpurging_config.model_dump()
                block_purger = BlockPurging(**block_purging_config)
                _ = block_purger.process(encoded_data)

            if config.blockfiltering_config:
                block_filtering_config = config.blockfiltering_config.model_dump()
                block_filterer = BlockFiltering(**block_filtering_config)
                _ = block_filterer.process(encoded_data)

            if config.metablocking_config:
                metablocking_config = config.metablocking_config.model_dump()
                cc : AbstractComparisonCleaning = metablocking_methods[metablocking_config['method']](metablocking_config['weighting_scheme'])
                _ = cc.process(encoded_data)

        if config.FAISS_config:
            faiss_config = config.FAISS_config.model_dump()
            faiss_blocker = FAISSBlocking(index_type=faiss_config['index_type'])
            if faiss_config['index_type'] == 'hnsw' and faiss_config['hnsw_m']:
                faiss_blocker.configure_hsnw(m=faiss_config['hnsw_m'])
            _ = faiss_blocker.build_blocks(encoded_data, top_k=faiss_config['top_k'])


        matching_config = config.matching_config.model_dump()
        matcher = Matcher(threshold=matching_config['threshold'], metric=matching_config['metric'])
        matches = matcher.predict(encoded_data)

        ev = matcher.evaluate(encoded_data, matches)

        if config.clustering_config:
            clustering_config = config.clustering_config.model_dump()
            if clustering_config['method'] == 'UMC':
                from privjedai.clustering import UniqueMappingClustering
                umc = UniqueMappingClustering()
                matches = umc.process(matches, encoded_data, threshold=clustering_config['threshold'])
                ev = umc.evaluate(encoded_data, matches)
            elif clustering_config['method'] == 'CCC':
                from privjedai.clustering import ConnectedComponentsClustering
                ccc = ConnectedComponentsClustering()
                matches = ccc.process(matches, encoded_data, threshold=clustering_config['threshold'])
                ev = ccc.evaluate(encoded_data, matches)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported clustering method: {clustering_config['method']}")

        return {
            "status": "success",
            "message": f"Job started with configuration '{config_id}'",
            "config": config.model_dump(),
            "evaluation_metrics": ev,
            "matches": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run job: {str(e)}")
