"""Embed patient event texts into vectors using multi-GPU parallel processing.

Step 4 of the MIMIC preprocessing pipeline. Reads patient JSON files
containing textual event sequences, computes embeddings for each event
using a transformer embedding model, and writes the result as per-patient
Parquet files with embedded event vectors. Distributes work across all
available GPUs using multiprocessing.

Usage:
    python -m preprocessing.data_embed_parallel \
        --model-path /path/to/Qwen3-Embedding-8B \
        --input-json-dir /path/to/json/input/ \
        --output-parquet-dir /path/to/parquet/output/ \
        --problem-dir /path/to/problem_files/ \
        --label 0 \
        --embedding-dim 4096
"""

import argparse
import json
import logging
import math
import shutil

import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Embed patient event texts into vectors with multi-GPU parallelism")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the embedding model")
    parser.add_argument("--input-json-dir", type=str, required=True, help="Path to input JSON directory")
    parser.add_argument("--output-parquet-dir", type=str, required=True, help="Path to output Parquet directory")
    parser.add_argument("--problem-dir", type=str, required=True, help="Path to directory for problem files")
    parser.add_argument("--label", type=int, default=0, help="Label value to assign to all patients")
    parser.add_argument("--embedding-dim", type=int, default=4096, help="Expected embedding dimension")
    return parser.parse_args()

class EmbeddingClient:
    def __init__(self, model_name_or_path: str, device_id: int = 0):
        self.device = f"cuda:{device_id}"
        print(f"[GPU {device_id}] Loading model: {model_name_or_path} ...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side='left', trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.float16, 
            device_map=self.device
        )
        self.model.eval()

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            batch_indices = torch.arange(batch_size, device=last_hidden_states.device)
            return last_hidden_states[batch_indices, sequence_lengths]

    def embed_batch(self, texts: List[str], batch_size: int = 140) -> Optional[List[List[float]]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                batch_dict = self.tokenizer(
                    batch_texts, max_length=8192, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    all_embeddings.extend(embeddings.float().cpu().numpy().tolist())
            except Exception as e:
                print(f"Embedding error: {e}")
                return None
        return all_embeddings

def serialize_content(content: Union[str, Dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "exams_findings" in content or "exams_result" in content:
            name = content.get("exams_name", "")
            findings = content.get("exams_findings", "")
            result = content.get("exams_result", "")
            return f"{name}\n{findings}\n{result}".strip()
        else:
            return " ".join([str(v).strip() for v in content.values() if isinstance(v, (str, int, float))])
    return ""

def process_file_error(json_file: Path, problem_dir: Path, error_msg: str):
    problem_dir.mkdir(parents=True, exist_ok=True)
    dest_path = problem_dir / json_file.name
    try:
        shutil.copy(json_file, dest_path)
        logging.warning(f"Error processing {json_file.name}: {error_msg}. Moved to {dest_path}")
    except Exception as e:
        logging.error(f"Failed to copy problem file {json_file.name}: {e}")

def gpu_worker_process(rank: int, model_path: str, file_list: List[Path], output_dir: Path, problem_dir: Path, label: int, embedding_dim: int):
    logging.basicConfig(
        level=logging.INFO, 
        format=f'%(asctime)s - [GPU {rank}] - %(levelname)s - %(message)s',
        force=True
    )
    worker_logger = logging.getLogger(f"Worker-{rank}")
    worker_logger.info(f"Worker started. Files assigned: {len(file_list)}")
    
    try:
        embedder = EmbeddingClient(model_name_or_path=model_path, device_id=rank)
    except Exception as e:
        worker_logger.error(f"Model initialization failed on GPU {rank}: {e}")
        return

    for idx, json_file in enumerate(file_list):
        patient_id = json_file.stem
        if (idx + 1) % 10 == 0:
            worker_logger.info(f"Progress: {idx+1}/{len(file_list)} - Processing {patient_id}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_events = data.get("sequence", [])
            if not raw_events:
                continue

            valid_event_buffer = [] 
            for event in raw_events:
                ts_str = event.get("timestamp", "")
                try:
                    if 'T' in ts_str:
                        ts = pd.to_datetime(ts_str).timestamp()
                    else:
                        ts = pd.to_datetime(ts_str, format="%Y-%m-%d %H:%M:%S").timestamp()
                except ValueError:
                    continue 

                content = event.get("event_content", "")
                text_to_embed = serialize_content(content)
                valid_event_buffer.append((ts, text_to_embed))

            if not valid_event_buffer:
                continue

            texts_to_embed = [item[1] for item in valid_event_buffer]
            embeddings_list = embedder.embed_batch(texts_to_embed, batch_size=256)

            if embeddings_list is None or len(embeddings_list) != len(texts_to_embed):
                error_msg = f"Embedding mismatch (Input: {len(texts_to_embed)}, Output: {len(embeddings_list) if embeddings_list else 'None'})"
                worker_logger.error(error_msg)
                process_file_error(json_file, problem_dir, error_msg)
                continue

            processed_events = []
            for i, embedding_vector in enumerate(embeddings_list):
                ts, _ = valid_event_buffer[i]
                
                if len(embedding_vector) != embedding_dim:
                    worker_logger.warning(f"Unexpected embedding dimension: {len(embedding_vector)}")

                event_record = {
                    'timestamp': float(ts),
                    'embedding': embedding_vector
                }
                processed_events.append(event_record)

            processed_events.sort(key=lambda x: x['timestamp'])
            
            if processed_events:
                start_time = processed_events[0]['timestamp']
                for e in processed_events:
                    e['timestamp'] = e['timestamp'] - start_time

            final_patient_data = {
                'patient_id': patient_id,
                'total_event': len(processed_events),
                'label': label, 
                'event': processed_events 
            }

            output_path = output_dir / f"{patient_id}.parquet"
            df_out = pd.DataFrame([final_patient_data])
            df_out.to_parquet(output_path, index=False, engine='pyarrow')
            
        except Exception as e:
            worker_logger.error(f"Failed to process {patient_id}: {e}")
            continue

    worker_logger.info("Worker finished.")

def main():
    args = parse_args()
    model_path = args.model_path
    input_json_dir = Path(args.input_json_dir)
    output_parquet_dir = Path(args.output_parquet_dir)
    problem_dir = Path(args.problem_dir)
    label = args.label
    embedding_dim = args.embedding_dim

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU detected for parallel processing.")
    
    print(f"Detected {num_gpus} GPUs. Starting parallel processing...")
    
    if not input_json_dir.exists():
        print(f"Input directory not found: {input_json_dir}")
        return
    
    output_parquet_dir.mkdir(parents=True, exist_ok=True)
    
    all_json_files = sorted(list(input_json_dir.glob("*.json")))
    total_files = len(all_json_files)
    print(f"Found {total_files} files to process.")

    chunk_size = math.ceil(total_files / num_gpus)
    chunks = [all_json_files[i : i + chunk_size] for i in range(0, total_files, chunk_size)]

    ctx = mp.get_context('spawn')
    processes = []

    for rank in range(num_gpus):
        if rank >= len(chunks):
            break
            
        file_chunk = chunks[rank]
        
        p = ctx.Process(
            target=gpu_worker_process,
            args=(rank, model_path, file_chunk, output_parquet_dir, problem_dir, label, embedding_dim)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print("All tasks completed.")

if __name__ == "__main__":
    main()
