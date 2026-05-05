"""Filter patient sequences to keep only events before the anchor event.

Step 2 of the MIMIC preprocessing pipeline. For each patient JSON, finds
the first event with flag==1 (the "anchor" event), truncates the sequence
to only include events before that anchor, and stores the anchor event
metadata separately. Files without an anchor event are kept unchanged
with Anchor set to None.

Usage:
    python -m preprocessing.filter_before_anchor \
        --input-dir /path/to/json/input/ \
        --output-dir /path/to/json/output/ \
        --max-workers 8
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def parse_args():
    parser = argparse.ArgumentParser(description="Filter patient sequences to events before the anchor")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to input JSON directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output JSON directory")
    parser.add_argument("--max-workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    return parser.parse_args()

def process_single_patient(file_path: str, output_path: str) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sequence = data.get("sequence", [])
        anchor_index = -1
        
        for i, event in enumerate(sequence):
            if event.get("flag") == 1:
                anchor_index = i
                break 
        
        if anchor_index != -1:
            anchor_event = sequence[anchor_index]
            truncated_sequence = sequence[:anchor_index]
            
            data["sequence"] = truncated_sequence
            data["Anchor"] = anchor_event
        else:
            data["Anchor"] = None 

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        return True

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

def _worker(args: Tuple[str, str]) -> bool:
    return process_single_patient(*args)

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.max_workers or os.cpu_count()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Output directory ready: {output_dir}")

    input_files = glob.glob(os.path.join(input_dir, "*.json"))
    total_files = len(input_files)
    
    if total_files == 0:
        print("No files found!")
        return

    print(f"Found {total_files} files. Starting processing with {max_workers} workers...")
    
    tasks: List[Tuple[str, str]] = []
    for file_path in input_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)
        tasks.append((file_path, output_path))

    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunk_size = max(1, total_files // (max_workers * 4)) if max_workers else 1
        results = executor.map(_worker, tasks, chunksize=chunk_size)
        
        if tqdm:
            list(tqdm(results, total=total_files, unit="file"))
        else:
            count = 0
            for _ in results:
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count}/{total_files}...")

    end_time = time.time()
    print(f"\nAll processing complete!")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
