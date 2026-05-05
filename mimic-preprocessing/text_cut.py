"""Truncate long event texts and filter patients by event count.

Step 3 of the MIMIC preprocessing pipeline. Performs three operations
on each patient JSON in-place:
  1. Splits event_content strings exceeding max_text_length into chunks.
  2. Deletes patients with fewer than min_events events.
  3. Truncates patients to the most recent max_events events.

Usage:
    python -m preprocessing.text_cut \
        --target-folder /path/to/json/files/ \
        --max-text-length 5000 \
        --min-events 50 \
        --max-events 512
"""

import argparse
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("splitter")

def parse_args():
    parser = argparse.ArgumentParser(description="Truncate long texts and filter patients by event count")
    parser.add_argument("--target-folder", type=str, required=True, help="Path to directory containing patient JSON files")
    parser.add_argument("--max-text-length", type=int, default=5000, help="Maximum character length per event_content before splitting")
    parser.add_argument("--min-events", type=int, default=50, help="Minimum number of events to keep a patient")
    parser.add_argument("--max-events", type=int, default=512, help="Maximum number of events (truncated from the end)")
    return parser.parse_args()

def split_text_by_length(text: str, max_len: int) -> List[str]:
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]

def process_patient_json(file_path: Path, max_text_len: int = 5000, min_events: int = 50, max_events: int = 512):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_sequence = data.get("sequence", [])
        
        if not raw_sequence:
            os.remove(file_path)
            logger.info(f"Deleted (empty sequence): {file_path.name}")
            return

        new_sequence = []
        is_modified = False

        for event in raw_sequence:
            content = event.get("event_content", "")
            
            if isinstance(content, str) and len(content) > max_text_len:
                is_modified = True
                chunks = split_text_by_length(content, max_text_len)
                
                for chunk in chunks:
                    new_event = deepcopy(event)
                    new_event["event_content"] = chunk
                    new_sequence.append(new_event)
            else:
                new_sequence.append(event)

        if len(new_sequence) < min_events:
            os.remove(file_path)
            logger.info(f"Deleted (too few events < {min_events}): {file_path.name} (Current: {len(new_sequence)})")
            return

        new_sequence.sort(key=lambda x: x.get("timestamp", ""))

        if len(new_sequence) > max_events:
            is_modified = True
            original_len = len(new_sequence)
            new_sequence = new_sequence[-max_events:]
            logger.info(f"Truncated: {file_path.name} ({original_len} -> {len(new_sequence)} events)")

        if not is_modified:
            return

        data["sequence"] = new_sequence
        data["total_event"] = len(new_sequence)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Failed to process file {file_path.name}: {e}")

def main():
    args = parse_args()
    folder_path = args.target_folder
    max_text_len = args.max_text_length
    min_events = args.min_events
    max_events = args.max_events

    target_dir = Path(folder_path)
    if not target_dir.exists():
        logger.error(f"Directory not found: {target_dir}")
        return

    json_files = list(target_dir.glob("*.json"))
    logger.info(f"Starting processing directory: {target_dir}")
    logger.info(f"Config: Max Text Len={max_text_len}, Min Events={min_events}, Max Events={max_events}")
    logger.info(f"Found {len(json_files)} JSON files")
    
    count = 0
    for json_file in json_files:
        process_patient_json(json_file, max_text_len, min_events, max_events)
        count += 1
        if count % 100 == 0:
            print(f"Scanned {count} files...", flush=True)

    logger.info("All files processed.")

if __name__ == "__main__":
    main()
