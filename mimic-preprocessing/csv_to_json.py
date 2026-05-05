"""Convert CSV clinical data to per-patient JSON sequences.

Step 1 of the MIMIC preprocessing pipeline. Reads a CSV file of clinical
events (vital signs, lab results, notes, diagnoses, chart text), groups
them by patient, and writes one JSON file per patient containing a
chronologically ordered sequence of narrative event objects.

Usage:
    python -m preprocessing.csv_to_json \
        --input-csv /path/to/events.csv \
        --output-dir /path/to/output/ \
        --n-jobs 100
"""

import argparse
import json
import logging
import multiprocessing
import os

import pandas as pd
from typing import Tuple, Dict, Any

CATEGORY_MAP = {
    'VITAL': "Vital signs",
    'LAB': "Laboratory test results",
    'NOTE': "Clinical notes",
    'DIAGNOSIS': "New diagnosis",
    'CHART_TEXT': "Nursing observation"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV clinical data to per-patient JSON format")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--n-jobs", type=int, default=100, help="Number of parallel workers")
    return parser.parse_args()

def format_event_content(group: pd.DataFrame, category: str, timestamp: pd.Timestamp) -> str:
    category_desc = CATEGORY_MAP.get(category, category)
    narrative_parts = [f"{category_desc}:"]
    
    for _, row in group.iterrows():
        item = row['item_name']
        
        if category in ['LAB', 'VITAL']:
            val = row['content_value']
            unit = row['content_unit']
            
            if pd.isna(unit):
                unit = ""
            
            if pd.isna(val):
                val_str = "N/A"
            else:
                val_str = f"{val} {unit}".strip()
            
            narrative_parts.append(f"{item} is {val_str}")
            
        elif category in ['NOTE', 'DIAGNOSIS', 'CHART_TEXT']:
            text = row['content_text']
            
            if pd.isna(text):
                text = ""
            
            text = str(text).strip().replace('\n', ' ')
            
            if category == 'DIAGNOSIS':
                narrative_parts.append(f"Patient diagnosed with {item}")
            else:
                narrative_parts.append(f"{item}: {text}")

    return "; ".join(narrative_parts) + "."

def process_single_patient(args: Tuple[int, pd.DataFrame, str]) -> int:
    subject_id, patient_df, output_folder = args
    
    try:
        patient_df.sort_values(by=['event_time', 'category'], kind='mergesort', inplace=True)
        
        patient_json: Dict[str, Any] = {
            "patient_id": str(subject_id),
            "sequence": []
        }
        
        sub_group = patient_df.groupby(['event_time', 'category'], sort=False)
        events_list = []
        
        for (timestamp, category), group in sub_group:
            full_content = format_event_content(group, category, timestamp)
            
            event_obj = {
                "timestamp": timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                "event_content": full_content,
                "metadata": {
                    "category": category,
                    "item_count": len(group)
                }
            }
            events_list.append(event_obj)
        
        events_list.sort(key=lambda x: x['timestamp'])
        patient_json["sequence"] = events_list
        
        output_path = os.path.join(output_folder, f"{subject_id}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(patient_json, f, indent=4, ensure_ascii=False)
            
        return 1
        
    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {e}")
        return 0

def main():
    args = parse_args()
    input_csv = args.input_csv
    output_dir = args.output_dir
    n_jobs = args.n_jobs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Number of Workers: {n_jobs}")
    logger.info("Reading CSV file...")
    
    df = pd.read_csv(input_csv)
    
    logger.info("Converting timestamps and preprocessing...")
    df['event_time'] = pd.to_datetime(df['event_time'], dayfirst=True)
    df['item_name'] = df['item_name'].fillna('Unknown')
    
    logger.info("Grouping by subject_id...")
    grouped = df.groupby('subject_id', sort=False)
    
    tasks = ((sid, group, output_dir) for sid, group in grouped)
    total_patients = len(grouped)
    
    logger.info(f"Total patients found: {total_patients}. Starting parallel processing...")
    
    counter = 0
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for result in pool.imap_unordered(process_single_patient, tasks, chunksize=10):
            counter += result
            if counter % 100 == 0:
                print(f"Progress: Processed {counter}/{total_patients} patients...", flush=True)
                
    logger.info(f"Processing complete. Successfully processed: {counter}/{total_patients}")

if __name__ == "__main__":
    main()
