"""Split embedded patient data into train/val/test sets (8:1:1 ratio).

Step 5 of the MIMIC preprocessing pipeline. Reads positive and negative
Parquet files from separate directories, performs an 8:1:1 random split
(stratified by class), and hard-links or copies the files into train,
validation, and test output directories.

Usage:
    python -m preprocessing.split_test_train \
        --src-pos /path/to/positive/parquets/ \
        --src-neg /path/to/negative/parquets/ \
        --target-train /path/to/train/ \
        --target-val /path/to/val/ \
        --target-test /path/to/test/
"""

import argparse
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Split embedded patient data into train/val/test sets (8:1:1)")
    parser.add_argument("--src-pos", type=str, required=True, help="Path to positive class Parquet directory")
    parser.add_argument("--src-neg", type=str, required=True, help="Path to negative class Parquet directory")
    parser.add_argument("--target-train", type=str, required=True, help="Path to train output directory")
    parser.add_argument("--target-val", type=str, required=True, help="Path to validation output directory")
    parser.add_argument("--target-test", type=str, required=True, help="Path to test output directory")
    return parser.parse_args()

def copy_single_file(args: Tuple[Path, Path]) -> int:
    src_path, target_dir = args
    dst_path = target_dir / src_path.name
    
    try:
        if dst_path.exists():
            os.remove(dst_path)
            
        os.link(src_path, dst_path)
        return 1
        
    except OSError:
        try:
            shutil.copy2(src_path, dst_path)
            return 1
        except Exception as copy_e:
            logger.error(f"Copy failed {src_path.name}: {copy_e}")
            return 0
            
    except Exception as e:
        logger.error(f"Link failed {src_path.name}: {e}")
        return 0

def fast_copy_files(file_list: List[Path], target_dir: Path, max_workers: int = 16) -> int:
    if not file_list:
        return 0
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = [(f, target_dir) for f in file_list]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(copy_single_file, tasks), 
                          total=len(tasks), 
                          desc=f"Linking to {target_dir.name}"))
        success_count = sum(results)
        
    return success_count

def split_and_copy_dataset_811(pos_dir: Path, neg_dir: Path, train_dir: Path, val_dir: Path, test_dir: Path):
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Target directories ready:\n -> Train: {train_dir}\n -> Val:   {val_dir}\n -> Test:  {test_dir}")

    pos_files = sorted(list(pos_dir.glob("*.parquet")))
    neg_files = sorted(list(neg_dir.glob("*.parquet")))
    
    logger.info(f"Source files scanned: Positive {len(pos_files)}, Negative {len(neg_files)}")
    
    if not pos_files and not neg_files:
        logger.error("No parquet files found. Please check source paths.")
        return

    pos_train, pos_temp = train_test_split(pos_files, test_size=0.2, random_state=None, shuffle=True)
    neg_train, neg_temp = train_test_split(neg_files, test_size=0.2, random_state=None, shuffle=True)
    
    pos_val, pos_test = train_test_split(pos_temp, test_size=0.5, random_state=None, shuffle=True)
    neg_val, neg_test = train_test_split(neg_temp, test_size=0.5, random_state=None, shuffle=True)
    
    train_files = pos_train + neg_train
    val_files = pos_val + neg_val
    test_files = pos_test + neg_test
    
    logger.info("-" * 40)
    logger.info(f"Split Plan (8:1:1) - Hard Link Mode:")
    logger.info(f"Train Set: {len(train_files)}")
    logger.info(f"Val Set  : {len(val_files)}")
    logger.info(f"Test Set : {len(test_files)}")
    logger.info("-" * 40)
    logger.info("Executing...")

    cnt_train = fast_copy_files(train_files, train_dir)
    logger.info(f"Train processing complete: {cnt_train}")

    cnt_val = fast_copy_files(val_files, val_dir)
    logger.info(f"Val processing complete: {cnt_val}")

    cnt_test = fast_copy_files(test_files, test_dir)
    logger.info(f"Test processing complete: {cnt_test}")

    logger.info("-" * 40)
    logger.info("Dataset split completed successfully.")

if __name__ == "__main__":
    args = parse_args()
    split_and_copy_dataset_811(
        Path(args.src_pos), 
        Path(args.src_neg), 
        Path(args.target_train), 
        Path(args.target_val), 
        Path(args.target_test)
    )
