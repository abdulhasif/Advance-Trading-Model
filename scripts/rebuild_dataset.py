"""
scripts/rebuild_dataset.py — Full Data Refresh Orchestrator
===========================================================
1. Clears processed bricks (storage/data/) while keeping raw_ticks.
2. Clears computed features (storage/features/).
3. Re-downloads and processes bricks (Batch Factory).
4. Verifies and patches data anomalies (Data Verifier).
5. Re-computes all alpha factors (Feature Engine).

Usage: python scripts/rebuild_dataset.py
"""

import sys
import shutil
import logging
from pathlib import Path

# Bootstrap project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data.batch_factory import run_batch_factory
from src.data.data_verifier import DataVerifier
from src.data.feature_engine import run_feature_engine

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "rebuild_dataset.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def clear_processed_data():
    """Deletes processed bricks and features, preserving raw_ticks."""
    logger.info("Step 1: Clearing existing processed data...")
    
    # 1. Clear storage/data (preserve raw_ticks)
    if config.DATA_DIR.exists():
        for item in config.DATA_DIR.iterdir():
            if item.is_dir() and item.name != "raw_ticks":
                logger.info(f"  Removing directory: {item}")
                shutil.rmtree(item)
            elif item.is_file() and item.suffix == ".parquet":
                # Some files might be in the root of storage/data/
                logger.info(f"  Removing file: {item}")
                item.unlink()

    # 2. Clear storage/features
    if config.FEATURES_DIR.exists():
        logger.info(f"  Removing all features in: {config.FEATURES_DIR}")
        for item in config.FEATURES_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    
    logger.info("Cleanup complete.")

def main():
    logger.info("=" * 70)
    logger.info("FULL DATA REFRESH PIPELINE — STARTING")
    logger.info("=" * 70)
    
    try:
        # --- PHASE 1: CLEANUP ---
        clear_processed_data()
        
        # --- PHASE 2: DOWNLOAD & BRICK FORMATION ---
        # We ensure FORCE_REFRESH is True to avoid skipping existing blocks (if any leaked)
        setattr(config, "FORCE_REFRESH", True)
        logger.info("\nStep 2: Starting Batch Factory (Download -> Sanitization -> Renko)...")
        run_batch_factory()
        
        # --- PHASE 3: DATA VERIFICATION ---
        logger.info("\nStep 3: Starting Data Verification and Patching...")
        verifier = DataVerifier(fix=True, check_raw=True, check_features=False)
        verifier.run()
        
        # --- PHASE 4: FEATURE ENGINEERING ---
        # Ensure incremental is disabled for a full fresh compute
        setattr(config, "FEATURE_INCREMENTAL_ENABLED", False)
        logger.info("\nStep 4: Starting Feature Engine (Alpha Factors)...")
        run_feature_engine()
        
        # --- PHASE 5: FINAL VERIFICATION ---
        logger.info("\nStep 5: Final Check on Features...")
        final_verifier = DataVerifier(fix=True, check_raw=False, check_features=True)
        final_verifier.run()
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"\nCRITICAL PIPELINE FAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
