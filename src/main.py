#!/usr/bin/env python3
"""
Axolotl - Main Pipeline Entry Point
Minimal oscillation detection pipeline for neural recordings
"""

import os
import sys
import yaml
import logging
import argparse
import traceback
import tempfile
import shutil
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

def make_json_serializable(obj):
    """Convert pandas DataFrames and numpy arrays to JSON serializable format"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'to_dict'):  # pandas DataFrame
        return obj.to_dict('records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

# Configure matplotlib for headless environment
import matplotlib
matplotlib.use('Agg')

# Add local modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor import OscillationProcessor
from visualizer import OscillationVisualizer  
from aggregator import StatsAggregator
from utils.s3_handler import S3Handler
from utils.data_loader import load_recording_data


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('axolotl')


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_s3_result_path(config: Dict, sample_name: str, condition_name: str, 
                      oscillation_type: str) -> str:
    """Generate S3 path for oscillation detection results"""
    base_path = config['experiment']['s3_output_base']
    exp_name = config['experiment']['name']
    return f"{base_path}/{exp_name}/oscillation_data/{sample_name}/{condition_name}/{oscillation_type}.npz"


def process_sample_condition(sample: Dict, condition_name: str, condition: Dict,
                           config: Dict, s3_handler: S3Handler, 
                           oscillation_processor: OscillationProcessor,
                           work_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Process a single sample/condition combination"""
    
    logger.info(f"Processing {sample['name']}/{condition_name}")
    
    try:
        # Create temporary directory for this condition
        temp_dir = work_dir / f"{sample['name']}_{condition_name}"
        temp_dir.mkdir(exist_ok=True)
        
        # Download required files
        logger.info(f"  Downloading files for {condition_name}")
        local_files = download_condition_files(condition['files'], temp_dir, s3_handler, logger)
        
        # Load data and create LFP processor
        logger.info(f"  Loading recording data for {condition_name}")
        lfp_processor = load_recording_data(local_files)
        
        # Add required frequency bands from config
        add_frequency_bands(lfp_processor, config, logger)
        
        # Process each oscillation type
        results = {}
        for osc_type, osc_params in config['oscillation_types'].items():
            logger.info(f"  Processing oscillation type: {osc_type}")
            
            # Check if results already exist in S3
            s3_result_path = get_s3_result_path(config, sample['name'], condition_name, osc_type)
            
            if s3_handler.check_exists(s3_result_path):
                logger.info(f"    Loading existing results from S3: {osc_type}")
                # Download existing results
                local_result_path = temp_dir / f"{osc_type}.npz"
                s3_handler.download_file(s3_result_path, str(local_result_path))
                results[osc_type] = dict(np.load(local_result_path, allow_pickle=True))
            else:
                logger.info(f"    Running detection for: {osc_type}")
                # Run oscillation detection
                results[osc_type] = oscillation_processor.process_oscillations(
                    lfp_processor, osc_params, osc_type, sample['name'], condition_name
                )
                
                # Save results locally
                local_result_path = temp_dir / f"{osc_type}.npz"
                save_results_to_npz(results[osc_type], str(local_result_path))
                
                # Upload to S3
                logger.info(f"    Uploading results to S3: {osc_type}")
                s3_handler.upload_file(str(local_result_path), s3_result_path)
        
        # Generate visualizations
        logger.info(f"  Generating visualizations for {condition_name}")
        visualizer = OscillationVisualizer()
        viz_output_dir = temp_dir / "visualizations"
        viz_output_dir.mkdir(exist_ok=True)
        
        visualizer.visualize_all_oscillations(
            results, viz_output_dir, config, lfp_processor,
            sample['name'], condition_name
        )
        
        # Upload visualization results
        viz_s3_path = f"{config['experiment']['s3_output_base']}/{config['experiment']['name']}/visualizations/{sample['name']}/{condition_name}/"
        s3_handler.upload_directory(str(viz_output_dir), viz_s3_path)
        
        return {
            'status': 'success',
            'sample': sample['name'],
            'condition': condition_name,
            'oscillation_types_processed': list(results.keys()),
            'oscillation_results': results,  # Include the actual results for aggregation
            'processing_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Error processing {sample['name']}/{condition_name}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            'status': 'failed',
            'sample': sample['name'],
            'condition': condition_name,
            'error': error_msg,
            'processing_time': datetime.now().isoformat()
        }
    
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def download_condition_files(files: Dict, temp_dir: Path, s3_handler: S3Handler, 
                           logger: logging.Logger) -> Dict[str, str]:
    """Download all required files for a condition"""
    local_files = {}
    
    file_mapping = {
        'h5': ('raw.h5', files['h5']),
        'zip': ('spike.zip', files['zip']),
        'npz': ('lfp.npz', files['npz'])
    }
    
    for file_type, (local_name, s3_path) in file_mapping.items():
        local_path = temp_dir / local_name
        logger.debug(f"    Downloading {file_type}: {s3_path}")
        
        if not s3_handler.download_file(s3_path, str(local_path)):
            if file_type == 'zip':
                # Spike data is optional - continue without it
                logger.warning(f"Spike data not available: {s3_path}")
                continue
            else:
                raise RuntimeError(f"Failed to download {file_type} file from {s3_path}")
        
        local_files[file_type] = str(local_path)
    
    return local_files


def add_frequency_bands(lfp_processor, config: Dict, logger: logging.Logger):
    """Add required frequency bands to LFP processor"""
    bands = config['processing']['frequency_bands']
    
    for band in bands:
        logger.debug(f"    Adding frequency band: {band['name']} ({band['low']}-{band['high']} Hz)")
        
        # Use GPU for larger bands, CPU for smaller ones
        use_gpu = (band['high'] - band['low']) > 100
        
        lfp_processor.add_frequency_band(
            band['low'], 
            band['high'], 
            band_name=band['name'],
            use_gpu=use_gpu,
            store_analytical=False
        )


def save_results_to_npz(results: Dict, file_path: str):
    """Save oscillation detection results to NPZ file"""
    np.savez_compressed(file_path, **results)


def aggregate_statistics(all_results: List[Dict], config: Dict, s3_handler: S3Handler,
                        logger: logging.Logger):
    """Generate aggregate statistics across all samples and conditions"""
    logger.info("Generating aggregate statistics")
    
    try:
        # Initialize aggregator
        aggregator = StatsAggregator()
        
        # Aggregate data across conditions
        aggregated_data = aggregator.aggregate_across_conditions(all_results, config)
        
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create violin plots
            aggregator.create_violin_plots(aggregated_data, temp_path)
            
            # Export statistics to CSV
            aggregator.export_summary_statistics(aggregated_data, temp_path)
            
            # Create summary report
            aggregator.create_summary_report(aggregated_data, temp_path, config)
            
            # Upload aggregate results to S3
            agg_s3_path = f"{config['experiment']['s3_output_base']}/{config['experiment']['name']}/aggregate_statistics/"
            s3_handler.upload_directory(str(temp_path / "aggregate_statistics"), agg_s3_path)
        
        # Save processing summary
        summary_path = f"{config['experiment']['s3_output_base']}/{config['experiment']['name']}/processing_summary.json"
        
        summary_data = {
            'experiment_name': config['experiment']['name'],
            'processing_completed': datetime.now().isoformat(),
            'total_conditions_processed': len(all_results),
            'successful_conditions': len([r for r in all_results if r['status'] == 'success']),
            'failed_conditions': len([r for r in all_results if r['status'] == 'failed']),
            'results': make_json_serializable(all_results)
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(summary_data, f, indent=2)
            temp_path = f.name
        
        s3_handler.upload_file(temp_path, summary_path)
        os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"Failed to generate aggregate statistics: {str(e)}")
        logger.error(traceback.format_exc())


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Axolotl Oscillation Detection Pipeline')
    parser.add_argument('config_path', help='Path to experiment configuration YAML file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--work-dir', default='/workspace/output', help='Working directory for temporary files')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Axolotl Oscillation Detection Pipeline")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config_path}")
        config = load_config(args.config_path)
        
        # Initialize S3 handler
        logger.info("Initializing S3 handler")
        # Use braingeneers S3 endpoint or environment variable override
        s3_endpoint = os.environ.get('AWS_ENDPOINT_URL', 'https://s3.braingeneers.gi.ucsc.edu')
        s3_handler = S3Handler(endpoint_url=s3_endpoint)
        
        # Initialize oscillation processor
        oscillation_processor = OscillationProcessor()
        
        # Create working directory
        work_dir = Path(args.work_dir)
        work_dir.mkdir(exist_ok=True)
        
        # Process each sample and condition
        all_results = []
        
        for sample in config['samples']:
            logger.info(f"Processing sample: {sample['name']}")
            
            for condition_name, condition in sample['conditions'].items():
                result = process_sample_condition(
                    sample, condition_name, condition, config, 
                    s3_handler, oscillation_processor, work_dir, logger
                )
                all_results.append(result)
        
        # Generate aggregate statistics
        aggregate_statistics(all_results, config, s3_handler, logger)
        
        # Print summary
        successful = len([r for r in all_results if r['status'] == 'success'])
        total = len(all_results)
        logger.info(f"Pipeline completed: {successful}/{total} conditions processed successfully")
        
        if successful < total:
            logger.warning(f"{total - successful} conditions failed processing")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()