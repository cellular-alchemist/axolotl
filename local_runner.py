#!/usr/bin/env python3
"""
Axolotl Local Runner
Script for local testing of the oscillation detection pipeline without S3 dependencies

Example Usage:
--------------
# Basic usage with local files
python local_runner.py --h5 data/recording.h5 --zip data/spikes.zip --npz data/lfp.npz --output results/

# Multiple oscillation types
python local_runner.py --h5 data/recording.h5 --zip data/spikes.zip --npz data/lfp.npz \
    --output results/ --oscillation-types sharp_wave_ripples fast_ripples

# Custom parameters
python local_runner.py --h5 data/recording.h5 --zip data/spikes.zip --npz data/lfp.npz \
    --output results/ --sample-name culture_001 --condition-name baseline

# Generate configuration file
python local_runner.py --generate-config --output results/

"""

import os
import sys
import yaml
import argparse
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add local modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from processor import OscillationProcessor
from visualizer import OscillationVisualizer
from aggregator import StatsAggregator
from utils.data_loader import load_recording_data


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration for local runner"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('axolotl.local')


def create_mock_config(sample_name: str = "local_sample", 
                      condition_name: str = "test_condition",
                      oscillation_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create a mock configuration for local testing.
    
    Parameters:
    -----------
    sample_name : str
        Name for the sample
    condition_name : str
        Name for the condition
    oscillation_types : list, optional
        List of oscillation types to include
        
    Returns:
    --------
    dict
        Mock configuration dictionary
    """
    
    if oscillation_types is None:
        oscillation_types = ['sharp_wave_ripples', 'fast_ripples', 'gamma_bursts']
    
    config = {
        'experiment': {
            'name': f'local_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            's3_output_base': 'local://output/',  # Placeholder for local
            'description': 'Local testing run of axolotl pipeline'
        },
        
        'oscillation_types': {},
        
        'processing': {
            'frequency_bands': [],
            'analysis_window': {
                'start': 0,
                'length': None
            }
        },
        
        'visualization': {
            'channels_per_plot': 5,
            'ripples_per_channel': 10,
            'time_window': 0.3,
            'save_format': ['png', 'svg']
        },
        
        'samples': [
            {
                'name': sample_name,
                'description': 'Local test sample',
                'conditions': {
                    condition_name: {
                        'description': 'Local test condition',
                        'files': {
                            'h5': 'local_file.h5',
                            'zip': 'local_file.zip',
                            'npz': 'local_file.npz'
                        }
                    }
                }
            }
        ]
    }
    
    # Add oscillation type configurations
    oscillation_configs = {
        'sharp_wave_ripples': {
            'narrowband_key': 'narrowRipples',
            'wideband_key': 'broadRipples',
            'low_threshold': 3.5,
            'high_threshold': 5.0,
            'min_duration': 20,
            'max_duration': 200,
            'require_sharp_wave': True,
            'sharp_wave_window': 50
        },
        'fast_ripples': {
            'narrowband_key': 'fastRipples',
            'wideband_key': 'ultraFastRipples',
            'low_threshold': 3.0,
            'high_threshold': 4.5,
            'min_duration': 10,
            'max_duration': 100,
            'require_sharp_wave': False
        },
        'gamma_bursts': {
            'narrowband_key': 'gammaEnvelope',
            'wideband_key': 'gammaBand',
            'low_threshold': 2.5,
            'high_threshold': 4.0,
            'min_duration': 50,
            'max_duration': 500,
            'require_sharp_wave': False
        }
    }
    
    # Add frequency bands for requested oscillation types
    frequency_bands = []
    for osc_type in oscillation_types:
        if osc_type in oscillation_configs:
            config['oscillation_types'][osc_type] = oscillation_configs[osc_type]
            
            # Add corresponding frequency bands
            osc_config = oscillation_configs[osc_type]
            
            if osc_type == 'sharp_wave_ripples':
                frequency_bands.extend([
                    {'name': 'narrowRipples', 'low': 150, 'high': 250},
                    {'name': 'broadRipples', 'low': 80, 'high': 250}
                ])
            elif osc_type == 'fast_ripples':
                frequency_bands.extend([
                    {'name': 'fastRipples', 'low': 250, 'high': 500},
                    {'name': 'ultraFastRipples', 'low': 200, 'high': 600}
                ])
            elif osc_type == 'gamma_bursts':
                frequency_bands.extend([
                    {'name': 'gammaEnvelope', 'low': 30, 'high': 100},
                    {'name': 'gammaBand', 'low': 25, 'high': 140}
                ])
    
    # Remove duplicates from frequency bands
    seen_bands = set()
    unique_bands = []
    for band in frequency_bands:
        band_tuple = (band['name'], band['low'], band['high'])
        if band_tuple not in seen_bands:
            seen_bands.add(band_tuple)
            unique_bands.append(band)
    
    config['processing']['frequency_bands'] = unique_bands
    
    return config


def add_frequency_bands(lfp_processor, config: Dict[str, Any], logger: logging.Logger):
    """Add required frequency bands to LFP processor"""
    bands = config['processing']['frequency_bands']
    
    for band in bands:
        logger.debug(f"Adding frequency band: {band['name']} ({band['low']}-{band['high']} Hz)")
        
        # Use CPU for most bands to avoid GPU memory issues in local testing
        use_gpu = False  # Set to True if you have GPU and want to use it
        
        lfp_processor.add_frequency_band(
            band['low'], 
            band['high'], 
            band_name=band['name'],
            use_gpu=use_gpu,
            store_analytical=False
        )


def process_local_files(h5_path: str, zip_path: str, npz_path: str,
                       output_dir: Path, config: Dict[str, Any],
                       logger: logging.Logger) -> Dict[str, Any]:
    """
    Process local files through the oscillation detection pipeline.
    
    Parameters:
    -----------
    h5_path : str
        Path to H5 file
    zip_path : str
        Path to ZIP file
    npz_path : str
        Path to NPZ file
    output_dir : Path
        Output directory
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    dict
        Processing results
    """
    
    logger.info("Starting local oscillation detection pipeline")
    
    try:
        # Prepare file paths
        file_paths = {
            'h5': h5_path,
            'zip': zip_path,
            'npz': npz_path
        }
        
        # Validate files exist
        for file_type, path in file_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{file_type.upper()} file not found: {path}")
            logger.info(f"Found {file_type.upper()} file: {path}")
        
        # Load data and create LFP processor
        logger.info("Loading recording data...")
        lfp_processor = load_recording_data(file_paths)
        
        # Add required frequency bands
        logger.info("Adding frequency bands...")
        add_frequency_bands(lfp_processor, config, logger)
        
        # Initialize processors
        oscillation_processor = OscillationProcessor()
        visualizer = OscillationVisualizer()
        
        # Get sample and condition info
        sample = config['samples'][0]
        sample_name = sample['name']
        condition_name = list(sample['conditions'].keys())[0]
        
        # Process each oscillation type
        results = {}
        for osc_type, osc_params in config['oscillation_types'].items():
            logger.info(f"Processing oscillation type: {osc_type}")
            
            try:
                # Run oscillation detection
                results[osc_type] = oscillation_processor.process_oscillations(
                    lfp_processor, osc_params, osc_type, sample_name, condition_name
                )
                
                # Save results locally
                result_file = output_dir / f"{osc_type}_results.npz"
                import numpy as np
                np.savez_compressed(result_file, **results[osc_type])
                logger.info(f"Saved {osc_type} results to {result_file}")
                
            except Exception as e:
                logger.error(f"Failed to process {osc_type}: {str(e)}")
                continue
        
        # Generate visualizations
        if results:
            logger.info("Generating visualizations...")
            viz_output_dir = output_dir / "visualizations"
            viz_output_dir.mkdir(exist_ok=True)
            
            visualizer.visualize_all_oscillations(
                results, viz_output_dir, config, lfp_processor, 
                sample_name, condition_name
            )
            
            # Create visualization archive
            zip_path = visualizer.create_visualization_zip(viz_output_dir)
            if zip_path:
                logger.info(f"Created visualization archive: {zip_path}")
        
        # Create summary report
        logger.info("Creating summary report...")
        _create_local_summary(results, output_dir, config, logger)
        
        return {
            'status': 'success',
            'sample': sample_name,
            'condition': condition_name,
            'oscillation_types_processed': list(results.keys()),
            'output_directory': str(output_dir),
            'processing_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'processing_time': datetime.now().isoformat()
        }


def _create_local_summary(results: Dict[str, Any], output_dir: Path,
                         config: Dict[str, Any], logger: logging.Logger):
    """Create a local summary report"""
    
    try:
        summary_path = output_dir / "local_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("AXOLOTL LOCAL PROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {config['experiment']['name']}\n")
            f.write(f"Sample: {config['samples'][0]['name']}\n")
            f.write(f"Condition: {list(config['samples'][0]['conditions'].keys())[0]}\n\n")
            
            f.write("OSCILLATION DETECTION RESULTS\n")
            f.write("-" * 30 + "\n\n")
            
            for osc_type, result in results.items():
                summary_stats = result.get('summary_stats', {})
                f.write(f"{osc_type.replace('_', ' ').title()}:\n")
                f.write(f"  Total Detections: {summary_stats.get('total_detections', 0)}\n")
                f.write(f"  Detection Rate: {summary_stats.get('mean_rate', 0):.3f} events/s\n")
                f.write(f"  Mean Duration: {summary_stats.get('mean_duration', 0):.1f} ms\n")
                f.write(f"  Mean Power: {summary_stats.get('mean_power', 0):.2f} z-score\n")
                f.write(f"  Channels with Detections: {summary_stats.get('channels_with_detections', 0)}\n\n")
            
            f.write("OUTPUT FILES\n")
            f.write("-" * 15 + "\n")
            f.write("- Individual oscillation results: *_results.npz\n")
            f.write("- Visualizations: visualizations/ directory\n")
            f.write("- Visualization archive: visualizations.zip\n")
            f.write("- This summary: local_summary.txt\n")
        
        logger.info(f"Created local summary: {summary_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create local summary: {str(e)}")


def generate_example_config(output_dir: Path):
    """Generate an example configuration file"""
    
    config = create_mock_config(
        sample_name="example_sample",
        condition_name="example_condition",
        oscillation_types=['sharp_wave_ripples', 'fast_ripples', 'gamma_bursts']
    )
    
    # Update file paths to be examples
    config['samples'][0]['conditions']['example_condition']['files'] = {
        'h5': '/path/to/raw_recording.h5',
        'zip': '/path/to/spike_data.zip',
        'npz': '/path/to/lfp_data.npz'
    }
    
    config_path = output_dir / "example_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Generated example configuration: {config_path}")
    return config_path


def main():
    """Main entry point for local runner"""
    
    parser = argparse.ArgumentParser(
        description='Axolotl Local Runner - Test oscillation detection pipeline locally',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # File inputs
    parser.add_argument('--h5', type=str, help='Path to H5 raw data file')
    parser.add_argument('--zip', type=str, help='Path to ZIP spike data file')
    parser.add_argument('--npz', type=str, help='Path to NPZ LFP data file')
    
    # Configuration options
    parser.add_argument('--sample-name', type=str, default='local_sample',
                       help='Name for the sample (default: local_sample)')
    parser.add_argument('--condition-name', type=str, default='test_condition',
                       help='Name for the condition (default: test_condition)')
    parser.add_argument('--oscillation-types', nargs='+', 
                       choices=['sharp_wave_ripples', 'fast_ripples', 'gamma_bursts'],
                       default=['sharp_wave_ripples', 'fast_ripples'],
                       help='Oscillation types to detect (default: sharp_wave_ripples fast_ripples)')
    
    # Output options
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    
    # Special modes
    parser.add_argument('--generate-config', action='store_true',
                       help='Generate example configuration file and exit')
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration YAML file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate config mode
    if args.generate_config:
        generate_example_config(output_dir)
        return
    
    # Validate required arguments
    if not args.config and not all([args.h5, args.zip, args.npz]):
        parser.error("Either --config or all of --h5, --zip, --npz must be provided")
    
    try:
        # Load or create configuration
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract file paths from config
            sample = config['samples'][0]
            condition = list(sample['conditions'].values())[0]
            h5_path = condition['files']['h5']
            zip_path = condition['files']['zip']
            npz_path = condition['files']['npz']
        else:
            logger.info("Creating mock configuration")
            config = create_mock_config(
                sample_name=args.sample_name,
                condition_name=args.condition_name,
                oscillation_types=args.oscillation_types
            )
            
            h5_path = args.h5
            zip_path = args.zip
            npz_path = args.npz
        
        # Save configuration for reference
        config_path = output_dir / "used_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Saved configuration to: {config_path}")
        
        # Process files
        result = process_local_files(
            h5_path, zip_path, npz_path, output_dir, config, logger
        )
        
        # Print summary
        if result['status'] == 'success':
            logger.info("Processing completed successfully!")
            logger.info(f"Results saved to: {output_dir}")
            logger.info(f"Processed oscillation types: {', '.join(result['oscillation_types_processed'])}")
        else:
            logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Copilot was here