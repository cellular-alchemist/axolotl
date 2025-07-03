"""
Axolotl Data Loader
Utility functions for loading neural recording data and creating LFPDataProcessor
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

# Import the required data loading functions and LFP processor
try:
    from loaders import load_curation, load_info_maxwell
    from new_lfp_processor_class import LFPDataProcessor
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise


def load_recording_data(file_paths: Dict[str, str]) -> 'LFPDataProcessor':
    """
    Load neural recording data from h5, zip, and npz files and create configured LFPDataProcessor.
    
    Parameters:
    -----------
    file_paths : dict
        Dictionary containing paths to data files:
        - 'h5': Path to raw Maxwell data file (.h5)
        - 'zip': Path to spike curation data file (.zip)
        - 'npz': Path to LFP/frequency band data file (.npz)
        
    Returns:
    --------
    LFPDataProcessor
        Configured LFP processor ready for oscillation detection
    """
    
    logger = logging.getLogger('axolotl.data_loader')
    
    try:
        # Validate required files exist
        _validate_file_paths(file_paths)
        
        # Load spike curation data
        logger.debug("Loading spike curation data")
        train, neuron_data, config, fs = load_curation(file_paths['zip'])
        train = [np.array(t) * 1000 for t in train]  # Convert to milliseconds
        
        # Load Maxwell raw data info
        logger.debug("Loading Maxwell raw data info")
        version, time_stamp, config_df, raster_df = load_info_maxwell(file_paths['h5'])
        
        # Load LFP/frequency band data
        logger.debug("Loading LFP data")
        waves = np.load(file_paths['npz'])
        
        # Extract electrode positions
        x_positions, y_positions = _extract_electrode_positions(waves, config_df, logger)
        
        # Create LFP processor
        logger.debug("Creating LFP processor")
        processor = LFPDataProcessor(waves, x_positions, y_positions, config_df)
        
        # Add basic required frequency bands
        _add_basic_frequency_bands(processor, logger)
        
        logger.info(f"Successfully loaded data with {len(x_positions)} electrodes")
        logger.info(f"Spike data: {len(train)} units, LFP data: {len(waves.files)} frequency bands")
        
        return processor
        
    except Exception as e:
        logger.error(f"Failed to load recording data: {str(e)}")
        raise


def _validate_file_paths(file_paths: Dict[str, str]) -> None:
    """
    Validate that all required file paths exist.
    
    Parameters:
    -----------
    file_paths : dict
        Dictionary of file paths to validate
    """
    
    required_files = ['h5', 'zip', 'npz']
    
    for file_type in required_files:
        if file_type not in file_paths:
            raise ValueError(f"Missing required file type: {file_type}")
        
        file_path = file_paths[file_type]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")


def _extract_electrode_positions(waves: np.lib.npyio.NpzFile, 
                                config_df: pd.DataFrame,
                                logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract electrode positions from the data.
    
    Parameters:
    -----------
    waves : np.lib.npyio.NpzFile
        Loaded NPZ file containing LFP data
    config_df : pd.DataFrame
        Configuration dataframe from Maxwell data
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    tuple
        (x_positions, y_positions) as numpy arrays
    """
    
    # Try to get positions from waves file first (preferred)
    if 'location' in waves.files:
        logger.debug("Using electrode positions from NPZ file")
        locations = waves['location']
        
        if locations.ndim == 2 and locations.shape[1] >= 2:
            x_positions = locations[:, 0]
            y_positions = locations[:, 1]
        else:
            raise ValueError(f"Invalid location array shape: {locations.shape}")
    
    # Fallback to config_df positions
    elif 'pos_x' in config_df.columns and 'pos_y' in config_df.columns:
        logger.debug("Using electrode positions from config dataframe")
        x_positions = config_df['pos_x'].values
        y_positions = config_df['pos_y'].values
    
    else:
        # Last resort: create dummy positions based on data shape
        logger.warning("No electrode positions found, creating dummy grid positions")
        
        # Assume we have data for the first frequency band to get number of channels
        first_band_key = list(waves.files)[0]
        if first_band_key != 'location':  # Skip location if it exists
            n_channels = waves[first_band_key].shape[0]
        else:
            # Try second key
            if len(waves.files) > 1:
                second_key = list(waves.files)[1]
                n_channels = waves[second_key].shape[0]
            else:
                raise ValueError("Cannot determine number of channels from data")
        
        # Create grid positions
        grid_size = int(np.ceil(np.sqrt(n_channels)))
        x_positions = np.tile(np.arange(grid_size), grid_size)[:n_channels]
        y_positions = np.repeat(np.arange(grid_size), grid_size)[:n_channels]
    
    # Validate positions
    if len(x_positions) != len(y_positions):
        raise ValueError(f"Position arrays have different lengths: x={len(x_positions)}, y={len(y_positions)}")
    
    logger.debug(f"Extracted {len(x_positions)} electrode positions")
    
    return x_positions, y_positions


def _add_basic_frequency_bands(processor: 'LFPDataProcessor', logger: logging.Logger) -> None:
    """
    Add basic frequency bands required for oscillation detection.
    
    Parameters:
    -----------
    processor : LFPDataProcessor
        LFP processor to add bands to
    logger : logging.Logger
        Logger instance
    """
    
    # Check which bands are already available in the waves data
    available_bands = list(processor.waves.files)
    logger.debug(f"Available frequency bands in data: {available_bands}")
    
    # Define basic bands that are commonly needed
    basic_bands = [
        {'name': 'lfp', 'low': 1, 'high': 300, 'description': 'Raw LFP for sharp wave detection'},
        {'name': 'sharpWave', 'low': 1, 'high': 30, 'description': 'Sharp wave band'}
    ]
    
    # Add bands if they don't already exist
    for band in basic_bands:
        if band['name'] not in available_bands:
            logger.debug(f"Adding frequency band: {band['name']} ({band['low']}-{band['high']} Hz)")
            
            try:
                # Use CPU for basic bands to avoid GPU memory issues
                processor.add_frequency_band(
                    band['low'], 
                    band['high'], 
                    band_name=band['name'],
                    use_gpu=False,
                    store_analytical=True
                )
            except Exception as e:
                logger.warning(f"Failed to add frequency band {band['name']}: {str(e)}")
        else:
            logger.debug(f"Frequency band {band['name']} already available")


def load_electrode_layout(config_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract electrode layout information from configuration.
    
    Parameters:
    -----------
    config_df : pd.DataFrame
        Configuration dataframe from Maxwell data
        
    Returns:
    --------
    dict
        Dictionary containing electrode layout information
    """
    
    layout_info = {
        'n_electrodes': len(config_df),
        'channels': config_df.get('channel', []).tolist(),
        'electrodes': config_df.get('electrode', []).tolist() if 'electrode' in config_df.columns else None
    }
    
    # Calculate spatial extent
    if 'pos_x' in config_df.columns and 'pos_y' in config_df.columns:
        x_positions = config_df['pos_x'].values
        y_positions = config_df['pos_y'].values
        
        layout_info.update({
            'x_range': [float(np.min(x_positions)), float(np.max(x_positions))],
            'y_range': [float(np.min(y_positions)), float(np.max(y_positions))],
            'spatial_extent': {
                'width': float(np.max(x_positions) - np.min(x_positions)),
                'height': float(np.max(y_positions) - np.min(y_positions))
            }
        })
    
    return layout_info


def verify_data_compatibility(waves: np.lib.npyio.NpzFile, 
                             config_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Verify that loaded data files are compatible with each other.
    
    Parameters:
    -----------
    waves : np.lib.npyio.NpzFile
        Loaded NPZ file containing LFP data
    config_df : pd.DataFrame
        Configuration dataframe from Maxwell data
        
    Returns:
    --------
    dict
        Dictionary of compatibility checks
    """
    
    logger = logging.getLogger('axolotl.data_loader')
    checks = {}
    
    try:
        # Check if number of channels match
        if waves.files:
            # Get first frequency band to check channel count
            first_band_key = list(waves.files)[0]
            if first_band_key != 'location':
                n_channels_waves = waves[first_band_key].shape[0]
            else:
                # Use second key if first is location
                if len(waves.files) > 1:
                    second_key = list(waves.files)[1]
                    n_channels_waves = waves[second_key].shape[0]
                else:
                    n_channels_waves = None
            
            n_channels_config = len(config_df)
            
            if n_channels_waves is not None:
                checks['channel_count_match'] = (n_channels_waves == n_channels_config)
                if not checks['channel_count_match']:
                    logger.warning(f"Channel count mismatch: waves={n_channels_waves}, config={n_channels_config}")
            else:
                checks['channel_count_match'] = None
        else:
            checks['channel_count_match'] = False
            logger.error("No frequency bands found in waves data")
        
        # Check if position data is available
        has_positions_npz = 'location' in waves.files
        has_positions_config = 'pos_x' in config_df.columns and 'pos_y' in config_df.columns
        checks['positions_available'] = has_positions_npz or has_positions_config
        
        if not checks['positions_available']:
            logger.warning("No electrode position data found")
        
        # Check for required columns in config
        required_config_columns = ['channel']
        checks['config_complete'] = all(col in config_df.columns for col in required_config_columns)
        
        if not checks['config_complete']:
            missing_cols = [col for col in required_config_columns if col not in config_df.columns]
            logger.warning(f"Missing required config columns: {missing_cols}")
        
        # Overall compatibility
        checks['overall_compatible'] = all([
            checks.get('channel_count_match', True),
            checks.get('positions_available', False),
            checks.get('config_complete', False)
        ])
        
    except Exception as e:
        logger.error(f"Error during compatibility check: {str(e)}")
        checks['overall_compatible'] = False
        checks['error'] = str(e)
    
    return checks


def get_data_summary(file_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Get summary information about the loaded data files.
    
    Parameters:
    -----------
    file_paths : dict
        Dictionary containing paths to data files
        
    Returns:
    --------
    dict
        Summary information about the data
    """
    
    logger = logging.getLogger('axolotl.data_loader')
    summary = {
        'file_paths': file_paths.copy(),
        'file_sizes': {},
        'data_info': {}
    }
    
    try:
        # Get file sizes
        for file_type, path in file_paths.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                summary['file_sizes'][file_type] = f"{size_mb:.2f} MB"
        
        # Get data-specific information
        if 'npz' in file_paths:
            waves = np.load(file_paths['npz'])
            summary['data_info']['frequency_bands'] = list(waves.files)
            
            # Get sample rate and duration from first band
            if waves.files:
                first_band_key = list(waves.files)[0]
                if first_band_key != 'location':
                    band_data = waves[first_band_key]
                    n_channels, n_samples = band_data.shape
                    summary['data_info']['n_channels'] = n_channels
                    summary['data_info']['n_samples'] = n_samples
                    
                    # Estimate duration (assuming 20kHz sampling rate)
                    duration_s = n_samples / 20000
                    summary['data_info']['estimated_duration'] = f"{duration_s:.2f} seconds"
        
        if 'h5' in file_paths:
            try:
                _, _, config_df, _ = load_info_maxwell(file_paths['h5'])
                summary['data_info']['n_electrodes_config'] = len(config_df)
            except Exception as e:
                logger.warning(f"Could not load Maxwell info: {e}")
        
    except Exception as e:
        logger.error(f"Error generating data summary: {str(e)}")
        summary['error'] = str(e)
    
    return summary