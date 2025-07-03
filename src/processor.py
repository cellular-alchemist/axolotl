"""
Axolotl Oscillation Processor
Handles oscillation detection using LFPDataProcessor.detect_ripples()
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime


class OscillationProcessor:
    """
    Core oscillation detection processor that wraps LFPDataProcessor.detect_ripples()
    and standardizes the output format for different oscillation types.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('axolotl.processor')
    
    def process_oscillations(self, lfp_processor, oscillation_params: Dict[str, Any], 
                           oscillation_type: str, sample_name: str, 
                           condition_name: str) -> Dict[str, Any]:
        """
        Process oscillations for a specific type using the configured parameters.
        
        Parameters:
        -----------
        lfp_processor : LFPDataProcessor
            Configured LFP processor with frequency bands added
        oscillation_params : dict
            Parameters for this oscillation type from config
        oscillation_type : str
            Name of the oscillation type (e.g., 'sharp_wave_ripples')
        sample_name : str
            Name of the sample being processed
        condition_name : str
            Name of the condition being processed
            
        Returns:
        --------
        dict
            Standardized results dictionary with detection results and metadata
        """
        
        self.logger.info(f"    Detecting {oscillation_type} oscillations")
        
        try:
            # Extract parameters for detect_ripples method
            detection_params = self._extract_detection_parameters(oscillation_params)
            
            # Log detection parameters
            self.logger.debug(f"    Detection parameters: {detection_params}")
            
            # Run oscillation detection
            ripples = lfp_processor.detect_ripples(**detection_params)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(ripples, lfp_processor)
            
            # Create standardized results dictionary
            results = {
                'ripples': ripples,
                'summary_stats': summary_stats,
                'metadata': {
                    'sample': sample_name,
                    'condition': condition_name,
                    'oscillation_type': oscillation_type,
                    'processing_date': datetime.now().isoformat(),
                    'parameters': oscillation_params.copy(),
                    'detection_parameters': detection_params
                }
            }
            
            self.logger.info(f"    Detected {summary_stats['total_detections']} {oscillation_type} events")
            
            return results
            
        except Exception as e:
            self.logger.error(f"    Failed to process {oscillation_type}: {str(e)}")
            raise
    
    def _extract_detection_parameters(self, oscillation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate parameters for the detect_ripples method.
        
        Parameters:
        -----------
        oscillation_params : dict
            Oscillation type parameters from configuration
            
        Returns:
        --------
        dict
            Parameters formatted for detect_ripples method
        """
        
        # Required parameters
        detection_params = {
            'narrowband_key': oscillation_params['narrowband_key'],
            'wideband_key': oscillation_params['wideband_key'],
            'low_threshold': oscillation_params['low_threshold'],
            'high_threshold': oscillation_params['high_threshold'],
            'min_duration': oscillation_params['min_duration'],
            'max_duration': oscillation_params['max_duration'],
        }
        
        # Optional parameters with defaults
        detection_params['require_sharp_wave'] = oscillation_params.get('require_sharp_wave', True)
        detection_params['sharp_wave_window'] = oscillation_params.get('sharp_wave_window', 50)
        detection_params['sharp_wave_threshold'] = oscillation_params.get('sharp_wave_threshold', 2.0)
        detection_params['min_interval'] = oscillation_params.get('min_interval', 30)
        detection_params['window_length'] = oscillation_params.get('window_length', 11)
        
        # LFP key for sharp wave detection
        detection_params['lfp_key'] = oscillation_params.get('lfp_key', 'lfp')
        
        # Time window restrictions (if specified in config)
        detection_params['time_start'] = oscillation_params.get('time_start', None)
        detection_params['time_end'] = oscillation_params.get('time_end', None)
        
        return detection_params
    
    def _calculate_summary_statistics(self, ripples: Dict, lfp_processor) -> Dict[str, float]:
        """
        Calculate summary statistics from detection results.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        lfp_processor : LFPDataProcessor
            LFP processor used for detection
            
        Returns:
        --------
        dict
            Summary statistics including rates, durations, and powers
        """
        
        if not ripples or 'metadata' not in ripples:
            return {
                'total_detections': 0,
                'mean_rate': 0.0,
                'mean_duration': 0.0,
                'mean_power': 0.0,
                'channels_with_detections': 0
            }
        
        # Get metadata
        metadata = ripples['metadata']
        fs = metadata['fs']
        
        # Calculate recording duration
        if 'time_start' in metadata and 'time_end' in metadata:
            recording_duration = metadata['time_end'] - metadata['time_start']
        else:
            # Fallback to sample-based calculation
            total_samples = metadata.get('sample_end', 0) - metadata.get('sample_start', 0)
            recording_duration = total_samples / fs
        
        # Collect all detections across channels
        all_detections = []
        channels_with_detections = 0
        
        for key, value in ripples.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                # This is a channel with detections
                channels_with_detections += 1
                all_detections.extend(value.to_dict('records'))
        
        total_detections = len(all_detections)
        
        if total_detections == 0:
            return {
                'total_detections': 0,
                'mean_rate': 0.0,
                'mean_duration': 0.0,
                'mean_power': 0.0,
                'channels_with_detections': 0
            }
        
        # Calculate statistics
        durations = [detection.get('duration', 0) for detection in all_detections]
        powers = [detection.get('peak_normalized_power', 0) for detection in all_detections]
        
        # Calculate mean rate per channel (events/second)
        if channels_with_detections > 0 and recording_duration > 0:
            mean_rate = total_detections / (channels_with_detections * recording_duration)
        else:
            mean_rate = 0.0
        
        summary_stats = {
            'total_detections': total_detections,
            'mean_rate': float(mean_rate),
            'mean_duration': float(np.mean(durations) * 1000) if durations else 0.0,  # Convert to ms
            'mean_power': float(np.mean(powers)) if powers else 0.0,
            'channels_with_detections': channels_with_detections,
            'recording_duration': float(recording_duration)
        }
        
        # Additional statistics
        if durations:
            summary_stats.update({
                'median_duration': float(np.median(durations) * 1000),  # ms
                'std_duration': float(np.std(durations) * 1000),  # ms
                'min_duration': float(np.min(durations) * 1000),  # ms
                'max_duration': float(np.max(durations) * 1000)   # ms
            })
        
        if powers:
            summary_stats.update({
                'median_power': float(np.median(powers)),
                'std_power': float(np.std(powers)),
                'min_power': float(np.min(powers)),
                'max_power': float(np.max(powers))
            })
        
        return summary_stats
    
    def get_detection_rate_per_channel(self, ripples: Dict) -> Dict[int, float]:
        """
        Calculate detection rate for each channel.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
            
        Returns:
        --------
        dict
            Mapping from channel number to detection rate (events/second)
        """
        
        if not ripples or 'metadata' not in ripples:
            return {}
        
        metadata = ripples['metadata']
        fs = metadata['fs']
        
        # Calculate recording duration
        if 'time_start' in metadata and 'time_end' in metadata:
            recording_duration = metadata['time_end'] - metadata['time_start']
        else:
            total_samples = metadata.get('sample_end', 0) - metadata.get('sample_start', 0)
            recording_duration = total_samples / fs
        
        if recording_duration <= 0:
            return {}
        
        channel_rates = {}
        
        for key, value in ripples.items():
            if isinstance(value, pd.DataFrame):
                try:
                    channel_num = int(key)
                    num_detections = len(value)
                    rate = num_detections / recording_duration
                    channel_rates[channel_num] = rate
                except (ValueError, TypeError):
                    # Skip non-numeric keys (like 'metadata')
                    continue
        
        return channel_rates
    
    def filter_detections_by_power(self, ripples: Dict, min_power: float) -> Dict:
        """
        Filter detections by minimum power threshold.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        min_power : float
            Minimum normalized power threshold
            
        Returns:
        --------
        dict
            Filtered ripples dictionary
        """
        
        filtered_ripples = {'metadata': ripples.get('metadata', {})}
        
        for key, value in ripples.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                # Filter by power threshold
                mask = value['peak_normalized_power'] >= min_power
                filtered_df = value[mask].copy()
                
                if not filtered_df.empty:
                    filtered_ripples[key] = filtered_df
        
        return filtered_ripples
    
    def get_oscillation_summary(self, all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create a summary across all oscillation types for a condition.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary mapping oscillation type names to results
            
        Returns:
        --------
        dict
            Summary statistics across all oscillation types
        """
        
        summary = {
            'oscillation_types': list(all_results.keys()),
            'total_detections_by_type': {},
            'mean_rates_by_type': {},
            'processing_date': datetime.now().isoformat()
        }
        
        for osc_type, results in all_results.items():
            if 'summary_stats' in results:
                stats = results['summary_stats']
                summary['total_detections_by_type'][osc_type] = stats.get('total_detections', 0)
                summary['mean_rates_by_type'][osc_type] = stats.get('mean_rate', 0.0)
        
        # Calculate overall statistics
        total_detections = sum(summary['total_detections_by_type'].values())
        mean_rate_overall = np.mean(list(summary['mean_rates_by_type'].values()))
        
        summary.update({
            'total_detections_all_types': total_detections,
            'mean_rate_overall': float(mean_rate_overall)
        })
        
        return summary