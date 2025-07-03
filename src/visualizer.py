"""
Axolotl Oscillation Visualizer
Creates comprehensive visualizations for oscillation detection results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure matplotlib for headless environment
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8')

# Configure matplotlib parameters for publication-quality figures
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})


class OscillationVisualizer:
    """
    Creates comprehensive visualizations for oscillation detection results.
    Organizes outputs in a structured folder hierarchy.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('axolotl.visualizer')
    
    def visualize_all_oscillations(self, results_dict: Dict[str, Dict], 
                                 output_dir: Path, config: Dict[str, Any],
                                 lfp_processor, sample_name: str, 
                                 condition_name: str) -> None:
        """
        Generate visualizations for all oscillation types in a condition.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary mapping oscillation type names to detection results
        output_dir : Path
            Base output directory
        config : dict
            Experiment configuration
        lfp_processor : LFPDataProcessor
            LFP processor used for detection
        sample_name : str
            Name of the sample
        condition_name : str
            Name of the condition
        """
        
        self.logger.info(f"Creating visualizations for {sample_name}/{condition_name}")
        
        # Get visualization configuration
        viz_config = config.get('visualization', {})
        channels_per_plot = viz_config.get('channels_per_plot', 5)
        ripples_per_channel = viz_config.get('ripples_per_channel', 10)
        time_window = viz_config.get('time_window', 0.3)
        save_formats = viz_config.get('save_format', ['png'])
        
        # Create base directory structure
        base_dir = output_dir / sample_name / condition_name
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each oscillation type
        for osc_type, results in results_dict.items():
            self.logger.info(f"  Creating visualizations for {osc_type}")
            
            osc_dir = base_dir / osc_type
            osc_dir.mkdir(exist_ok=True)
            
            try:
                # Extract ripples data
                ripples = results.get('ripples', {})
                if not ripples:
                    self.logger.warning(f"    No ripples data found for {osc_type}")
                    continue
                
                # Create channel plots
                self._create_channel_plots(
                    ripples, osc_dir, lfp_processor, osc_type,
                    channels_per_plot, ripples_per_channel, time_window, save_formats
                )
                
                # Create spatial heatmap
                self._create_spatial_heatmap(
                    ripples, osc_dir, lfp_processor, osc_type, save_formats
                )
                
                # Create summary plots
                self._create_summary_plots(
                    ripples, results, osc_dir, osc_type, save_formats
                )
                
                # Create detection statistics plot
                self._create_detection_stats_plot(
                    results, osc_dir, osc_type, save_formats
                )
                
            except Exception as e:
                self.logger.error(f"    Error creating visualizations for {osc_type}: {str(e)}")
                continue
        
        # Create condition summary
        self._create_condition_summary(
            results_dict, base_dir, sample_name, condition_name, save_formats
        )
    
    def _create_channel_plots(self, ripples: Dict, output_dir: Path,
                            lfp_processor, osc_type: str, channels_per_plot: int,
                            ripples_per_channel: int, time_window: float,
                            save_formats: List[str]) -> None:
        """Create individual channel visualization plots"""
        
        channel_dir = output_dir / "channel_plots"
        channel_dir.mkdir(exist_ok=True)
        
        # Get channels with detections
        channels_with_ripples = []
        for key, value in ripples.items():
            if isinstance(key, (int, np.integer)) and isinstance(value, pd.DataFrame) and not value.empty:
                channels_with_ripples.append(key)
        
        if not channels_with_ripples:
            self.logger.warning(f"    No channels with detections for {osc_type}")
            return
        
        # Sort channels by detection count
        channel_counts = [(ch, len(ripples[ch])) for ch in channels_with_ripples]
        channel_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Create plots for top channels
        n_plots = min(channels_per_plot, len(channel_counts))
        
        for i, (channel, count) in enumerate(channel_counts[:n_plots]):
            try:
                # Create visualization using LFPDataProcessor method
                fig = lfp_processor.visualize_ripples(
                    ripples,
                    channel=channel,
                    n_ripples=ripples_per_channel,
                    window=time_window,
                    figsize=(15, 10)
                )
                
                # Add title with detection count
                fig.suptitle(f'{osc_type.replace("_", " ").title()} - Channel {channel} ({count} detections)', 
                           fontsize=16, y=0.95)
                
                # Save in requested formats
                for fmt in save_formats:
                    filename = channel_dir / f"channel_{channel:03d}.{fmt}"
                    fig.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
                
                plt.close(fig)
                
            except Exception as e:
                self.logger.warning(f"    Failed to create plot for channel {channel}: {str(e)}")
                continue
    
    def _create_spatial_heatmap(self, ripples: Dict, output_dir: Path,
                              lfp_processor, osc_type: str, 
                              save_formats: List[str]) -> None:
        """Create spatial heatmap of detection rates"""
        
        try:
            # Create heatmap using LFPDataProcessor method
            fig = lfp_processor.plot_ripple_rate_heatmap(
                ripples,
                figsize=(10, 8),
                cmap='viridis',
                show_labels=True,
                show_stats=True
            )
            
            # Update title
            fig.suptitle(f'{osc_type.replace("_", " ").title()} - Detection Rate Heatmap', 
                        fontsize=16, y=0.95)
            
            # Save in requested formats
            for fmt in save_formats:
                filename = output_dir / f"heatmap.{fmt}"
                fig.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"    Failed to create spatial heatmap: {str(e)}")
    
    def _create_summary_plots(self, ripples: Dict, results: Dict, output_dir: Path,
                            osc_type: str, save_formats: List[str]) -> None:
        """Create summary statistics plots"""
        
        try:
            # Extract detection data for summary
            all_detections = []
            channel_rates = []
            
            metadata = ripples.get('metadata', {})
            recording_duration = metadata.get('time_end', 1) - metadata.get('time_start', 0)
            
            for key, value in ripples.items():
                if isinstance(key, (int, np.integer)) and isinstance(value, pd.DataFrame) and not value.empty:
                    # Collect detection data
                    for _, detection in value.iterrows():
                        all_detections.append({
                            'channel': key,
                            'duration': detection.get('duration', 0) * 1000,  # Convert to ms
                            'power': detection.get('peak_normalized_power', 0),
                            'peak_time': detection.get('peak_time', 0)
                        })
                    
                    # Calculate rate for this channel
                    rate = len(value) / recording_duration if recording_duration > 0 else 0
                    channel_rates.append({'channel': key, 'rate': rate})
            
            if not all_detections:
                self.logger.warning(f"    No detections found for summary plots")
                return
            
            # Create summary figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{osc_type.replace("_", " ").title()} - Summary Statistics', fontsize=16)
            
            # 1. Duration histogram
            durations = [d['duration'] for d in all_detections]
            axes[0, 0].hist(durations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Duration (ms)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title(f'Duration Distribution (n={len(durations)})')
            axes[0, 0].axvline(np.mean(durations), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(durations):.1f} ms')
            axes[0, 0].legend()
            
            # 2. Power histogram
            powers = [d['power'] for d in all_detections]
            axes[0, 1].hist(powers, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_xlabel('Peak Power (z-score)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Power Distribution')
            axes[0, 1].axvline(np.mean(powers), color='red', linestyle='--',
                              label=f'Mean: {np.mean(powers):.1f}')
            axes[0, 1].legend()
            
            # 3. Detection rate by channel
            if channel_rates:
                rates = [c['rate'] for c in channel_rates]
                channels = [c['channel'] for c in channel_rates]
                
                # Show top 20 channels or all if fewer
                n_show = min(20, len(channel_rates))
                sorted_indices = np.argsort(rates)[-n_show:]
                
                axes[1, 0].barh([channels[i] for i in sorted_indices], 
                               [rates[i] for i in sorted_indices],
                               alpha=0.7, color='lightgreen')
                axes[1, 0].set_xlabel('Detection Rate (events/s)')
                axes[1, 0].set_ylabel('Channel')
                axes[1, 0].set_title(f'Top {n_show} Channels by Rate')
            
            # 4. Time course of detections
            peak_times = [d['peak_time'] for d in all_detections]
            if peak_times:
                # Create time bins
                time_bins = np.linspace(min(peak_times), max(peak_times), 50)
                hist, bin_edges = np.histogram(peak_times, bins=time_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                axes[1, 1].plot(bin_centers, hist, 'o-', alpha=0.7, color='purple')
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Detections per bin')
                axes[1, 1].set_title('Detection Time Course')
            
            plt.tight_layout()
            
            # Save in requested formats
            for fmt in save_formats:
                filename = output_dir / f"summary.{fmt}"
                fig.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"    Failed to create summary plots: {str(e)}")
    
    def _create_detection_stats_plot(self, results: Dict, output_dir: Path,
                                   osc_type: str, save_formats: List[str]) -> None:
        """Create detection statistics visualization"""
        
        try:
            summary_stats = results.get('summary_stats', {})
            if not summary_stats:
                return
            
            # Create statistics plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Prepare data for display
            stats_to_show = [
                ('Total Detections', summary_stats.get('total_detections', 0)),
                ('Mean Rate (Hz)', summary_stats.get('mean_rate', 0)),
                ('Mean Duration (ms)', summary_stats.get('mean_duration', 0)),
                ('Mean Power (z-score)', summary_stats.get('mean_power', 0)),
                ('Channels with Detections', summary_stats.get('channels_with_detections', 0)),
                ('Recording Duration (s)', summary_stats.get('recording_duration', 0))
            ]
            
            # Create text display
            y_pos = 0.9
            for label, value in stats_to_show:
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        text = f"{label}: {value:.2f}"
                    else:
                        text = f"{label}: {value}"
                else:
                    text = f"{label}: {value}"
                
                ax.text(0.1, y_pos, text, transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                y_pos -= 0.12
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(f'{osc_type.replace("_", " ").title()} - Detection Statistics', 
                        fontsize=14, fontweight='bold')
            
            # Save in requested formats
            for fmt in save_formats:
                filename = output_dir / f"stats.{fmt}"
                fig.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"    Failed to create detection stats plot: {str(e)}")
    
    def _create_condition_summary(self, results_dict: Dict[str, Dict], output_dir: Path,
                                sample_name: str, condition_name: str, 
                                save_formats: List[str]) -> None:
        """Create summary comparison across oscillation types for a condition"""
        
        try:
            # Prepare data for comparison
            osc_types = list(results_dict.keys())
            metrics = {
                'total_detections': [],
                'mean_rate': [],
                'mean_duration': [],
                'mean_power': []
            }
            
            for osc_type in osc_types:
                summary_stats = results_dict[osc_type].get('summary_stats', {})
                metrics['total_detections'].append(summary_stats.get('total_detections', 0))
                metrics['mean_rate'].append(summary_stats.get('mean_rate', 0))
                metrics['mean_duration'].append(summary_stats.get('mean_duration', 0))
                metrics['mean_power'].append(summary_stats.get('mean_power', 0))
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{sample_name} - {condition_name} - Oscillation Type Comparison', fontsize=16)
            
            # Plot each metric
            metric_info = [
                ('total_detections', 'Total Detections', 'Count'),
                ('mean_rate', 'Mean Detection Rate', 'Events/s'),
                ('mean_duration', 'Mean Duration', 'ms'),
                ('mean_power', 'Mean Peak Power', 'z-score')
            ]
            
            for idx, (metric_key, title, ylabel) in enumerate(metric_info):
                ax = axes[idx // 2, idx % 2]
                values = metrics[metric_key]
                
                bars = ax.bar(range(len(osc_types)), values, alpha=0.7, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(osc_types))))
                ax.set_xlabel('Oscillation Type')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_xticks(range(len(osc_types)))
                ax.set_xticklabels([osc.replace('_', ' ').title() for osc in osc_types], 
                                  rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    if value > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                               f'{value:.1f}' if isinstance(value, float) else str(value),
                               ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save in requested formats
            for fmt in save_formats:
                filename = output_dir / f"condition_summary.{fmt}"
                fig.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"    Failed to create condition summary: {str(e)}")
    
    def create_visualization_zip(self, output_dir: Path) -> Optional[Path]:
        """
        Create a ZIP archive of all visualization files.
        
        Parameters:
        -----------
        output_dir : Path
            Directory containing visualization files
            
        Returns:
        --------
        Path or None
            Path to created ZIP file or None if failed
        """
        
        try:
            zip_path = output_dir / "visualizations.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through all files in the output directory
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.endswith(('.png', '.svg', '.pdf')):
                            file_path = Path(root) / file
                            # Create relative path for ZIP
                            rel_path = file_path.relative_to(output_dir)
                            zipf.write(file_path, rel_path)
            
            self.logger.info(f"Created visualization archive: {zip_path}")
            return zip_path
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization ZIP: {str(e)}")
            return None
    
    def create_publication_figure(self, results_dict: Dict[str, Dict], output_dir: Path,
                                sample_name: str, condition_name: str,
                                save_formats: List[str] = ['png', 'svg']) -> None:
        """
        Create a publication-ready figure combining key visualizations.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary of oscillation detection results
        output_dir : Path
            Output directory
        sample_name : str
            Sample name
        condition_name : str
            Condition name
        save_formats : list
            File formats to save
        """
        
        try:
            # Create a comprehensive figure
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Title for the entire figure
            fig.suptitle(f'{sample_name} - {condition_name} - Oscillation Detection Results', 
                        fontsize=18, fontweight='bold')
            
            # Summary statistics across oscillation types
            ax1 = fig.add_subplot(gs[0, :2])
            osc_types = list(results_dict.keys())
            detection_counts = [results_dict[osc]['summary_stats'].get('total_detections', 0) 
                              for osc in osc_types]
            
            bars = ax1.bar(range(len(osc_types)), detection_counts, alpha=0.7,
                          color=plt.cm.Set2(np.linspace(0, 1, len(osc_types))))
            ax1.set_xlabel('Oscillation Type')
            ax1.set_ylabel('Total Detections')
            ax1.set_title('Detection Counts by Oscillation Type')
            ax1.set_xticks(range(len(osc_types)))
            ax1.set_xticklabels([osc.replace('_', ' ').title() for osc in osc_types], 
                               rotation=45, ha='right')
            
            # Add value labels
            for bar, count in zip(bars, detection_counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(detection_counts)*0.01,
                        str(count), ha='center', va='bottom')
            
            # Detection rates
            ax2 = fig.add_subplot(gs[0, 2:])
            rates = [results_dict[osc]['summary_stats'].get('mean_rate', 0) for osc in osc_types]
            ax2.bar(range(len(osc_types)), rates, alpha=0.7,
                   color=plt.cm.Set2(np.linspace(0, 1, len(osc_types))))
            ax2.set_xlabel('Oscillation Type')
            ax2.set_ylabel('Mean Rate (events/s)')
            ax2.set_title('Detection Rates by Oscillation Type')
            ax2.set_xticks(range(len(osc_types)))
            ax2.set_xticklabels([osc.replace('_', ' ').title() for osc in osc_types], 
                               rotation=45, ha='right')
            
            # Individual oscillation type details (use first 2 types with most detections)
            sorted_types = sorted(results_dict.items(), 
                                key=lambda x: x[1]['summary_stats'].get('total_detections', 0), 
                                reverse=True)
            
            for idx, (osc_type, results) in enumerate(sorted_types[:2]):
                # Duration distribution
                ripples = results.get('ripples', {})
                all_durations = []
                for key, value in ripples.items():
                    if isinstance(key, (int, np.integer)) and isinstance(value, pd.DataFrame) and not value.empty:
                        durations = value.get('duration', pd.Series()).values * 1000  # Convert to ms
                        all_durations.extend(durations)
                
                if all_durations:
                    ax = fig.add_subplot(gs[1 + idx, :2])
                    ax.hist(all_durations, bins=20, alpha=0.7, color=plt.cm.Set2(idx))
                    ax.set_xlabel('Duration (ms)')
                    ax.set_ylabel('Count')
                    ax.set_title(f'{osc_type.replace("_", " ").title()} - Duration Distribution')
                    ax.axvline(np.mean(all_durations), color='red', linestyle='--',
                              label=f'Mean: {np.mean(all_durations):.1f} ms')
                    ax.legend()
                
                # Power distribution
                all_powers = []
                for key, value in ripples.items():
                    if isinstance(key, (int, np.integer)) and isinstance(value, pd.DataFrame) and not value.empty:
                        powers = value.get('peak_normalized_power', pd.Series()).values
                        all_powers.extend(powers)
                
                if all_powers:
                    ax = fig.add_subplot(gs[1 + idx, 2:])
                    ax.hist(all_powers, bins=20, alpha=0.7, color=plt.cm.Set2(idx))
                    ax.set_xlabel('Peak Power (z-score)')
                    ax.set_ylabel('Count')
                    ax.set_title(f'{osc_type.replace("_", " ").title()} - Power Distribution')
                    ax.axvline(np.mean(all_powers), color='red', linestyle='--',
                              label=f'Mean: {np.mean(all_powers):.1f}')
                    ax.legend()
            
            # Save the publication figure
            for fmt in save_formats:
                filename = output_dir / f"publication_figure.{fmt}"
                fig.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
            
            plt.close(fig)
            self.logger.info(f"Created publication figure: {output_dir}/publication_figure.*")
            
        except Exception as e:
            self.logger.error(f"Failed to create publication figure: {str(e)}")


def save_figure_both_formats(fig, base_path: str, formats: List[str] = ['png', 'svg']) -> None:
    """
    Save a matplotlib figure in multiple formats.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    base_path : str
        Base path without extension
    formats : list
        List of formats to save
    """
    
    for fmt in formats:
        filename = f"{base_path}.{fmt}"
        fig.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
        plt.close(fig)