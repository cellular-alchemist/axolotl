"""
Axolotl Statistical Aggregator
Performs statistical analysis and comparison across experimental conditions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from collections import defaultdict

# Configure matplotlib for headless environment. # This is necessary for environments without a display (e.g., servers).
matplotlib.use('Agg')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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


class StatsAggregator:
    """
    Aggregates statistics across experimental conditions and performs 
    statistical comparisons for oscillation detection results.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('axolotl.aggregator')
    
    def aggregate_across_conditions(self, all_results: List[Dict[str, Any]], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate statistics across all samples and conditions.
        
        Parameters:
        -----------
        all_results : list
            List of processing results from all conditions
        config : dict
            Experiment configuration
            
        Returns:
        --------
        dict
            Aggregated statistics and analysis results
        """
        
        self.logger.info("Aggregating statistics across conditions")
        
        # Organize results by oscillation type and condition
        organized_data = self._organize_results(all_results)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(organized_data)
        
        # Perform statistical comparisons
        statistical_tests = self._perform_statistical_tests(organized_data, config)
        
        # Create aggregated dataset for plotting
        plotting_data = self._prepare_plotting_data(organized_data)
        
        aggregated_results = {
            'organized_data': organized_data,
            'summary_statistics': summary_stats,
            'statistical_tests': statistical_tests,
            'plotting_data': plotting_data,
            'oscillation_types': list(config['oscillation_types'].keys()),
            'conditions': self._extract_conditions(all_results),
            'samples': self._extract_samples(all_results)
        }
        
        self.logger.info(f"Aggregated data for {len(aggregated_results['oscillation_types'])} oscillation types "
                        f"across {len(aggregated_results['conditions'])} conditions")
        
        return aggregated_results
    
    def _organize_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, List]]:
        """Organize results by oscillation type and condition"""
        
        organized = defaultdict(lambda: defaultdict(list))
        
        for result in all_results:
            if result.get('status') != 'success':
                continue
                
            sample = result.get('sample', 'unknown')
            condition = result.get('condition', 'unknown')
            
            # Load the actual oscillation detection results
            # Note: In practice, these would be loaded from S3 or local files
            # For now, we'll extract from the result structure
            oscillation_results = result.get('oscillation_results', {})
            
            for osc_type, osc_data in oscillation_results.items():
                summary_stats = osc_data.get('summary_stats', {})
                
                # Handle case where summary_stats might be a numpy array
                if isinstance(summary_stats, np.ndarray):
                    summary_stats = {}
                
                organized[osc_type][condition].append({
                    'sample': sample,
                    'condition': condition,
                    'total_detections': summary_stats.get('total_detections', 0),
                    'mean_rate': summary_stats.get('mean_rate', 0.0),
                    'mean_duration': summary_stats.get('mean_duration', 0.0),
                    'mean_power': summary_stats.get('mean_power', 0.0),
                    'channels_with_detections': summary_stats.get('channels_with_detections', 0),
                    'recording_duration': summary_stats.get('recording_duration', 0.0)
                })
        
        return dict(organized)
    
    def _calculate_summary_statistics(self, organized_data: Dict[str, Dict[str, List]]) -> Dict[str, Any]:
        """Calculate summary statistics across conditions"""
        
        summary = {}
        
        for osc_type, condition_data in organized_data.items():
            summary[osc_type] = {}
            
            for condition, samples in condition_data.items():
                if not samples:
                    continue
                
                # Extract metrics
                rates = [s['mean_rate'] for s in samples]
                durations = [s['mean_duration'] for s in samples]
                powers = [s['mean_power'] for s in samples]
                detections = [s['total_detections'] for s in samples]
                
                summary[osc_type][condition] = {
                    'n_samples': len(samples),
                    'rate': {
                        'mean': np.mean(rates),
                        'std': np.std(rates),
                        'median': np.median(rates),
                        'sem': stats.sem(rates) if len(rates) > 1 else 0
                    },
                    'duration': {
                        'mean': np.mean(durations),
                        'std': np.std(durations),
                        'median': np.median(durations),
                        'sem': stats.sem(durations) if len(durations) > 1 else 0
                    },
                    'power': {
                        'mean': np.mean(powers),
                        'std': np.std(powers),
                        'median': np.median(powers),
                        'sem': stats.sem(powers) if len(powers) > 1 else 0
                    },
                    'total_detections': {
                        'mean': np.mean(detections),
                        'std': np.std(detections),
                        'median': np.median(detections),
                        'sum': np.sum(detections)
                    }
                }
        
        return summary
    
    def _perform_statistical_tests(self, organized_data: Dict[str, Dict[str, List]],
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical tests between conditions"""
        
        test_results = {}
        
        for osc_type, condition_data in organized_data.items():
            test_results[osc_type] = {}
            conditions = list(condition_data.keys())
            
            # Perform pairwise comparisons
            for i, cond1 in enumerate(conditions):
                for j, cond2 in enumerate(conditions[i+1:], i+1):
                    comparison_key = f"{cond1}_vs_{cond2}"
                    test_results[osc_type][comparison_key] = {}
                    
                    # Get data for both conditions
                    data1 = condition_data[cond1]
                    data2 = condition_data[cond2]
                    
                    if len(data1) == 0 or len(data2) == 0:
                        continue
                    
                    # Test each metric
                    metrics = ['mean_rate', 'mean_duration', 'mean_power', 'total_detections']
                    
                    for metric in metrics:
                        values1 = [s[metric] for s in data1]
                        values2 = [s[metric] for s in data2]
                        
                        # Mann-Whitney U test (non-parametric)
                        if len(values1) > 1 and len(values2) > 1:
                            try:
                                statistic, p_value = stats.mannwhitneyu(values1, values2, 
                                                                       alternative='two-sided')
                                test_results[osc_type][comparison_key][metric] = {
                                    'test': 'Mann-Whitney U',
                                    'statistic': float(statistic),
                                    'p_value': float(p_value),
                                    'significant': p_value < 0.05,
                                    'effect_size': self._calculate_effect_size(values1, values2)
                                }
                            except Exception as e:
                                self.logger.warning(f"Failed statistical test for {metric}: {e}")
                                test_results[osc_type][comparison_key][metric] = {
                                    'test': 'Mann-Whitney U',
                                    'error': str(e)
                                }
        
        return test_results
    
    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        try:
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            # Cohen's d
            cohens_d = (mean1 - mean2) / pooled_std
            return float(cohens_d)
        except:
            return 0.0
    
    def _prepare_plotting_data(self, organized_data: Dict[str, Dict[str, List]]) -> pd.DataFrame:
        """Prepare data in long format for plotting"""
        
        rows = []
        
        for osc_type, condition_data in organized_data.items():
            for condition, samples in condition_data.items():
                for sample_data in samples:
                    rows.append({
                        'oscillation_type': osc_type,
                        'condition': condition,
                        'sample': sample_data['sample'],
                        'detection_rate': sample_data['mean_rate'],
                        'mean_duration': sample_data['mean_duration'],
                        'mean_power': sample_data['mean_power'],
                        'total_detections': sample_data['total_detections'],
                        'channels_with_detections': sample_data['channels_with_detections']
                    })
        
        return pd.DataFrame(rows)
    
    def _extract_conditions(self, all_results: List[Dict[str, Any]]) -> List[str]:
        """Extract unique condition names"""
        conditions = set()
        for result in all_results:
            if result.get('status') == 'success':
                conditions.add(result.get('condition', 'unknown'))
        return sorted(list(conditions))
    
    def _extract_samples(self, all_results: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sample names"""
        samples = set()
        for result in all_results:
            if result.get('status') == 'success':
                samples.add(result.get('sample', 'unknown'))
        return sorted(list(samples))
    
    def create_violin_plots(self, aggregated_data: Dict[str, Any], 
                          output_dir: Path, save_formats: List[str] = ['png', 'svg']) -> None:
        """
        Create violin plots comparing conditions across oscillation types.
        
        Parameters:
        -----------
        aggregated_data : dict
            Aggregated statistics from aggregate_across_conditions
        output_dir : Path
            Output directory for plots
        save_formats : list
            File formats to save
        """
        
        self.logger.info("Creating violin plots for condition comparisons")
        
        plotting_data = aggregated_data['plotting_data']
        if plotting_data.empty:
            self.logger.warning("No data available for violin plots")
            return
        
        # Create subplot directory
        plot_dir = output_dir / "aggregate_statistics"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics to plot
        metrics_info = [
            ('detection_rate', 'Detection Rate (events/s)', 'Detection Rate'),
            ('mean_duration', 'Mean Duration (ms)', 'Duration'),
            ('mean_power', 'Mean Power (z-score)', 'Power'),
            ('total_detections', 'Total Detections', 'Detections')
        ]
        
        for metric, ylabel, title in metrics_info:
            try:
                # Create figure with subplots for each oscillation type
                oscillation_types = plotting_data['oscillation_type'].unique()
                n_types = len(oscillation_types)
                
                if n_types == 0:
                    continue
                
                # Determine subplot layout
                n_cols = min(3, n_types)
                n_rows = (n_types + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_types == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes
                else:
                    axes = axes.flatten()
                
                fig.suptitle(f'{title} Comparison Across Conditions', fontsize=16)
                
                for idx, osc_type in enumerate(oscillation_types):
                    ax = axes[idx] if n_types > 1 else axes[0]
                    
                    # Filter data for this oscillation type
                    osc_data = plotting_data[plotting_data['oscillation_type'] == osc_type]
                    
                    if not osc_data.empty and metric in osc_data.columns:
                        # Create violin plot
                        sns.violinplot(data=osc_data, x='condition', y=metric, ax=ax)
                        
                        # Add individual points
                        sns.stripplot(data=osc_data, x='condition', y=metric, 
                                    color='black', alpha=0.6, size=3, ax=ax)
                        
                        ax.set_title(f'{osc_type.replace("_", " ").title()}')
                        ax.set_xlabel('Condition')
                        ax.set_ylabel(ylabel)
                        
                        # Rotate x-axis labels if needed
                        ax.tick_params(axis='x', rotation=45)
                        
                        # Add statistical annotations if significant differences exist
                        self._add_statistical_annotations(ax, aggregated_data, osc_type, metric)
                
                # Remove empty subplots
                for idx in range(n_types, len(axes)):
                    if n_types > 1:
                        fig.delaxes(axes[idx])
                
                plt.tight_layout()
                
                # Save the plot
                for fmt in save_formats:
                    filename = plot_dir / f"violin_plot_{metric}.{fmt}"
                    fig.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
                
                plt.close(fig)
                
            except Exception as e:
                self.logger.error(f"Failed to create violin plot for {metric}: {str(e)}")
                continue
    
    def _add_statistical_annotations(self, ax, aggregated_data: Dict[str, Any], 
                                   osc_type: str, metric: str) -> None:
        """Add statistical significance annotations to plots"""
        
        try:
            test_results = aggregated_data.get('statistical_tests', {}).get(osc_type, {})
            
            # Find significant comparisons
            y_max = ax.get_ylim()[1]
            y_offset = y_max * 0.02
            
            annotation_y = y_max + y_offset
            
            for comparison, results in test_results.items():
                if metric in results and results[metric].get('significant', False):
                    p_value = results[metric]['p_value']
                    
                    # Extract condition names from comparison
                    cond1, cond2 = comparison.split('_vs_')
                    
                    # Add significance annotation
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        continue
                    
                    # Simple annotation (could be improved with bracket positioning)
                    ax.text(0.5, 0.95, f'{cond1} vs {cond2}: {sig_text}', 
                           transform=ax.transAxes, ha='center', va='top',
                           fontsize=8, style='italic')
                    break  # Only show one annotation to avoid clutter
                    
        except Exception as e:
            self.logger.debug(f"Could not add statistical annotations: {e}")
    
    def export_summary_statistics(self, aggregated_data: Dict[str, Any], 
                                output_dir: Path) -> None:
        """
        Export summary statistics to CSV files.
        
        Parameters:
        -----------
        aggregated_data : dict
            Aggregated statistics
        output_dir : Path
            Output directory
        """
        
        self.logger.info("Exporting summary statistics to CSV")
        
        stats_dir = output_dir / "aggregate_statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export summary statistics
            summary_rows = []
            summary_stats = aggregated_data['summary_statistics']
            
            for osc_type, condition_data in summary_stats.items():
                for condition, stats in condition_data.items():
                    row = {
                        'oscillation_type': osc_type,
                        'condition': condition,
                        'n_samples': stats['n_samples'],
                        'rate_mean': stats['rate']['mean'],
                        'rate_std': stats['rate']['std'],
                        'rate_sem': stats['rate']['sem'],
                        'duration_mean': stats['duration']['mean'],
                        'duration_std': stats['duration']['std'],
                        'duration_sem': stats['duration']['sem'],
                        'power_mean': stats['power']['mean'],
                        'power_std': stats['power']['std'],
                        'power_sem': stats['power']['sem'],
                        'total_detections_mean': stats['total_detections']['mean'],
                        'total_detections_sum': stats['total_detections']['sum']
                    }
                    summary_rows.append(row)
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(stats_dir / "summary_statistics.csv", index=False)
            
            # Export statistical test results
            test_rows = []
            test_results = aggregated_data['statistical_tests']
            
            for osc_type, comparisons in test_results.items():
                for comparison, metrics in comparisons.items():
                    for metric, test_result in metrics.items():
                        if isinstance(test_result, dict) and 'p_value' in test_result:
                            row = {
                                'oscillation_type': osc_type,
                                'comparison': comparison,
                                'metric': metric,
                                'test_type': test_result.get('test', 'unknown'),
                                'statistic': test_result.get('statistic', None),
                                'p_value': test_result.get('p_value', None),
                                'significant': test_result.get('significant', False),
                                'effect_size': test_result.get('effect_size', None)
                            }
                            test_rows.append(row)
            
            if test_rows:
                test_df = pd.DataFrame(test_rows)
                test_df.to_csv(stats_dir / "statistical_tests.csv", index=False)
            
            # Export raw plotting data
            plotting_data = aggregated_data['plotting_data']
            if not plotting_data.empty:
                plotting_data.to_csv(stats_dir / "raw_data.csv", index=False)
            
            self.logger.info(f"Exported statistics to {stats_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to export statistics: {str(e)}")
    
    def create_summary_report(self, aggregated_data: Dict[str, Any], 
                            output_dir: Path, config: Dict[str, Any]) -> None:
        """
        Create a comprehensive summary report.
        
        Parameters:
        -----------
        aggregated_data : dict
            Aggregated statistics
        output_dir : Path
            Output directory
        config : dict
            Experiment configuration
        """
        
        self.logger.info("Creating summary report")
        
        try:
            report_path = output_dir / "aggregate_statistics" / "summary_report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write("AXOLOTL OSCILLATION DETECTION SUMMARY REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                # Experiment information
                f.write(f"Experiment: {config['experiment']['name']}\n")
                f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Oscillation Types: {', '.join(aggregated_data['oscillation_types'])}\n")
                f.write(f"Conditions: {', '.join(aggregated_data['conditions'])}\n")
                f.write(f"Samples: {', '.join(aggregated_data['samples'])}\n\n")
                
                # Summary statistics
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 30 + "\n\n")
                
                summary_stats = aggregated_data['summary_statistics']
                for osc_type, condition_data in summary_stats.items():
                    f.write(f"{osc_type.replace('_', ' ').title()}:\n")
                    for condition, stats in condition_data.items():
                        f.write(f"  {condition}:\n")
                        f.write(f"    Samples: {stats['n_samples']}\n")
                        f.write(f"    Detection Rate: {stats['rate']['mean']:.3f} ± {stats['rate']['sem']:.3f} events/s\n")
                        f.write(f"    Duration: {stats['duration']['mean']:.1f} ± {stats['duration']['sem']:.1f} ms\n")
                        f.write(f"    Power: {stats['power']['mean']:.2f} ± {stats['power']['sem']:.2f} z-score\n")
                        f.write(f"    Total Detections: {stats['total_detections']['sum']}\n\n")
                
                # Statistical comparisons
                f.write("STATISTICAL COMPARISONS\n")
                f.write("-" * 30 + "\n\n")
                
                test_results = aggregated_data['statistical_tests']
                for osc_type, comparisons in test_results.items():
                    f.write(f"{osc_type.replace('_', ' ').title()}:\n")
                    for comparison, metrics in comparisons.items():
                        f.write(f"  {comparison.replace('_vs_', ' vs ')}:\n")
                        for metric, test_result in metrics.items():
                            if isinstance(test_result, dict) and 'p_value' in test_result:
                                sig_marker = "*" if test_result['significant'] else ""
                                f.write(f"    {metric}: p = {test_result['p_value']:.4f}{sig_marker}, "
                                       f"effect size = {test_result.get('effect_size', 0):.3f}\n")
                        f.write("\n")
                
                f.write("* p < 0.05 (statistically significant)\n")
                f.write("\nEnd of Report\n")
            
            self.logger.info(f"Summary report created: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create summary report: {str(e)}")