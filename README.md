# Axolotl: Neural Oscillation Detection Pipeline

![Axolotl Pipeline](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![Docker](https://img.shields.io/badge/docker-supported-blue)

**Axolotl** is a minimal, Docker-based pipeline for detecting multiple types of neural oscillations in hippocampal recordings. The system is designed for cloud deployment with Kubernetes and GitLab CI/CD, featuring automated caching, comprehensive visualizations, and statistical analysis.

## Features

- **Multi-type Oscillation Detection**: Sharp wave ripples, fast ripples, gamma bursts
- **Cloud-First Design**: S3 storage, Kubernetes deployment, Docker containerization
- **Intelligent Caching**: NPZ-based result caching prevents redundant computation
- **Comprehensive Visualizations**: Individual channel plots, spatial heatmaps, summary statistics
- **Statistical Analysis**: Cross-condition comparisons with Mann-Whitney U tests
- **Flexible Configuration**: YAML-based experimental setup
- **Local Testing**: Run pipeline locally without cloud dependencies

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Cloud Deployment](#cloud-deployment)
  - [Local Testing](#local-testing)
- [Output Structure](#output-structure)
- [Parameter Tuning](#parameter-tuning)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [Citation](#citation)

## Installation

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Kubernetes cluster (for cloud deployment)
- AWS credentials (for S3 access)

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd axolotl

# Install dependencies
pip install -r docker/requirements.txt

# Add source directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Docker Installation

```bash
# Build the Docker image
docker build -f docker/Dockerfile -t axolotl:latest .

# Or pull from registry (if available)
docker pull <registry>/axolotl:latest
```

## Quick Start

### Local Testing

```bash
# Generate example configuration
python local_runner.py --generate-config --output ./test_output/

# Run with local files
python local_runner.py \
    --h5 /path/to/recording.h5 \
    --zip /path/to/spikes.zip \
    --npz /path/to/lfp.npz \
    --output ./results/ \
    --oscillation-types sharp_wave_ripples fast_ripples

# Run with custom configuration
python local_runner.py --config experiment_config.yaml --output ./results/
```

### Cloud Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/job.yaml

# Check job status
kubectl get jobs -n braingeneers

# View logs
kubectl logs job/axolotl-<commit-sha> -n braingeneers
```

## Configuration

The pipeline uses YAML configuration files to define experiments. Here's the structure:

### Basic Configuration

```yaml
experiment:
  name: "hippocampal_oscillations_study"
  s3_output_base: "s3://braingeneers/oscillations/results/"
  description: "Multi-type oscillation detection study"

oscillation_types:
  sharp_wave_ripples:
    narrowband_key: "narrowRipples"    # 150-250 Hz filtered signal
    wideband_key: "broadRipples"       # 80-250 Hz filtered signal  
    low_threshold: 3.5                 # Starting/ending threshold (std dev)
    high_threshold: 5.0                # Peak threshold (std dev)
    min_duration: 20                   # Minimum duration (ms)
    max_duration: 200                  # Maximum duration (ms)
    require_sharp_wave: true           # Validate with sharp wave
    sharp_wave_window: 50              # Sharp wave search window (±ms)

processing:
  frequency_bands:
    - name: "narrowRipples"
      low: 150
      high: 250
    - name: "broadRipples" 
      low: 80
      high: 250

visualization:
  channels_per_plot: 5
  ripples_per_channel: 10
  time_window: 0.3
  save_format: ["png", "svg"]

samples:
  - name: "culture_001"
    conditions:
      baseline:
        files:
          h5: "s3://path/to/baseline.h5"
          zip: "s3://path/to/baseline.zip"
          npz: "s3://path/to/baseline.npz"
      drug_treatment:
        files:
          h5: "s3://path/to/treatment.h5"
          zip: "s3://path/to/treatment.zip" 
          npz: "s3://path/to/treatment.npz"
```

### Oscillation Types

#### Sharp Wave Ripples
High-frequency oscillations (150-250 Hz) coincident with sharp waves in hippocampal LFP.

```yaml
sharp_wave_ripples:
  narrowband_key: "narrowRipples"
  wideband_key: "broadRipples"
  low_threshold: 3.5
  high_threshold: 5.0
  min_duration: 20
  max_duration: 200
  require_sharp_wave: true
  sharp_wave_window: 50
```

#### Fast Ripples
Ultra-high frequency oscillations (250-500 Hz) associated with pathological states.

```yaml
fast_ripples:
  narrowband_key: "fastRipples"
  wideband_key: "ultraFastRipples"
  low_threshold: 3.0
  high_threshold: 4.5
  min_duration: 10
  max_duration: 100
  require_sharp_wave: false
```

#### Gamma Bursts
Lower frequency oscillations (30-100 Hz) related to cognitive processing.

```yaml
gamma_bursts:
  narrowband_key: "gammaEnvelope"
  wideband_key: "gammaBand"
  low_threshold: 2.5
  high_threshold: 4.0
  min_duration: 50
  max_duration: 500
  require_sharp_wave: false
```

## Usage

### Cloud Deployment

The pipeline is designed for Kubernetes deployment with GitLab CI/CD:

1. **Push to Repository**: Commits to `main`/`master` trigger automatic builds
2. **Container Build**: Kaniko builds and pushes Docker images
3. **Manifest Generation**: Kubernetes job manifests are auto-generated
4. **Job Deployment**: Apply the generated manifest to run analysis

```bash
# Manual deployment
git push origin main

# Or deploy specific commit
kubectl apply -f k8s/axolotl-job.yaml
```

### Local Testing

For development and testing without cloud infrastructure:

```bash
# Basic usage
python local_runner.py \
    --h5 data/recording.h5 \
    --zip data/spikes.zip \
    --npz data/lfp.npz \
    --output results/

# Custom sample/condition names
python local_runner.py \
    --h5 data/recording.h5 \
    --zip data/spikes.zip \
    --npz data/lfp.npz \
    --output results/ \
    --sample-name culture_001 \
    --condition-name baseline

# Specific oscillation types
python local_runner.py \
    --h5 data/recording.h5 \
    --zip data/spikes.zip \
    --npz data/lfp.npz \
    --output results/ \
    --oscillation-types sharp_wave_ripples gamma_bursts

# Using configuration file
python local_runner.py \
    --config my_experiment.yaml \
    --output results/
```

## Output Structure

The pipeline generates organized outputs in both cloud (S3) and local formats:

### Cloud (S3) Structure
```
s3://braingeneers/oscillations/results/experiment_name/
├── oscillation_data/
│   ├── culture_001/
│   │   ├── baseline/
│   │   │   ├── sharp_wave_ripples.npz
│   │   │   ├── fast_ripples.npz
│   │   │   └── gamma_bursts.npz
│   │   └── drug_treatment/
│   │       ├── sharp_wave_ripples.npz
│   │       ├── fast_ripples.npz
│   │       └── gamma_bursts.npz
├── visualizations/
│   ├── culture_001/
│   │   ├── baseline/
│   │   │   ├── sharp_wave_ripples/
│   │   │   │   ├── channel_plots/
│   │   │   │   ├── heatmap.png
│   │   │   │   └── summary.png
│   │   │   └── fast_ripples/...
│   │   └── drug_treatment/...
├── aggregate_statistics/
│   ├── violin_plots_detection_rate.png
│   ├── violin_plots_mean_duration.png
│   ├── summary_statistics.csv
│   └── statistical_tests.csv
└── processing_summary.json
```

### Local Structure
```
results/
├── sharp_wave_ripples_results.npz
├── fast_ripples_results.npz
├── visualizations/
│   ├── local_sample/
│   │   └── test_condition/
│   │       ├── sharp_wave_ripples/
│   │       │   ├── channel_plots/
│   │       │   ├── heatmap.png
│   │       │   └── summary.png
│   │       └── fast_ripples/...
│   └── visualizations.zip
├── local_summary.txt
└── used_config.yaml
```

### NPZ File Contents

Each oscillation detection result is saved as an NPZ file containing:

```python
{
    'ripples': {
        # Per-channel detection results
        0: pd.DataFrame,  # Channel 0 detections
        1: pd.DataFrame,  # Channel 1 detections
        ...
        'metadata': {
            'fs': 20000,
            'low_threshold': 3.5,
            'high_threshold': 5.0,
            # ... other parameters
        }
    },
    'summary_stats': {
        'total_detections': 1234,
        'mean_rate': 2.5,           # Hz
        'mean_duration': 45.2,      # ms
        'mean_power': 6.8,          # z-score
        'channels_with_detections': 64
    },
    'metadata': {
        'sample': 'culture_001',
        'condition': 'baseline',
        'oscillation_type': 'sharp_wave_ripples',
        'processing_date': '2024-01-20T10:30:00',
        'parameters': {...}
    }
}
```

## Parameter Tuning

### Detection Thresholds

**Low Threshold** (`low_threshold`): Starting/ending threshold in standard deviations
- Higher values: More conservative detection, fewer false positives
- Lower values: More sensitive detection, may include more noise
- Typical range: 2.0-4.0

**High Threshold** (`high_threshold`): Peak threshold that must be exceeded
- Should be higher than low_threshold
- Controls minimum peak amplitude
- Typical range: 3.0-6.0

### Temporal Constraints

**Duration Limits** (`min_duration`, `max_duration`): Event duration in milliseconds
- Sharp wave ripples: 20-200 ms
- Fast ripples: 10-100 ms  
- Gamma bursts: 50-500 ms

**Minimum Interval** (`min_interval`): Minimum time between events (ms)
- Prevents detection of overlapping events
- Typical range: 20-50 ms

### Sharp Wave Validation

For sharp wave ripples, enable validation against coincident sharp waves:

```yaml
require_sharp_wave: true
sharp_wave_window: 50        # ±50ms search window
sharp_wave_threshold: 2.0    # Sharp wave detection threshold
```

### Frequency Bands

Ensure frequency bands match your oscillation types:

```yaml
frequency_bands:
  - name: "narrowRipples"     # For sharp wave ripples
    low: 150
    high: 250
  - name: "fastRipples"       # For fast ripples  
    low: 250
    high: 500
  - name: "gammaEnvelope"     # For gamma bursts
    low: 30
    high: 100
```

## Advanced Usage

### Custom Oscillation Types

Add custom oscillation detection by defining new types in the configuration:

```yaml
oscillation_types:
  custom_oscillation:
    narrowband_key: "customBand"
    wideband_key: "customWide" 
    low_threshold: 2.0
    high_threshold: 4.0
    min_duration: 30
    max_duration: 150
    require_sharp_wave: false
```

### Batch Processing

Process multiple experiments by creating separate configuration files:

```bash
for config in configs/*.yaml; do
    python main.py "$config"
done
```

### GPU Acceleration

Enable GPU processing for large datasets by modifying the processor initialization:

```python
# In data_loader.py
processor.add_frequency_band(
    low, high, band_name=name,
    use_gpu=True,  # Enable GPU
    store_analytical=True
)
```

### Custom Visualization

Extend the visualizer with custom plots:

```python
from src.visualizer import OscillationVisualizer

class CustomVisualizer(OscillationVisualizer):
    def create_custom_plot(self, results, output_dir):
        # Your custom visualization code
        pass
```

## API Reference

### Core Classes

#### `OscillationProcessor`
Main processing class for oscillation detection.

```python
processor = OscillationProcessor()
results = processor.process_oscillations(
    lfp_processor, oscillation_params, 
    oscillation_type, sample_name, condition_name
)
```

#### `OscillationVisualizer`  
Visualization generation and management.

```python
visualizer = OscillationVisualizer()
visualizer.visualize_all_oscillations(
    results_dict, output_dir, config,
    lfp_processor, sample_name, condition_name
)
```

#### `StatsAggregator`
Statistical analysis across conditions.

```python
aggregator = StatsAggregator()
aggregated_data = aggregator.aggregate_across_conditions(all_results, config)
aggregator.create_violin_plots(aggregated_data, output_dir)
```

#### `S3Handler`
Cloud storage operations.

```python
s3_handler = S3Handler()
s3_handler.upload_file(local_path, s3_path)
s3_handler.download_file(s3_path, local_path)
```

### Utility Functions

#### `load_recording_data(file_paths)`
Load neural recording data and create LFPDataProcessor.

```python
from src.utils.data_loader import load_recording_data

file_paths = {
    'h5': 'data.h5',
    'zip': 'spikes.zip', 
    'npz': 'lfp.npz'
}
lfp_processor = load_recording_data(file_paths)
```

## Troubleshooting

### Common Issues

**Memory Errors**: Reduce batch size or disable GPU processing
```yaml
# In configuration
advanced_processing:
  use_gpu: false
  batch_size: 50
```

**No Detections Found**: Lower detection thresholds
```yaml
oscillation_types:
  sharp_wave_ripples:
    low_threshold: 2.0    # Lower from 3.5
    high_threshold: 3.5   # Lower from 5.0
```

**S3 Access Denied**: Check AWS credentials and bucket permissions
```bash
aws configure list
aws s3 ls s3://your-bucket/
```

**Docker Build Fails**: Check network connectivity and update package versions
```dockerfile
# In Dockerfile, pin specific versions
RUN pip install numpy==1.24.0 scipy==1.11.0
```

### Performance Optimization

- Use SSD storage for local processing
- Enable GPU acceleration for large datasets  
- Process samples in parallel using multiple job instances
- Use smaller time windows for memory-constrained environments

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Implement** your changes with tests
4. **Update** documentation as needed
5. **Submit** a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r docker/requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new functionality
- Update README for new features

## Citation

If you use Axolotl in your research, please cite:

```bibtex
@software{axolotl2024,
  title={Axolotl: Neural Oscillation Detection Pipeline},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-org]/axolotl}
}
```

### Detection Method Citation

The oscillation detection is based on Buzsáki's approach. Please also cite:

```bibtex
@article{csicsvari2003mechanisms,
  title={Mechanisms of gamma oscillations in the hippocampus of the behaving rat},
  author={Csicsvari, Jozsef and Jamieson, Brian and Wise, Kensall D and Buzs{\'a}ki, Gy{\"o}rgy},
  journal={Neuron},
  volume={37},
  number={2},
  pages={311--322},
  year={2003},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Submit bug reports and feature requests via GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions
- **Contact**: [your-email@institution.edu]

---

*Axolotl - Because neural oscillations should regenerate insights, not headaches.*