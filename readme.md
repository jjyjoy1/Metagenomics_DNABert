# DNABERT-Hybrid Metagenomics Pipeline for Novel Species Detection

A comprehensive pipeline combining DNABERT with traditional metagenomics approaches for novel species detection.

## Pipeline Overview

1. **Preprocessing & Meta-Assembly**
   - Quality control with `fastp`
   - Assembly with MEGAHIT/SPAdes

2. **DNABERT Embeddings**
   - Converts contigs to sequence embeddings using DNABERT-2

3. **Clustering**
   - UMAP dimensionality reduction
   - HDBSCAN clustering to group similar sequences

4. **Sourmash Validation**
   - Creates k-mer signatures
   - Searches against reference databases

5. **Novel Species Detection**
   - Identifies clusters with low similarity to known species

## Key Features

- **Scalable**: Handles large metagenomic datasets with batch processing
- **Configurable**: Easy parameter tuning for different datasets
- **Validation**: Multiple validation steps using alignment-free methods
- **Visualization**: Interactive cluster plots for analysis
- **Comprehensive Reporting**: Detailed output with candidate novel species

---

## Quick Start Guide

### Installation Requirements

```bash
# Install bioinformatics tools
conda install -c bioconda megahit spades fastp sourmash

# Install Python dependencies
pip install torch transformers umap-learn hdbscan scikit-learn
pip install biopython pandas numpy matplotlib seaborn plotly
pip install sourmash pyyaml

# For DNABERT-2 specifically
pip install triton  # For GPU acceleration
```

### Basic Usage

```bash
# Generate configuration template
python quick_start_script.py config

# Test with small dataset
python quick_start_script.py test --contigs your_contigs.fasta --output test_run

# Optimize parameters
python quick_start_script.py optimize --contigs your_contigs.fasta --output optimization

# Run full pipeline
python quick_start_script.py full --input reads.fastq.gz --config optimized_config.yaml --database sourmash_db.sbt.zip
```

---

## Technical Advantages

### Core Benefits

1. **Semantic Understanding**  
   DNABERT captures sequence patterns that traditional k-mer methods might miss

2. **Robust Clustering**  
   UMAP+HDBSCAN handles complex, non-spherical clusters better than k-means

3. **Validation**  
   Sourmash provides independent confirmation using alignment-free k-mer signatures

4. **Scalability**  
   Batch processing and GPU acceleration for large datasets

### Parameter Tuning Guide

| Parameter            | Recommendation                     | Notes                          |
|----------------------|------------------------------------|--------------------------------|
| UMAP n_neighbors     | Start with 15                      | Increase for smoother manifolds|
| HDBSCAN min_cluster  | Adjust based on expected abundance |                                |
| Sourmash threshold   | 0.05-0.1                           | Lower values = stricter        |
| Contig length        | Balance resolution vs cost         |                                |

---

## Expected Outputs

- 📊 Cluster visualizations showing potential novel species groups
- 🔍 Sourmash search results indicating similarity to known taxa
- 🆕 Candidate novel species with supporting evidence
- 📝 Comprehensive reports with cluster statistics

## Pipeline Script

```python
#!/usr/bin/env python3
"""
Quick Start Script for DNABERT-Hybrid Metagenomics Pipeline
Includes parameter optimization and testing utilities
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
```

---

## Scientific Value Proposition

This hybrid approach is particularly effective for:

- Detecting divergent strains
- Identifying potentially novel species missed by traditional homology-based methods
- Combining deep learning sequence understanding with established bioinformatics validation
- Providing robust novel species detection with multiple evidence lines
```
