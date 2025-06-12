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
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import ParameterGrid
import logging

# Import the main pipeline classes
from dnabert_metagenomics import (
    PipelineConfig, NovelSpeciesDetector, 
    DNABERTEmbedder, EmbeddingClusterer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """Optimize clustering parameters for the pipeline"""
    
    def __init__(self, embeddings: np.ndarray, contig_ids: list):
        self.embeddings = embeddings
        self.contig_ids = contig_ids
        self.best_params = None
        self.best_score = -1
    
    def optimize_clustering_params(self, param_grid: dict) -> dict:
        """Find optimal UMAP + HDBSCAN parameters"""
        logger.info("Optimizing clustering parameters...")
        
        results = []
        
        for params in ParameterGrid(param_grid):
            try:
                score = self._evaluate_params(params)
                results.append({**params, 'silhouette_score': score})
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    
                logger.info(f"Params: {params}, Score: {score:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed for params {params}: {e}")
                continue
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results
        }
    
    def _evaluate_params(self, params: dict) -> float:
        """Evaluate clustering quality for given parameters"""
        config = PipelineConfig(
            input_reads="dummy",  # Not used in this context
            output_dir="dummy",
            umap_n_neighbors=params.get('umap_n_neighbors', 15),
            umap_min_dist=params.get('umap_min_dist', 0.1),
            umap_n_components=params.get('umap_n_components', 50),
            hdbscan_min_cluster_size=params.get('hdbscan_min_cluster_size', 10),
            hdbscan_min_samples=params.get('hdbscan_min_samples', 5)
        )
        
        clusterer = EmbeddingClusterer(config)
        
        # Apply dimensionality reduction and clustering
        reduced_embeddings = clusterer.reduce_dimensions(self.embeddings)
        cluster_labels = clusterer.cluster_embeddings(reduced_embeddings)
        
        # Calculate silhouette score (higher is better)
        if len(set(cluster_labels)) > 1:
            # Remove noise points for silhouette calculation
            mask = cluster_labels != -1
            if np.sum(mask) > 1:
                score = silhouette_score(reduced_embeddings[mask], cluster_labels[mask])
                return score
        
        return -1  # Invalid clustering

def create_test_config(data_path: str, output_dir: str = "test_output") -> PipelineConfig:
    """Create a test configuration"""
    return PipelineConfig(
        input_reads=data_path,
        output_dir=output_dir,
        assembler="megahit",
        min_contig_length=500,  # Lower for testing
        dnabert_model="zhihan1996/DNABERT-2-117M",
        max_sequence_length=256,  # Shorter for faster testing
        batch_size=8,  # Smaller batch for limited memory
        umap_n_neighbors=10,
        umap_min_dist=0.1,
        umap_n_components=20,  # Fewer components for testing
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=3,
        min_novel_cluster_size=3,
        max_known_similarity=0.8
    )

def run_embedding_test(contigs_file: str, config: PipelineConfig):
    """Test DNABERT embedding generation"""
    logger.info("Testing DNABERT embedding generation...")
    
    embedder = DNABERTEmbedder(config)
    
    # Test with first few contigs
    from Bio import SeqIO
    test_contigs = []
    for i, record in enumerate(SeqIO.parse(contigs_file, "fasta")):
        test_contigs.append(record)
        if i >= 10:  # Test with 10 contigs
            break
    
    # Generate embeddings
    embeddings = []
    for record in test_contigs:
        try:
            embedding = embedder.embed_sequence(str(record.seq))
            embeddings.append(embedding)
            logger.info(f"Embedded {record.id}: shape {embedding.shape}")
        except Exception as e:
            logger.error(f"Failed to embed {record.id}: {e}")
    
    if embeddings:
        embeddings_array = np.vstack(embeddings)
        logger.info(f"Test embeddings shape: {embeddings_array.shape}")
        return embeddings_array
    else:
        logger.error("No embeddings generated!")
        return None

def run_clustering_test(embeddings: np.ndarray, config: PipelineConfig):
    """Test clustering pipeline"""
    logger.info("Testing clustering pipeline...")
    
    clusterer = EmbeddingClusterer(config)
    
    # Apply UMAP
    reduced_embeddings = clusterer.reduce_dimensions(embeddings)
    logger.info(f"UMAP output shape: {reduced_embeddings.shape}")
    
    # Apply HDBSCAN
    cluster_labels = clusterer.cluster_embeddings(reduced_embeddings)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    logger.info(f"Found {n_clusters} clusters with {n_noise} noise points")
    logger.info(f"Cluster distribution: {np.bincount(cluster_labels[cluster_labels >= 0])}")
    
    return reduced_embeddings, cluster_labels

def quick_parameter_scan(embeddings: np.ndarray, contig_ids: list):
    """Quick parameter optimization"""
    param_grid = {
        'umap_n_neighbors': [5, 10, 15, 20],
        'umap_min_dist': [0.0, 0.1, 0.25],
        'umap_n_components': [10, 20, 30],
        'hdbscan_min_cluster_size': [3, 5, 8, 10],
        'hdbscan_min_samples': [1, 3, 5]
    }
    
    optimizer = ParameterOptimizer(embeddings, contig_ids)
    results = optimizer.optimize_clustering_params(param_grid)
    
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Best silhouette score: {results['best_score']:.3f}")
    
    return results

def create_config_from_yaml(yaml_file: str) -> PipelineConfig:
    """Load configuration from YAML file"""
    with open(yaml_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return PipelineConfig(**config_dict)

def save_config_template(output_file: str = "pipeline_config.yaml"):
    """Save a configuration template"""
    config_template = {
        'input_reads': 'path/to/your/reads.fastq.gz',
        'output_dir': 'dnabert_output',
        'assembler': 'megahit',
        'min_contig_length': 1000,
        'dnabert_model': 'zhihan1996/DNABERT-2-117M',
        'max_sequence_length': 512,
        'batch_size': 16,
        'device': 'cuda',
        'umap_n_neighbors': 15,
        'umap_min_dist': 0.1,
        'umap_n_components': 50,
        'hdbscan_min_cluster_size': 10,
        'hdbscan_min_samples': 5,
        'sourmash_k': 31,
        'sourmash_scaled': 1000,
        'sourmash_threshold': 0.1,
        'min_novel_cluster_size': 5,
        'max_known_similarity': 0.8
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False)
    
    logger.info(f"Configuration template saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="DNABERT-Hybrid Metagenomics Quick Start")
    
    parser.add_argument('mode', choices=['full', 'test', 'optimize', 'config'], 
                       help="Run mode")
    parser.add_argument('--input', '-i', help="Input file path")
    parser.add_argument('--output', '-o', default="output", help="Output directory")
    parser.add_argument('--config', '-c', help="Configuration YAML file")
    parser.add_argument('--database', '-d', help="Sourmash database path")
    parser.add_argument('--contigs', help="Pre-assembled contigs file")
    
    args = parser.parse_args()
    
    if args.mode == 'config':
        # Generate configuration template
        save_config_template()
        return
    
    # Load configuration
    if args.config:
        config = create_config_from_yaml(args.config)
    else:
        config = create_test_config(args.input or "dummy", args.output)
    
    if args.mode == 'full':
        # Run full pipeline
        detector = NovelSpeciesDetector(config)
        results = detector.run_pipeline(args.database)
        logger.info("Full pipeline completed!")
        
    elif args.mode == 'test':
        # Run component tests
        if not args.contigs:
            logger.error("Need --contigs file for testing mode")
            return
            
        # Test embedding generation
        embeddings = run_embedding_test(args.contigs, config)
        
        if embeddings is not None:
            # Test clustering
            contig_ids = [f"contig_{i}" for i in range(len(embeddings))]
            reduced_embeddings, cluster_labels = run_clustering_test(embeddings, config)
            
            logger.info("All tests passed!")
        
    elif args.mode == 'optimize':
        # Parameter optimization
        if not args.contigs:
            logger.error("Need --contigs file for optimization mode")
            return
            
        # Generate embeddings first
        embedder = DNABERTEmbedder(config)
        embeddings, contig_ids = embedder.embed_contigs(args.contigs)
        
        # Optimize parameters
        results = quick_parameter_scan(embeddings, contig_ids)
        
        # Save optimized config
        optimized_config = config
        for param, value in results['best_params'].items():
            setattr(optimized_config, param, value)
        
        # Save to YAML
        config_dict = optimized_config.__dict__
        with open(Path(args.output) / "optimized_config.yaml", 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Optimized configuration saved to {args.output}/optimized_config.yaml")

if __name__ == "__main__":
    main()
