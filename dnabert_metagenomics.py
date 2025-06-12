#!/usr/bin/env python3
"""
DNABERT-Hybrid Metagenomics Pipeline for Novel Species Detection
Combines DNABERT embeddings with UMAP+HDBSCAN clustering and Sourmash validation
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import pickle

# Deep learning and embedding libraries
import torch
from transformers import AutoTokenizer, AutoModel
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Bioinformatics libraries
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sourmash

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the metagenomics pipeline"""
    # Input/Output paths
    input_reads: str
    output_dir: str
    
    # Assembly parameters
    assembler: str = "megahit"  # or "spades", "metaflye"
    min_contig_length: int = 1000
    
    # DNABERT parameters
    dnabert_model: str = "zhihan1996/DNABERT-2-117M"
    max_sequence_length: int = 512
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Clustering parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 50
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5
    
    # Sourmash parameters
    sourmash_k: int = 31
    sourmash_scaled: int = 1000
    sourmash_threshold: float = 0.1
    
    # Validation parameters
    min_novel_cluster_size: int = 5
    max_known_similarity: float = 0.8

class MetagenomicsPreprocessor:
    """Handles preprocessing and assembly of metagenomic reads"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def quality_control(self, input_reads: str) -> str:
        """Perform quality control using fastp"""
        logger.info("Performing quality control with fastp")
        
        output_reads = self.output_dir / "qc_reads.fastq.gz"
        fastp_cmd = [
            "fastp",
            "-i", input_reads,
            "-o", str(output_reads),
            "--detect_adapter_for_pe",
            "--correction",
            "--cut_front", "--cut_tail",
            "--thread", "8",
            "--html", str(self.output_dir / "fastp_report.html")
        ]
        
        try:
            subprocess.run(fastp_cmd, check=True, capture_output=True)
            logger.info(f"Quality control completed: {output_reads}")
            return str(output_reads)
        except subprocess.CalledProcessError as e:
            logger.error(f"fastp failed: {e}")
            raise
    
    def meta_assembly(self, qc_reads: str) -> str:
        """Perform metagenomic assembly"""
        logger.info(f"Performing meta-assembly with {self.config.assembler}")
        
        assembly_dir = self.output_dir / "assembly"
        assembly_dir.mkdir(exist_ok=True)
        
        if self.config.assembler == "megahit":
            return self._run_megahit(qc_reads, assembly_dir)
        elif self.config.assembler == "spades":
            return self._run_spades(qc_reads, assembly_dir)
        else:
            raise ValueError(f"Unsupported assembler: {self.config.assembler}")
    
    def _run_megahit(self, reads: str, output_dir: Path) -> str:
        """Run MEGAHIT assembler"""
        contigs_file = output_dir / "final.contigs.fa"
        
        megahit_cmd = [
            "megahit",
            "-r", reads,
            "-o", str(output_dir),
            "--min-contig-len", str(self.config.min_contig_length),
            "-t", "8"
        ]
        
        subprocess.run(megahit_cmd, check=True)
        return str(contigs_file)
    
    def _run_spades(self, reads: str, output_dir: Path) -> str:
        """Run metaSPAdes assembler"""
        contigs_file = output_dir / "contigs.fasta"
        
        spades_cmd = [
            "metaspades.py",
            "-s", reads,
            "-o", str(output_dir),
            "-t", "8"
        ]
        
        subprocess.run(spades_cmd, check=True)
        return str(contigs_file)
    
    def filter_contigs(self, contigs_file: str) -> str:
        """Filter contigs by length and quality"""
        logger.info("Filtering contigs")
        
        filtered_file = self.output_dir / "filtered_contigs.fasta"
        filtered_contigs = []
        
        for record in SeqIO.parse(contigs_file, "fasta"):
            if len(record.seq) >= self.config.min_contig_length:
                # Additional quality filters can be added here
                filtered_contigs.append(record)
        
        SeqIO.write(filtered_contigs, filtered_file, "fasta")
        logger.info(f"Filtered to {len(filtered_contigs)} contigs")
        
        return str(filtered_file)

class DNABERTEmbedder:
    """Generates DNABERT embeddings for DNA sequences"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load DNABERT model and tokenizer
        logger.info(f"Loading DNABERT model: {config.dnabert_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.dnabert_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(config.dnabert_model, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
    
    def sequence_to_kmers(self, sequence: str, k: int = 6) -> str:
        """Convert DNA sequence to k-mers for DNABERT"""
        return ' '.join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])
    
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Generate embedding for a single sequence"""
        # Truncate sequence if too long
        if len(sequence) > self.config.max_sequence_length:
            sequence = sequence[:self.config.max_sequence_length]
        
        # Convert to k-mers
        kmer_sequence = self.sequence_to_kmers(sequence)
        
        # Tokenize and encode
        inputs = self.tokenizer(kmer_sequence, return_tensors="pt", 
                              truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embedding.flatten()
    
    def embed_contigs(self, contigs_file: str) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for all contigs"""
        logger.info("Generating DNABERT embeddings for contigs")
        
        embeddings = []
        contig_ids = []
        
        for i, record in enumerate(SeqIO.parse(contigs_file, "fasta")):
            if i % 100 == 0:
                logger.info(f"Processed {i} contigs")
            
            try:
                embedding = self.embed_sequence(str(record.seq))
                embeddings.append(embedding)
                contig_ids.append(record.id)
            except Exception as e:
                logger.warning(f"Failed to embed contig {record.id}: {e}")
                continue
        
        embeddings_array = np.vstack(embeddings)
        logger.info(f"Generated embeddings for {len(contig_ids)} contigs")
        
        return embeddings_array, contig_ids

class EmbeddingClusterer:
    """Performs UMAP dimensionality reduction and HDBSCAN clustering"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.umap_reducer = None
        self.clusterer = None
        self.scaler = StandardScaler()
    
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP dimensionality reduction"""
        logger.info("Applying UMAP dimensionality reduction")
        
        # Standardize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Apply UMAP
        self.umap_reducer = umap.UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            min_dist=self.config.umap_min_dist,
            n_components=self.config.umap_n_components,
            random_state=42,
            metric='cosine'
        )
        
        reduced_embeddings = self.umap_reducer.fit_transform(embeddings_scaled)
        logger.info(f"Reduced dimensions to {reduced_embeddings.shape}")
        
        return reduced_embeddings
    
    def cluster_embeddings(self, reduced_embeddings: np.ndarray) -> np.ndarray:
        """Apply HDBSCAN clustering"""
        logger.info("Applying HDBSCAN clustering")
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            metric='euclidean'
        )
        
        cluster_labels = self.clusterer.fit_predict(reduced_embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Found {n_clusters} clusters with {n_noise} noise points")
        
        return cluster_labels
    
    def visualize_clusters(self, reduced_embeddings: np.ndarray, 
                          cluster_labels: np.ndarray, output_path: str):
        """Create visualization of clusters"""
        logger.info("Creating cluster visualization")
        
        # Reduce to 2D for visualization if needed
        if reduced_embeddings.shape[1] > 2:
            umap_2d = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = umap_2d.fit_transform(reduced_embeddings)
        else:
            embeddings_2d = reduced_embeddings
        
        # Create interactive plot
        fig = px.scatter(
            x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
            color=cluster_labels,
            title="DNABERT Embedding Clusters",
            labels={'color': 'Cluster'},
            color_continuous_scale='viridis'
        )
        
        fig.write_html(output_path)
        logger.info(f"Cluster visualization saved to {output_path}")

class SourmashValidator:
    """Validates clusters using Sourmash for taxonomic assignment"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.signature_dir = Path(config.output_dir) / "sourmash_signatures"
        self.signature_dir.mkdir(exist_ok=True)
    
    def create_signatures(self, contigs_file: str, cluster_labels: np.ndarray, 
                         contig_ids: List[str]) -> Dict[int, str]:
        """Create Sourmash signatures for each cluster"""
        logger.info("Creating Sourmash signatures for clusters")
        
        cluster_signatures = {}
        cluster_contigs = defaultdict(list)
        
        # Group contigs by cluster
        for contig_id, cluster_label in zip(contig_ids, cluster_labels):
            if cluster_label != -1:  # Skip noise points
                cluster_contigs[cluster_label].append(contig_id)
        
        # Create signatures for each cluster
        contigs_dict = {record.id: record for record in SeqIO.parse(contigs_file, "fasta")}
        
        for cluster_id, contig_list in cluster_contigs.items():
            if len(contig_list) < self.config.min_novel_cluster_size:
                continue
            
            # Combine contigs from this cluster
            cluster_sequences = []
            for contig_id in contig_list:
                if contig_id in contigs_dict:
                    cluster_sequences.append(contigs_dict[contig_id])
            
            # Create signature file
            sig_file = self.signature_dir / f"cluster_{cluster_id}.sig"
            fasta_file = self.signature_dir / f"cluster_{cluster_id}.fasta"
            
            # Write cluster sequences to FASTA
            SeqIO.write(cluster_sequences, fasta_file, "fasta")
            
            # Create Sourmash signature
            self._create_signature(str(fasta_file), str(sig_file))
            cluster_signatures[cluster_id] = str(sig_file)
        
        return cluster_signatures
    
    def _create_signature(self, fasta_file: str, sig_file: str):
        """Create a Sourmash signature from FASTA file"""
        sourmash_cmd = [
            "sourmash", "sketch", "dna",
            "-p", f"k={self.config.sourmash_k},scaled={self.config.sourmash_scaled}",
            "--name-from-first",
            "-o", sig_file,
            fasta_file
        ]
        
        subprocess.run(sourmash_cmd, check=True, capture_output=True)
    
    def search_against_database(self, cluster_signatures: Dict[int, str], 
                               database_path: str) -> Dict[int, List[Dict]]:
        """Search cluster signatures against reference database"""
        logger.info("Searching clusters against reference database")
        
        search_results = {}
        
        for cluster_id, sig_file in cluster_signatures.items():
            search_cmd = [
                "sourmash", "search",
                sig_file, database_path,
                "--threshold", str(self.config.sourmash_threshold),
                "--csv", f"{self.signature_dir}/cluster_{cluster_id}_search.csv"
            ]
            
            try:
                result = subprocess.run(search_cmd, check=True, capture_output=True, text=True)
                
                # Parse search results
                csv_file = f"{self.signature_dir}/cluster_{cluster_id}_search.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    search_results[cluster_id] = df.to_dict('records')
                else:
                    search_results[cluster_id] = []
                    
            except subprocess.CalledProcessError:
                logger.warning(f"No matches found for cluster {cluster_id}")
                search_results[cluster_id] = []
        
        return search_results

class NovelSpeciesDetector:
    """Main pipeline class that orchestrates the entire workflow"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = MetagenomicsPreprocessor(config)
        self.embedder = DNABERTEmbedder(config)
        self.clusterer = EmbeddingClusterer(config)
        self.validator = SourmashValidator(config)
    
    def run_pipeline(self, database_path: Optional[str] = None) -> Dict:
        """Run the complete pipeline"""
        logger.info("Starting DNABERT-Hybrid Metagenomics Pipeline")
        
        results = {}
        
        # Step 1: Preprocessing and Assembly
        logger.info("Step 1: Preprocessing and Assembly")
        qc_reads = self.preprocessor.quality_control(self.config.input_reads)
        contigs_file = self.preprocessor.meta_assembly(qc_reads)
        filtered_contigs = self.preprocessor.filter_contigs(contigs_file)
        results['contigs_file'] = filtered_contigs
        
        # Step 2: Generate DNABERT Embeddings
        logger.info("Step 2: Generating DNABERT Embeddings")
        embeddings, contig_ids = self.embedder.embed_contigs(filtered_contigs)
        results['embeddings'] = embeddings
        results['contig_ids'] = contig_ids
        
        # Save embeddings
        embeddings_file = self.output_dir / "dnabert_embeddings.npz"
        np.savez(embeddings_file, embeddings=embeddings, contig_ids=contig_ids)
        
        # Step 3: Clustering
        logger.info("Step 3: Clustering Embeddings")
        reduced_embeddings = self.clusterer.reduce_dimensions(embeddings)
        cluster_labels = self.clusterer.cluster_embeddings(reduced_embeddings)
        results['cluster_labels'] = cluster_labels
        
        # Visualize clusters
        viz_file = self.output_dir / "cluster_visualization.html"
        self.clusterer.visualize_clusters(reduced_embeddings, cluster_labels, str(viz_file))
        
        # Step 4: Sourmash Validation
        if database_path:
            logger.info("Step 4: Sourmash Validation")
            cluster_signatures = self.validator.create_signatures(
                filtered_contigs, cluster_labels, contig_ids
            )
            search_results = self.validator.search_against_database(
                cluster_signatures, database_path
            )
            results['search_results'] = search_results
        
        # Step 5: Identify Novel Species
        novel_clusters = self.identify_novel_species(
            cluster_labels, search_results if database_path else None
        )
        results['novel_clusters'] = novel_clusters
        
        # Generate final report
        self.generate_report(results)
        
        logger.info("Pipeline completed successfully!")
        return results
    
    def identify_novel_species(self, cluster_labels: np.ndarray, 
                              search_results: Optional[Dict] = None) -> List[int]:
        """Identify clusters that represent novel species"""
        novel_clusters = []
        
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # Remove noise cluster
        
        for cluster_id in unique_clusters:
            cluster_size = np.sum(cluster_labels == cluster_id)
            
            # Filter by minimum cluster size
            if cluster_size < self.config.min_novel_cluster_size:
                continue
            
            # Check Sourmash results if available
            is_novel = True
            if search_results and cluster_id in search_results:
                for match in search_results[cluster_id]:
                    if match.get('similarity', 0) > self.config.max_known_similarity:
                        is_novel = False
                        break
            
            if is_novel:
                novel_clusters.append(cluster_id)
                logger.info(f"Identified novel cluster {cluster_id} with {cluster_size} contigs")
        
        return novel_clusters
    
    def generate_report(self, results: Dict):
        """Generate a comprehensive report"""
        report_file = self.output_dir / "pipeline_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("DNABERT-Hybrid Metagenomics Pipeline Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total contigs processed: {len(results['contig_ids'])}\n")
            f.write(f"Embedding dimensions: {results['embeddings'].shape}\n")
            
            unique_clusters = set(results['cluster_labels'])
            unique_clusters.discard(-1)
            f.write(f"Number of clusters found: {len(unique_clusters)}\n")
            
            noise_points = np.sum(results['cluster_labels'] == -1)
            f.write(f"Noise points: {noise_points}\n\n")
            
            if 'novel_clusters' in results:
                f.write(f"Novel species candidates: {len(results['novel_clusters'])}\n")
                for cluster_id in results['novel_clusters']:
                    cluster_size = np.sum(results['cluster_labels'] == cluster_id)
                    f.write(f"  - Cluster {cluster_id}: {cluster_size} contigs\n")
        
        logger.info(f"Report saved to {report_file}")

# Example usage and configuration
def main():
    """Example usage of the pipeline"""
    
    # Configure the pipeline
    config = PipelineConfig(
        input_reads="path/to/metagenomic_reads.fastq.gz",
        output_dir="dnabert_metagenomics_output",
        assembler="megahit",
        min_contig_length=1000,
        dnabert_model="zhihan1996/DNABERT-2-117M",
        max_sequence_length=512,
        batch_size=16,  # Adjust based on GPU memory
        umap_n_components=50,
        hdbscan_min_cluster_size=10,
        min_novel_cluster_size=5,
        max_known_similarity=0.8
    )
    
    # Initialize and run pipeline
    detector = NovelSpeciesDetector(config)
    
    # Optional: provide path to Sourmash database for validation
    database_path = "path/to/sourmash_database.sbt.zip"  # or None
    
    # Run the complete pipeline
    results = detector.run_pipeline(database_path)
    
    print(f"Pipeline completed! Found {len(results['novel_clusters'])} novel species candidates.")
    print(f"Results saved to: {config.output_dir}")

if __name__ == "__main__":
    main()

