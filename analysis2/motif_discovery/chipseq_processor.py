#!/usr/bin/env python3
"""
Stratified ChIP-seq Processing Pipeline
Replaces the current bash script and separate Python files with a modular approach.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for ChIP-seq processing"""
    interval_size: int = 200
    allowed_chromosomes: List[str] = None
    max_peak_size: int = 4000
    blacklist_file: str = "past_data/blacklist.bed"
    genome_fasta: str = "past_data/hg38.fa"
    samtools_path: str = "./miniconda3/envs/test/bin/samtools"
    bedtools_path: str = "./miniconda3/envs/test/bin/bedtools"
    
    def __post_init__(self):
        if self.allowed_chromosomes is None:
            self.allowed_chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

class ChIPSeqProcessor:
    """
    Stratified ChIP-seq processing pipeline
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._validate_tools()
    
    def _validate_tools(self):
        """Validate that required tools are available"""
        tools = {
            'samtools': self.config.samtools_path,
            'bedtools': self.config.bedtools_path
        }
        
        for tool_name, tool_path in tools.items():
            if not os.path.exists(tool_path):
                raise FileNotFoundError(f"{tool_name} not found at {tool_path}")
    
    def process_experiment(self, experiment_dir: str, target_directory: str = None) -> Dict[str, str]:
        """
        Main processing pipeline for a single experiment
        
        Args:
            experiment_dir: Path to experiment directory
            target_directory: Specific directory to process (if None, process all)
            
        Returns:
            Dictionary with paths to processed files
        """
        logger.info(f"Starting processing for {experiment_dir}")
        
        # Phase 1: Setup and Validation
        experiment_info = self._setup_experiment(experiment_dir, target_directory)
        if not experiment_info:
            logger.warning(f"Skipping {experiment_dir} - setup failed")
            return {}
        
        # Phase 2: BAM Processing
        filtered_bams = self._process_bam_files(experiment_info)
        
        # Phase 3: Peak Processing
        normalized_peaks = self._process_peaks(experiment_info)
        
        # Phase 4: Sequence Processing
        sequences = self._extract_sequences(normalized_peaks)
        clean_sequences = self._handle_ambiguous_bases(sequences, normalized_peaks)
        
        # Phase 5: Feature Engineering
        one_hot_data = self._create_one_hot_encoding(clean_sequences, experiment_info['test_dir'])
        coverage_data = self._calculate_coverage(normalized_peaks, filtered_bams, experiment_info)
        
        logger.info(f"Completed processing for {experiment_dir}")
        
        return {
            'one_hot_file': one_hot_data,
            'coverage_file': coverage_data['coverage'],
            'control_file': coverage_data['control']
        }
    
    def _setup_experiment(self, experiment_dir: str, target_directory: str = None) -> Optional[Dict]:
        """Phase 1: Setup and validation"""
        logger.info("Phase 1: Setting up experiment")
        
        # Check if this is the target directory
        dir_name = os.path.basename(experiment_dir)
        if target_directory and dir_name != target_directory:
            logger.info(f"Skipping {dir_name} - not target directory")
            return None
        
        # Define paths
        control_dir = os.path.join(experiment_dir, "control")
        alignment_dir = os.path.join(experiment_dir, "alignment")
        experiment_200_dir = os.path.join(experiment_dir, "experiment200")
        test_dir = os.path.join(experiment_dir, "test200")
        
        # Check if already processed
        if os.path.exists(test_dir):
            logger.info(f"Test directory already exists for {dir_name}. Skipping.")
            return None
        
        if os.path.exists(experiment_200_dir):
            logger.info(f"Experiment200 directory already exists for {dir_name}. Skipping.")
            return None
        
        # Create directories
        os.makedirs(experiment_200_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Copy raw peak file
        raw_peak_file = os.path.join(experiment_dir, "experiment", "rawpeak.bed")
        if os.path.exists(raw_peak_file):
            shutil.copy2(raw_peak_file, os.path.join(experiment_200_dir, "rawpeak.bed"))
        else:
            logger.error(f"Raw peak file not found: {raw_peak_file}")
            return None
        
        return {
            'dir_name': dir_name,
            'control_dir': control_dir,
            'alignment_dir': alignment_dir,
            'experiment_200_dir': experiment_200_dir,
            'test_dir': test_dir,
            'raw_peak_file': raw_peak_file
        }
    
    def _process_bam_files(self, experiment_info: Dict) -> Dict[str, str]:
        """Phase 2: BAM file processing"""
        logger.info("Phase 2: Processing BAM files")
        
        experiment_200_dir = experiment_info['experiment_200_dir']
        control_dir = experiment_info['control_dir']
        alignment_dir = experiment_info['alignment_dir']
        
        # Count BAM files
        alignment_bams = list(Path(alignment_dir).glob("*.bam"))
        control_bams = list(Path(control_dir).glob("*.bam"))
        
        if len(alignment_bams) == 1 and len(control_bams) == 1:
            # Single BAM files - filter directly
            logger.info("Processing single BAM files")
            filtered_alignment = os.path.join(experiment_200_dir, "alignment.f.bam")
            filtered_control = os.path.join(experiment_200_dir, "control.f.bam")
            
            self._filter_bam_file(str(alignment_bams[0]), filtered_alignment)
            self._filter_bam_file(str(control_bams[0]), filtered_control)
            
        else:
            # Multiple BAM files - merge first
            logger.info("Merging multiple BAM files")
            merged_alignment = os.path.join(experiment_200_dir, "alignment.bam")
            merged_control = os.path.join(experiment_200_dir, "control.bam")
            
            self._merge_bam_files([str(bam) for bam in alignment_bams], merged_alignment)
            self._merge_bam_files([str(bam) for bam in control_bams], merged_control)
            
            # Then filter
            filtered_alignment = os.path.join(experiment_200_dir, "alignment.f.bam")
            filtered_control = os.path.join(experiment_200_dir, "control.f.bam")
            
            self._filter_bam_file(merged_alignment, filtered_alignment)
            self._filter_bam_file(merged_control, filtered_control)
        
        return {
            'alignment': filtered_alignment,
            'control': filtered_control
        }
    
    def _filter_bam_file(self, input_bam: str, output_bam: str):
        """Filter BAM file to remove blacklisted regions"""
        cmd = [
            self.config.bedtools_path, "intersect",
            "-a", input_bam,
            "-b", self.config.blacklist_file,
            "-v"
        ]
        
        with open(output_bam, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)
        
        logger.info(f"Filtered BAM: {input_bam} -> {output_bam}")
    
    def _merge_bam_files(self, input_bams: List[str], output_bam: str):
        """Merge multiple BAM files"""
        cmd = [self.config.samtools_path, "merge", output_bam] + input_bams
        subprocess.run(cmd, check=True)
        logger.info(f"Merged BAM files -> {output_bam}")
    
    def _process_peaks(self, experiment_info: Dict) -> str:
        """Phase 3: Peak processing and normalization"""
        logger.info("Phase 3: Processing peaks")
        
        raw_peak_file = os.path.join(experiment_info['experiment_200_dir'], "rawpeak.bed")
        normalized_peak_file = os.path.join(experiment_info['experiment_200_dir'], "norm.bed")
        
        # Read and filter peaks
        peaks_df = pd.read_csv(raw_peak_file, sep='\t', header=None)
        peaks_df.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand', 
                           'signalValue', 'pValue', 'qValue', 'peak']
        
        # Filter by chromosome
        peaks_df = peaks_df[peaks_df['chrom'].isin(self.config.allowed_chromosomes)]
        
        # Filter by peak size
        peak_sizes = peaks_df['end'] - peaks_df['start']
        peaks_df = peaks_df[peak_sizes <= self.config.max_peak_size]
        
        # Normalize to fixed interval size
        interval_range = self.config.interval_size // 2
        
        # Check if peak summit column exists, otherwise use actual midpoint
        if 'peak' in peaks_df.columns and not peaks_df['peak'].isna().all():
            # Use reported peak summit
            midpoints = peaks_df['start'] + peaks_df['peak']
        else:
            # Use actual midpoint of the peak
            midpoints = peaks_df['start'] + (peaks_df['end'] - peaks_df['start']) // 2
            logger.info("No peak summit information found, using actual midpoint")
        
        peaks_df['start'] = midpoints - interval_range
        peaks_df['end'] = midpoints + interval_range
        
        # Save normalized peaks
        peaks_df.to_csv(normalized_peak_file, sep='\t', header=False, index=False)
        logger.info(f"Normalized peaks: {raw_peak_file} -> {normalized_peak_file}")
        
        return normalized_peak_file
    
    def _extract_sequences(self, normalized_peaks: str) -> str:
        """Phase 4a: Extract DNA sequences"""
        logger.info("Phase 4a: Extracting DNA sequences")
        
        experiment_200_dir = os.path.dirname(normalized_peaks)
        sequence_file = os.path.join(experiment_200_dir, "seq.out")
        
        cmd = [
            self.config.bedtools_path, "getfasta",
            "-fi", self.config.genome_fasta,
            "-bed", normalized_peaks,
            "-fo", sequence_file
        ]
        
        subprocess.run(cmd, check=True)
        logger.info(f"Extracted sequences: {sequence_file}")
        
        return sequence_file
    
    def _handle_ambiguous_bases(self, sequence_file: str, normalized_peaks: str) -> Tuple[List[str], str]:
        """Phase 4b: Handle sequences with ambiguous bases"""
        logger.info("Phase 4b: Handling ambiguous bases")
        
        # Read sequences
        with open(sequence_file, 'r') as f:
            content = f.read()
        
        sequences = content.split(">")[1:]  # Skip empty first element
        
        clean_sequences = []
        n_indices = []
        
        for i, seq_block in enumerate(sequences):
            lines = seq_block.strip().split('\n')
            if len(lines) >= 2:
                sequence = lines[1]
                if 'n' in sequence.lower():
                    n_indices.append(i)
                else:
                    clean_sequences.append(sequence)
        
        # If we found N bases, update the normalized peaks file
        if n_indices:
            logger.info(f"Found {len(n_indices)} sequences with N bases, removing them")
            
            peaks_df = pd.read_csv(normalized_peaks, sep='\t', header=None)
            peaks_df = peaks_df.drop(n_indices, axis=0)
            peaks_df.to_csv(normalized_peaks, sep='\t', header=False, index=False)
            
            # Re-extract sequences without N bases
            return self._extract_sequences(normalized_peaks), normalized_peaks
        
        return clean_sequences, normalized_peaks
    
    def _create_one_hot_encoding(self, sequences: List[str], test_dir: str) -> str:
        """Phase 5a: Create one-hot encoding"""
        logger.info("Phase 5a: Creating one-hot encoding")
        
        def one_hot_encode(seq):
            seq = seq.upper()
            bases = 'ACGT'
            base_dict = dict(zip(bases, range(len(bases))))
            one_hot = []
            for base in seq:
                if base in base_dict:
                    one_hot.append([True if j == base_dict[base] else False for j in range(len(bases))])
            return one_hot
        
        # Convert sequences to one-hot encoding
        one_hot_sequences = [one_hot_encode(seq) for seq in sequences]
        X = np.array(one_hot_sequences)
        
        # Save to file
        output_file = os.path.join(test_dir, "X.dat")
        X.dump(output_file)
        
        logger.info(f"One-hot encoding saved: {output_file}")
        return output_file
    
    def _calculate_coverage(self, normalized_peaks: str, filtered_bams: Dict[str, str], 
                          experiment_info: Dict) -> Dict[str, str]:
        """Phase 5b: Calculate coverage"""
        logger.info("Phase 5b: Calculating coverage")
        
        experiment_200_dir = experiment_info['experiment_200_dir']
        test_dir = experiment_info['test_dir']
        
        # Convert BAM to BED and sort
        alignment_bed = os.path.join(experiment_200_dir, "alignment.f.sorted.bed")
        control_bed = os.path.join(experiment_200_dir, "control.f.sorted.bed")
        
        self._bam_to_bed(filtered_bams['alignment'], alignment_bed)
        self._bam_to_bed(filtered_bams['control'], control_bed)
        
        # Calculate coverage
        alignment_coverage = os.path.join(experiment_200_dir, "y.counts.bed")
        control_coverage = os.path.join(experiment_200_dir, "y.control.counts.bed")
        
        self._calculate_bed_coverage(normalized_peaks, alignment_bed, alignment_coverage)
        self._calculate_bed_coverage(normalized_peaks, control_bed, control_coverage)
        
        # Get total mapped reads
        alignment_mapped = self._count_mapped_reads(filtered_bams['alignment'])
        control_mapped = self._count_mapped_reads(filtered_bams['control'])
        
        # Normalize coverage
        coverage_csv = os.path.join(test_dir, "y.counts.csv")
        control_csv = os.path.join(test_dir, "y.control.csv")
        
        self._normalize_coverage(alignment_coverage, alignment_mapped, coverage_csv)
        self._normalize_coverage(control_coverage, control_mapped, control_csv)
        
        return {
            'coverage': coverage_csv,
            'control': control_csv
        }
    
    def _bam_to_bed(self, bam_file: str, bed_file: str):
        """Convert BAM to BED format and sort"""
        # Convert to BED
        cmd = [self.config.bedtools_path, "bamtobed", "-i", bam_file]
        
        with open(bed_file, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)
        
        # Sort BED file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            subprocess.run(['sort', '-k1,1', '-k2,2n', bed_file], stdout=tmp, check=True)
            tmp_name = tmp.name
        
        shutil.move(tmp_name, bed_file)
        logger.info(f"BAM to BED: {bam_file} -> {bed_file}")
    
    def _calculate_bed_coverage(self, peaks_bed: str, reads_bed: str, output_bed: str):
        """Calculate coverage using bedtools"""
        cmd = [
            self.config.bedtools_path, "coverage",
            "-a", peaks_bed,
            "-b", reads_bed,
            "-sorted",
            "-counts"
        ]
        
        with open(output_bed, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)
        
        logger.info(f"Coverage calculated: {output_bed}")
    
    def _count_mapped_reads(self, bam_file: str) -> int:
        """Count total mapped reads in BAM file"""
        cmd = [self.config.samtools_path, "view", "-c", "-F", "260", bam_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    
    def _normalize_coverage(self, coverage_bed: str, total_reads: int, output_csv: str):
        """Normalize coverage by total mapped reads (RPM)"""
        df = pd.read_csv(coverage_bed, sep='\t', header=None)
        # Assuming coverage is in the last column (index 10)
        normalized_coverage = df.iloc[:, -1] * 1000000 / total_reads
        normalized_coverage.to_csv(output_csv, index=False, header=False)
        logger.info(f"Normalized coverage: {coverage_bed} -> {output_csv}")

def main():
    """Main function to run the pipeline"""
    if len(sys.argv) < 2:
        print("Usage: python chipseq_processor.py <experiment_dir> [target_directory]")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    target_directory = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create configuration
    config = ProcessingConfig()
    
    # Create processor
    processor = ChIPSeqProcessor(config)
    
    # Process experiment
    try:
        results = processor.process_experiment(experiment_dir, target_directory)
        if results:
            print("Processing completed successfully!")
            print(f"Output files: {results}")
        else:
            print("No processing performed (experiment already exists or not target)")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 