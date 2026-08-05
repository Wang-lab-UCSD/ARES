#!/usr/bin/env python
"""Self-contained motif scoring for the partner-motif discovery pipeline.

This replaces the three things the pipeline used to import from the HanX package
(tools.parse_meme_file, tools.extract_TF_name_from_matrix_id and
networks.MotifScanHanX.scan_with_motifs_conv). The scan is a plain PWM convolution
followed by a per-sequence max, so it is reimplemented here in NumPy with no
TensorFlow / Keras dependency and gives numerically identical scores to the
original TensorFlow version.

The one-hot encoding is (A, C, G, T) on the last axis; the MEME letter-probability
matrix is read in that column order, which is JASPAR's convention.
"""
import numpy as np


def parse_meme_file(file_path):
    """MEME file -> {motif_id: PWM array of shape (motif_length, 4)}."""
    motifs, current, rows = {}, None, []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('MOTIF'):
                if current and rows:
                    motifs[current] = np.array(rows, dtype=float)
                    rows = []
                current = line.split()[1]
            elif line.startswith('letter-probability matrix'):
                rows = []
            elif line and current and (line[0].isdigit() or line[0] == '.'):
                rows.append([float(x) for x in line.split()])
        if current and rows:
            motifs[current] = np.array(rows, dtype=float)
    return motifs


def motif_id_to_tf_name(matrix_id, meme_path):
    """JASPAR matrix id (e.g. 'MA0138.2') -> TF name (e.g. 'REST')."""
    for line in open(meme_path):
        if line.startswith('MOTIF'):
            parts = line.strip().split()
            if parts[1] == matrix_id:
                return ' '.join(parts[2:])
    raise KeyError(f'{matrix_id} not found in {meme_path}')


def scan_scores(sequences, motifs):
    """Best-match PWM score of every motif against every sequence.

    sequences : (N, L, 4) one-hot float array, channels ordered A, C, G, T.
    motifs    : dict of {motif_id: (w, 4) PWM}, as returned by parse_meme_file.
    returns   : (N, M) array, score[i, j] = max over all start positions of the
                dot product between sequence i and the background-centred PWM j.

    Identical to the original tf.nn.conv1d(..., padding='VALID') then
    tf.reduce_max: each PWM is centred by subtracting the 0.25 uniform
    background, convolved along the sequence, and the strongest window kept.
    """
    N = len(sequences)
    seqs = np.asarray(sequences, dtype=np.float32)
    out = np.empty((N, len(motifs)), dtype=np.float32)
    for j, pwm in enumerate(motifs.values()):
        f = pwm.astype(np.float32) - 0.25          # background-centred filter, (w, 4)
        w = f.shape[0]
        if w > seqs.shape[1]:                      # motif longer than the sequence
            out[:, j] = 0.0
            continue
        # sliding windows: (N, L-w+1, w, 4), scored against the filter
        best = np.full(N, -np.inf, dtype=np.float32)
        for s in range(seqs.shape[1] - w + 1):
            score = np.tensordot(seqs[:, s:s + w, :], f, axes=([1, 2], [0, 1]))
            best = np.maximum(best, score)
        out[:, j] = best
    return out
