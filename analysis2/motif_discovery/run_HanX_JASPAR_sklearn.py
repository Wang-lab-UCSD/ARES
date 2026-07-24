#!/usr/bin/env python3

"""
HanX JASPAR Sklearn Baseline Script

Two-stage workflow:
  --mode scan  (GPU)  Scans JASPAR motifs for all sequences, caches to
                      {TF_dir}/HanX_JASPAR_sklearn/motif_scores_all.npy
  --mode fit   (CPU)  Loads cached scores, fits RF/Ridge/Lasso/ElasticNet,
                      saves models and appends results to output CSV

Output format matches run_HanX_JASPAR_performance.py for easy comparison.
"""

import sys
import random
import numpy as np
import os
import argparse



# ── scan mode helpers ────────────────────────────────────────────────────────

def scan_all_sequences(meme_path, X, batch_size=10000):
    """Scan all sequences (forward + RC) in batches to avoid OOM on large datasets.

    Uses the self-contained NumPy PWM scanner (motif_scan.scan_scores), which is
    numerically identical to the original TensorFlow conv1d-then-max but has no
    TensorFlow / Keras dependency. Each sequence is scored on both strands and the
    stronger strand kept.
    """
    from motif_scan import parse_meme_file, scan_scores

    motifs = parse_meme_file(meme_path)
    n = X.shape[0]
    all_scores = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        print(f"  Scanning batch {start}-{end} / {n}...", flush=True)
        X_batch = X[start:end]
        X_batch_rc = X_batch[:, ::-1, ::-1]          # reverse-complement one-hot
        fwd = scan_scores(X_batch, motifs)
        rc = scan_scores(X_batch_rc, motifs)
        all_scores.append(np.maximum(fwd, rc))

    return np.concatenate(all_scores, axis=0)


def run_scan(TF_dir, jaspar_meme_path, task='regression', BG_DIR=None):
    """GPU stage: scan sequences and cache motif_scores_all.npy."""
    cache_dir  = os.path.join(TF_dir, "HanX_JASPAR_sklearn")
    cache_path = os.path.join(cache_dir, "motif_scores_all.npy")

    X = np.load(f"{TF_dir}/onehot.npy")

    # For classification, concatenate peaks + background before scanning
    if task == 'classification':
        bg_path = BG_DIR if BG_DIR is not None else f"{TF_dir}/HOMER_background/bg_onehot.npy"
        X_bg = np.load(bg_path)
        X = np.concatenate([X, X_bg], axis=0)

    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == X.shape[0]:
            print(f"Cache already exists and shape matches: {cached.shape}. Skipping scan.", flush=True)
            return
        else:
            print(f"Cache shape mismatch ({cached.shape[0]} vs {X.shape[0]}), re-scanning...", flush=True)

    print(f"Scanning JASPAR motifs for {X.shape[0]} sequences...", flush=True)
    motif_scores = scan_all_sequences(jaspar_meme_path, X)
    print(f"Motif score matrix: {motif_scores.shape}", flush=True)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, motif_scores)
    print(f"Cached motif scores to {cache_path}", flush=True)


# ── fit mode helpers ─────────────────────────────────────────────────────────

def train_test_split_py(X, y, test_size):
    data = list(zip(X, y))
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    X_train, y_train = zip(*data[:split_index])
    X_test, y_test = zip(*data[split_index:])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def build_models(task):
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression

    if task == 'regression':
        return {
            'RandomForest': RandomForestRegressor(
                n_estimators=300, max_depth=None, min_samples_leaf=1,
                max_features=0.3, n_jobs=-1, random_state=42
            ),
            'Ridge': Ridge(alpha=100.0),
            'Lasso': Lasso(alpha=0.0001, max_iter=10000),
            'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=10000),
        }
    else:
        return {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=None, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                penalty='l2', C=1.0, max_iter=10000, solver='lbfgs'
            ),
        }


def evaluate(model, X_test, y_test, task):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score, average_precision_score

    if task == 'regression':
        y_pred = model.predict(X_test)
        return {
            'Pearson': pearsonr(y_test, y_pred)[0],
            'Spearman': spearmanr(y_test, y_pred)[0],
        }
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        return {
            'AUC': roc_auc_score(y_test, y_prob),
            'AUPRC': average_precision_score(y_test, y_prob),
        }


def run_fit(cellline, TF_dir, TF, jaspar_meme_path, output_path,
            task='regression', BG_DIR=None, top_signal_enriched_peak=None, seed=1337):
    """CPU stage: load cached motif scores, fit sklearn models, save results."""
    import fcntl
    import joblib
    import pandas as pd
    from motif_scan import parse_meme_file, motif_id_to_tf_name

    cache_path = os.path.join(TF_dir, "HanX_JASPAR_sklearn", "motif_scores_all.npy")
    if not os.path.exists(cache_path):
        print(f"Error: {cache_path} not found. Run --mode scan first.", flush=True)
        sys.exit(1)

    X_motif = np.load(cache_path)

    # Load labels
    if task == 'regression':
        y = np.load(f"{TF_dir}/signal.npy")
        y = np.log1p(y)
    elif task == 'classification':
        y_peak = np.ones(len(np.load(f"{TF_dir}/onehot.npy")), dtype=np.float32)
        bg_path = BG_DIR if BG_DIR is not None else f"{TF_dir}/HOMER_background/bg_onehot.npy"
        y_bg = np.zeros(len(np.load(bg_path)), dtype=np.float32)
        y = np.concatenate([y_peak, y_bg], axis=0)
    else:
        print(f"Error: Invalid task: {task}", flush=True)
        sys.exit(1)

    if X_motif.shape[0] != y.shape[0]:
        print(f"Error: cache has {X_motif.shape[0]} rows but labels have {y.shape[0]}. Re-run scan.", flush=True)
        sys.exit(1)

    if top_signal_enriched_peak is not None:
        top_signal_enriched_peak = int(top_signal_enriched_peak)
        sorted_indices = np.argsort(y)[::-1][:top_signal_enriched_peak]
        y = y[sorted_indices]
        X_motif = X_motif[sorted_indices]
        print(f"Using top {len(y)} signal-enriched peaks", flush=True)

    print(f"Loaded motif scores: {X_motif.shape}", flush=True)

    np.random.seed(seed)
    random.seed(seed)

    # Same 70:10:20 split as HanX-JASPAR
    XTrain, XTest, YTrain, YTest = train_test_split_py(X_motif, y, test_size=0.1)
    validation_ratio = 2.0 / 9.0
    XTrain, XVal, YTrain, YVal = train_test_split_py(XTrain, YTrain, test_size=validation_ratio)
    print(f"Data split - Train: {len(XTrain)}, Test: {len(XTest)}, Validation: {len(XVal)}", flush=True)

    models = build_models(task)
    results_rows = []
    model_dir = os.path.join(TF_dir, "HanX_JASPAR_sklearn")

    n_train, n_features = XTrain.shape
    if n_train < n_features:
        print(f"Warning: n_train ({n_train}) < n_features ({n_features}). Linear models may be unstable.", flush=True)

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...", flush=True)
        try:
            model.fit(XTrain, YTrain)
            metrics = evaluate(model, XTest, YTest, task)
        except Exception as e:
            print(f"  SKIPPED {model_name}: {e}", flush=True)
            continue
        print(f"  {model_name} results: {metrics}", flush=True)

        row = {'Cellline': cellline, 'Experiment': TF, 'Model': model_name,
               'N_samples': len(X_motif), 'N_train': n_train}
        row.update(metrics)
        results_rows.append(row)

        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"  Saved model to {model_path}", flush=True)

        # Print top motifs
        motifs    = parse_meme_file(jaspar_meme_path)
        motif_ids = list(motifs.keys())

        if 'RandomForest' in model_name:
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            print(f"  Top 20 motifs by importance:", flush=True)
            for rank, idx in enumerate(top_idx, 1):
                tf_name = motif_id_to_tf_name(motif_ids[idx], jaspar_meme_path)
                print(f"    {rank:2d}. {motif_ids[idx]} ({tf_name}): {importances[idx]:.4f}")

        if model_name in ('Ridge', 'Lasso', 'ElasticNet'):
            coefs   = model.coef_
            nonzero = np.count_nonzero(coefs)
            print(f"  Non-zero coefficients: {nonzero}/{len(coefs)}", flush=True)
            top_idx = np.argsort(np.abs(coefs))[::-1][:20]
            print(f"  Top 20 motifs by |coefficient|:", flush=True)
            for rank, idx in enumerate(top_idx, 1):
                if coefs[idx] == 0:
                    break
                tf_name = motif_id_to_tf_name(motif_ids[idx], jaspar_meme_path)
                print(f"    {rank:2d}. {motif_ids[idx]} ({tf_name}): {coefs[idx]:.6f}")

    # Save results with file lock (safe for concurrent array jobs)
    if output_path is not None:
        print(f"\nSaving results to: {output_path}", flush=True)
        new_df    = pd.DataFrame(results_rows)
        lock_path = output_path + ".lock"
        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                existing_df  = pd.read_csv(output_path)
                combined_df  = pd.concat([existing_df, new_df], ignore_index=True)
            except FileNotFoundError:
                combined_df = new_df
            combined_df.to_csv(output_path, index=False)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        print(f"Results saved to {output_path}", flush=True)


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HanX JASPAR sklearn baselines (two-stage)')
    parser.add_argument('--mode',     required=True, choices=['scan', 'fit'], help='scan: GPU motif scan; fit: CPU model fitting')
    parser.add_argument('--cellline', required=False, help='cell line name (fit mode only)')
    parser.add_argument('--TF_dir',   required=True,  help='path to TF directory')
    parser.add_argument('--TF',       required=False, help='transcription factor name (fit mode only)')
    parser.add_argument('--JASPAR_meme_path', required=True, help='path to JASPAR meme file')
    parser.add_argument('--output_path', required=False, help='path to output CSV (fit mode only)')
    parser.add_argument('--task',     required=False, default='regression', choices=['regression', 'classification'])
    parser.add_argument('--BG_DIR',   required=False, help='background directory (classification only)')
    parser.add_argument('--top_signal_enriched_peak', required=False, help='number of top signal-enriched peaks')
    parser.add_argument('--seed',     required=False, default=1337, type=int)
    args = parser.parse_args()

    if args.mode == 'scan':
        run_scan(args.TF_dir, args.JASPAR_meme_path, args.task, args.BG_DIR)
    else:
        for required in ('cellline', 'TF', 'output_path'):
            if getattr(args, required) is None:
                parser.error(f"--{required} is required for --mode fit")
        run_fit(args.cellline, args.TF_dir, args.TF, args.JASPAR_meme_path,
                args.output_path, args.task, args.BG_DIR,
                args.top_signal_enriched_peak, args.seed)
