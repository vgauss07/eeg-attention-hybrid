"""
Download and preprocess BCI Competition IV dataset 2a.

Dataset: 9 subjects, 22 EEG channels, 4 motor imagery classes
  (left hand, right hand, feet, tongue), 250 Hz sampling rate.

Files expected (GDF format):
  A01T.gdf ... A09T.gdf  (training sessions)
  A01E.gdf ... A09E.gdf  (evaluation sessions)
  A01E.mat ... A09E.mat  (true labels for evaluation)

NOTE: The BCI Competition IV 2a data cannot be auto-downloaded due to
license restrictions. You must manually download from:
  https://www.bbci.de/competition/iv/#dataset2a
and place the .gdf and label .mat files in --output_dir.

This script handles preprocessing once files are in place.
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


def preprocess_subject(
    subject_id: int,
    raw_dir: str,
    output_dir: str,
    sfreq: int = 250,
    bandpass: tuple = (4.0, 38.0),
    tmin: float = 0.5,
    tmax: float = 4.5,
):
    """Load, filter, epoch, and save a single subject's data."""
    import mne
    import scipy.io as sio

    mne.set_log_level("ERROR")

    sub = f"A{subject_id:02d}"
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for session, suffix in [("train", "T"), ("test", "E")]:
        gdf_path = os.path.join(raw_dir, f"{sub}{suffix}.gdf")
        if not os.path.exists(gdf_path):
            print(f"  [SKIP] {gdf_path} not found.")
            continue

        # Load raw GDF
        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

        # Pick only the 22 EEG channels (drop EOG)
        eeg_ch_names = raw.ch_names[:22]
        raw.pick_channels(eeg_ch_names)

        # Bandpass filter
        raw.filter(bandpass[0], bandpass[1], fir_design="firwin", verbose=False)

        # Extract events
        events, event_id_raw = mne.events_from_annotations(raw, verbose=False)

        # Map event codes to 4 MI classes (769=left, 770=right, 771=feet, 772=tongue)
        mi_event_codes = {"769": 0, "770": 1, "771": 2, "772": 3}
        event_id = {}
        for code_str, class_idx in mi_event_codes.items():
            if code_str in event_id_raw:
                event_id[code_str] = event_id_raw[code_str]

        if not event_id:
            # Try numeric codes
            code_map = {7: 0, 8: 1, 9: 2, 10: 3}
            for code_int, class_idx in code_map.items():
                for k, v in event_id_raw.items():
                    if v == code_int:
                        event_id[k] = v

        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )

        X = epochs.get_data()  # (n_trials, n_channels, n_timepoints)

        # Labels from epochs events
        labels_raw = epochs.events[:, -1]
        # Re-map to 0-3
        unique_codes = sorted(event_id.values())
        code_to_class = {c: i for i, c in enumerate(unique_codes)}
        y = np.array([code_to_class[l] for l in labels_raw])

        # For evaluation session, true labels come from .mat file
        if suffix == "E":
            mat_path = os.path.join(raw_dir, f"{sub}{suffix}.mat")
            if os.path.exists(mat_path):
                mat = sio.loadmat(mat_path)
                # Labels stored under 'classlabel'
                if "classlabel" in mat:
                    y_true = mat["classlabel"].flatten() - 1  # 1-indexed → 0-indexed
                    # Truncate/pad to match epoch count
                    n = min(len(y_true), len(y))
                    y[:n] = y_true[:n]

        results[session] = {"X": X.astype(np.float32), "y": y.astype(np.int64)}
        print(
            f"  Subject {subject_id} {session}: "
            f"X={X.shape}, y={y.shape}, classes={np.unique(y)}"
        )

    # Save
    out_path = os.path.join(output_dir, f"subject_{subject_id:02d}.npz")
    save_dict = {}
    for session, data in results.items():
        save_dict[f"X_{session}"] = data["X"]
        save_dict[f"y_{session}"] = data["y"]
    np.savez_compressed(out_path, **save_dict)
    print(f"  Saved → {out_path}")


def download_and_preprocess(
    output_dir: str = "./data/raw",
    processed_dir: str = "./data/processed",
    subjects: list = None,
    **kwargs,
):
    """Process all subjects."""
    if subjects is None:
        subjects = list(range(1, 10))

    print("=" * 60)
    print("BCI Competition IV 2a — Preprocessing")
    print("=" * 60)
    print(f"Raw data dir  : {output_dir}")
    print(f"Output dir    : {processed_dir}")
    print(f"Subjects      : {subjects}")
    print()

    # Check if raw files exist
    sample_file = os.path.join(output_dir, "A01T.gdf")
    if not os.path.exists(sample_file):
        print(
            "⚠  Raw GDF files not found!\n"
            "   Please download from:\n"
            "   https://www.bbci.de/competition/iv/#dataset2a\n"
            f"   and place .gdf + .mat files in: {output_dir}\n"
        )
        print("Creating synthetic demo data for pipeline testing...")
        _create_synthetic_data(processed_dir, subjects, **kwargs)
        return

    for sid in subjects:
        print(f"\nProcessing Subject {sid}...")
        preprocess_subject(
            subject_id=sid,
            raw_dir=output_dir,
            output_dir=processed_dir,
            **kwargs,
        )

    print("\n✓ All subjects processed.")


def _create_synthetic_data(
    output_dir: str,
    subjects: list,
    n_channels: int = 22,
    n_timepoints: int = 1000,
    n_train: int = 288,
    n_test: int = 288,
    n_classes: int = 4,
    **kwargs,
):
    """Create synthetic data for pipeline testing when real data unavailable."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    for sid in subjects:
        X_train = rng.standard_normal((n_train, n_channels, n_timepoints)).astype(
            np.float32
        )
        y_train = rng.integers(0, n_classes, size=n_train).astype(np.int64)
        X_test = rng.standard_normal((n_test, n_channels, n_timepoints)).astype(
            np.float32
        )
        y_test = rng.integers(0, n_classes, size=n_test).astype(np.int64)

        out_path = os.path.join(output_dir, f"subject_{sid:02d}.npz")
        np.savez_compressed(
            out_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        print(f"  Synthetic subject {sid}: train={n_train}, test={n_test} → {out_path}")

    print("\n✓ Synthetic data created (replace with real data for publication).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BCI-IV-2a")
    parser.add_argument("--output_dir", type=str, default="./data/raw")
    parser.add_argument("--processed_dir", type=str, default="./data/processed")
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=list(range(1, 10))
    )
    parser.add_argument("--bandpass_low", type=float, default=4.0)
    parser.add_argument("--bandpass_high", type=float, default=38.0)
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=4.5)
    args = parser.parse_args()

    download_and_preprocess(
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        subjects=args.subjects,
        bandpass=(args.bandpass_low, args.bandpass_high),
        tmin=args.tmin,
        tmax=args.tmax,
    )
