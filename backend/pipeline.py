# backend/pipeline.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew, entropy as scipy_entropy

# -----------------------------
# Core FFT helpers
# -----------------------------
def compute_fft(signal, sampling_rate):
    N = len(signal)
    T = 1.0 / sampling_rate
    xf = fftfreq(N, T)[:N // 2]
    yf = fft(signal)
    amplitude = 2.0 / N * np.abs(yf[:N // 2])
    return xf, amplitude

def zoom_fft(xf, amp, f_min=200, f_max=1000):
    mask = (xf >= f_min) & (xf <= f_max)
    return xf[mask], amp[mask]

# -----------------------------
# Feature extraction per segment
# -----------------------------
def extract_fft_features(signal, sampling_rate):
    xf, amp = compute_fft(signal, sampling_rate)
    power_spectrum = amp ** 2 + 1e-12
    prob_dist = power_spectrum / np.sum(power_spectrum)

    rms = np.sqrt(np.mean(signal ** 2))
    crest_factor = np.max(np.abs(signal)) / rms if rms != 0 else 0
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    iqr = np.percentile(signal, 75) - np.percentile(signal, 25)

    return {
        'mean_fft': float(np.mean(amp)),
        'max_fft': float(np.max(amp)),
        'std_fft': float(np.std(amp)),
        'energy': float(np.sum(power_spectrum)),
        'kurtosis': float(kurtosis(amp)),
        'skewness': float(skew(amp)),
        'entropy': float(scipy_entropy(prob_dist)),
        'peak_idx': float(xf[np.argmax(amp)] if len(xf) else 0.0),
        'mean': float(np.mean(signal)),
        'std': float(np.std(signal)),
        'min': float(np.min(signal)),
        'max': float(np.max(signal)),
        'range': float(np.ptp(signal)),
        'median': float(np.median(signal)),
        'rms': float(rms),
        'crest_factor': float(crest_factor),
        'zero_crossings': int(len(zero_crossings)),
        'ptp': float(np.ptp(signal)),
        'iqr': float(iqr)
    }

# -----------------------------
# Optional plotting for one signal (returns a Matplotlib Figure)
# -----------------------------
def make_time_fft_fig(signal, sampling_rate, filename, component="FZ", f_min=200, f_max=1000):
    N = len(signal)
    time_axis = np.arange(N) / sampling_rate
    xf, amp = compute_fft(signal, sampling_rate)
    xf_zoom, amp_zoom = zoom_fft(xf, amp, f_min=f_min, f_max=f_max)

    max_amp = np.max(amp_zoom) if len(amp_zoom) and np.max(amp_zoom) != 0 else 1.0
    amp_zoom_normalized = amp_zoom / max_amp

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time domain
    axes[0].plot(time_axis, signal)
    axes[0].set_title(f"Time Domain - {component} - {filename}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Zoomed FFT
    axes[1].plot(xf_zoom, amp_zoom_normalized)
    axes[1].set_title(f"Zoomed FFT ({f_min}–{f_max} Hz) - {component} - {filename}")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Normalized Amplitude")
    axes[1].set_ylim([0, 1])
    axes[1].grid(True)
    if len(xf_zoom):
        axes[1].set_xticks(np.arange(max(f_min, int(xf_zoom.min()//50)*50), f_max+1, 150))

    fig.tight_layout()
    return fig

# -----------------------------
# Batch: build features from a folder of CSV/XLSX files
# Expects columns: FZ, DOC, SPEED, FEED, CHATTER
# -----------------------------
def extract_features_from_folder(
    folder_path: str,
    segment_size: int = 1024,
    step_size: int = 256,
    sampling_rate: int = 10005,
    preview_plots: bool = False,
    preview_limit: int = 2
) -> pd.DataFrame:
    all_features = []
    plot_figs = []

    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".xlsx", ".csv"))]
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.lower().endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # keep numeric FZ rows
        df = df[pd.to_numeric(df['FZ'], errors='coerce').notnull()].reset_index(drop=True)

        try:
            fz = df['FZ'].astype(float).values
            doc = float(df['DOC'].iloc[0])
            speed = float(df['SPEED'].iloc[0])
            feed = float(df['FEED'].iloc[0])
            chatter = int(df['CHATTER'].iloc[0])

            if preview_plots and len(plot_figs) < preview_limit:
                fig = make_time_fft_fig(fz, sampling_rate, filename, component="FZ")
                plot_figs.append(fig)

            # sliding windows
            for i in range(0, len(fz) - segment_size + 1, step_size):
                fz_seg = fz[i:i + segment_size]
                feats = extract_fft_features(fz_seg, sampling_rate)
                combined = {
                    **{f'FZ_{k}': v for k, v in feats.items()},
                    'DOC': doc,
                    'SPEED': speed,
                    'FEED': feed,
                    'chatter': chatter,
                    'source_file': filename,
                }
                all_features.append(combined)

        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")

    features_df = pd.DataFrame(all_features)
    return features_df, plot_figs

# -----------------------------
# Utilities for ML
# -----------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def scale_features(features_df: pd.DataFrame):
    X = features_df.drop(columns=['chatter', 'source_file'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    y = features_df['chatter'].astype(int).values
    return X_scaled_df, y, scaler

def correlations(X_scaled_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    tmp = X_scaled_df.copy()
    tmp['chatter'] = y
    return tmp.corr()

def select_kbest(X_scaled_df: pd.DataFrame, y: np.ndarray, k: int = 6):
    selector = SelectKBest(score_func=f_classif, k=min(k, X_scaled_df.shape[1]))
    X_selected = selector.fit_transform(X_scaled_df, y)
    cols = X_scaled_df.columns[selector.get_support()]
    X_sel_df = pd.DataFrame(X_selected, columns=cols)
    return X_sel_df, cols, selector

def train_random_forest(X, y, test_size=0.2, random_state=42, params=None):
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=test_size, random_state=random_state)
    model = RandomForestClassifier(random_state=random_state, **(params or {}))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, digits=3),
        "train_acc": float(model.score(X_train, y_train)),
        "test_acc": float(model.score(X_test, y_test)),
    }
    return model, (X_train, X_test, y_train, y_test), metrics

def train_xgboost(X, y, test_size=0.2, random_state=42, params=None):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError("xgboost is not installed. Add it to requirements.txt and pip install.") from e

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=test_size, random_state=random_state)
    default = dict(use_label_encoder=False, eval_metric='logloss',
                   n_estimators=100, max_depth=5, learning_rate=0.1,
                   random_state=random_state)
    default.update(params or {})
    model = XGBClassifier(**default)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    metrics = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, digits=3),
        "train_acc": float(model.score(X_train, y_train)),
        "test_acc": float(model.score(X_test, y_test)),
    }
    return model, (X_train, X_test, y_train, y_test), metrics
