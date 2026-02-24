import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Optional

#!/usr/bin/env python3

# Check and install required dependencies
def check_and_install_dependencies():
    """Check for required packages and install if missing."""
    required_packages = {
        'pandas': 'pandas',
        'openpyxl': 'openpyxl',  # For Excel support
        'matplotlib': 'matplotlib',  # For plotting
        'scipy': 'scipy',  # For WAV file analysis
        'numpy': 'numpy',  # For WAV file analysis
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("=" * 70)
        print("Missing required packages detected!")
        print(f"Missing: {', '.join(missing_packages)}")
        print("=" * 70)
        
        install_choice = input("\nInstall missing packages automatically? (y/n): ").strip().lower()
        
        if install_choice == 'y':
            print("\nInstalling missing packages...")
            for package in missing_packages:
                try:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"✓ {package} installed successfully")
                except subprocess.CalledProcessError:
                    print(f"✗ Failed to install {package}")
                    print(f"Please install manually: pip install {package}")
                    sys.exit(1)
            print("\n✓ All packages installed successfully!\n")
        else:
            print("\nCannot proceed without required packages.")
            print("Please install manually:")
            for package in missing_packages:
                print(f"  pip install {package}")
            sys.exit(1)

# Run dependency check
check_and_install_dependencies()    

# Import pandas after checking dependencies
import pandas as pd

# Plotting (Tkinter + Matplotlib)
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



LOG_COLUMNS = [
    "log_id",
    "timestamp",
    "target_msr",
    "threshold",
    "current_msr_value",
    "details_path",
]

OUTLIER_COLUMNS = [
    "log_id",
    "level",
    "measurement",
    "noise",
]


CORE_BIT_MAP = {"pcore": "0", "ecore": "1"}
HYSTERESIS_REGISTER = "0x608"
HYSTERESIS_CONTROL_REGISTER = "0x607"

FREQUENCY_LABELS = [
    "12.5 Hz",
    "16 Hz",
    "20 Hz",
    "25 Hz",
    "31.5 Hz",
    "40 Hz",
    "50 Hz",
    "63 Hz",
    "80 Hz",
    "100 Hz",
    "125 Hz",
    "160 Hz",
    "200 Hz",
    "250 Hz",
    "315 Hz",
    "400 Hz",
    "500 Hz",
    "630 Hz",
    "800 Hz",
    "1 kHz",
    "1.25 kHz",
    "1.6 kHz",
    "2 kHz",
    "2.5 kHz",
    "3.15 kHz",
    "4 kHz",
    "5 kHz",
    "6.3 kHz",
    "8 kHz",
    "10 kHz",
    "12.5 kHz",
    "16 kHz",
    "20 kHz",
]


def parse_frequency(value: object) -> Optional[float]:
    """Parse frequency labels like '12.5 Hz', '1 kHz', ignoring A/Z."""
    text = str(value).strip()
    if not text:
        return None
    if text.upper() in {"A", "Z"}:
        return None
    normalized = text.replace(" ", "").lower()
    try:
        if normalized.endswith("khz"):
            return float(normalized[:-3]) * 1000
        if normalized.endswith("hz"):
            return float(normalized[:-2])
        return float(normalized)
    except ValueError:
        return None


class PlotWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Noise vs Frequency (Iterative)")

        self.mode_var = tk.StringVar(value="current_prev")
        self.data_runs: list[dict[str, object]] = []
        self.frequency_index = {
            label.lower().replace(" ", ""): idx for idx, label in enumerate(FREQUENCY_LABELS)
        }

        controls = ttk.LabelFrame(self.root, text="View")
        controls.pack(fill=tk.X, padx=10, pady=10)

        ttk.Radiobutton(
            controls,
            text="Current vs Previous",
            value="current_prev",
            variable=self.mode_var,
            command=self.update_plot,
        ).pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Radiobutton(
            controls,
            text="Last 5 runs",
            value="last_5",
            variable=self.mode_var,
            command=self.update_plot,
        ).pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Radiobutton(
            controls,
            text="All runs",
            value="all",
            variable=self.mode_var,
            command=self.update_plot,
        ).pack(side=tk.LEFT, padx=10, pady=5)

        self.figure = Figure(figsize=(9, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def add_run(self, run_label: str, df: pd.DataFrame, threshold: float) -> None:
        series = self._extract_series(df)
        if not series:
            return
        self.data_runs.append({"label": run_label, "series": series, "threshold": threshold})
        self.update_plot()

    def _extract_series(self, df: pd.DataFrame) -> list[tuple[int, str, float]]:
        if "level" not in df.columns:
            return []
        numeric_cols = [
            col for col in df.columns
            if col != "level" and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors='coerce'))
        ]
        if not numeric_cols:
            return []
        noise_col = numeric_cols[0]
        series: list[tuple[int, str, float]] = []
        for _, row in df.iterrows():
            label = str(row["level"]).strip()
            normalized = label.lower().replace(" ", "")
            if normalized in {"a", "z"}:
                continue
            idx = self.frequency_index.get(normalized)
            if idx is None:
                continue
            noise = row[noise_col]
            if pd.notna(noise):
                series.append((idx, label, float(noise)))
        series.sort(key=lambda item: item[0])
        return series

    def _select_runs(self) -> list[dict[str, object]]:
        if not self.data_runs:
            return []
        mode = self.mode_var.get()
        if mode == "current_prev":
            return self.data_runs[-2:] if len(self.data_runs) >= 2 else self.data_runs[-1:]
        if mode == "last_5":
            return self.data_runs[-5:]
        return self.data_runs

    def update_plot(self) -> None:
        self.ax.clear()
        selected_runs = self._select_runs()
        for run in selected_runs:
            label = run["label"]
            series = run["series"]
            x_vals = [item[0] for item in series]
            y_vals = [item[2] for item in series]
            self.ax.plot(x_vals, y_vals, marker="o", label=str(label))

        if selected_runs:
            latest_threshold = selected_runs[-1].get("threshold")
            if isinstance(latest_threshold, (int, float)):
                self.ax.axhline(
                    y=float(latest_threshold),
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Threshold ({latest_threshold})",
                )

        self.ax.set_xlabel("Frequency")
        self.ax.set_ylabel("Noise (dB)")
        self.ax.set_ylim(-25, 15)
        self.ax.set_xticks(range(len(FREQUENCY_LABELS)))
        self.ax.set_xticklabels(FREQUENCY_LABELS, rotation=45, ha="right")
        if selected_runs:
            self.ax.legend()
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.canvas.draw()
        self.root.update_idletasks()
        self.root.update()


# ─────────────────────────────────────────────────────────────────────────────
# WAV → DataFrame conversion (modular, reused by load_csv)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import scipy.io.wavfile
from scipy.signal import butter, filtfilt

WAV2CSV_DIR = Path(__file__).parent / "wav2csv"

def _load_audio(file_path: Path):
    """Read WAV file and return (sample_rate, mono float32 audio)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate, audio_data = scipy.io.wavfile.read(str(file_path))
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    return sample_rate, audio_data

def _get_third_octave_frequencies() -> np.ndarray:
    return np.array([
        12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
        250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
        3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
    ])

def _third_octave_filter(audio_data: np.ndarray, sample_rate: int, center_freq: float) -> np.ndarray:
    factor = 2 ** (1 / 6)
    f_lower = center_freq / factor
    f_upper = center_freq * factor
    nyquist = sample_rate / 2
    low = f_lower / nyquist
    high = f_upper / nyquist
    if low <= 0:
        low = 1e-4
    if high >= 1.0:
        return np.zeros_like(audio_data)
    try:
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, audio_data)
    except Exception:
        return np.zeros_like(audio_data)

def _apply_a_weighting(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    fft_data = np.fft.fft(audio_data)
    freqs = np.abs(np.fft.fftfreq(len(audio_data), 1 / sample_rate))
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = 1.9997
    freqs = np.where(freqs == 0, 1e-10, freqs)
    f2_sq = freqs ** 2
    numerator = A1000 * (f4 ** 2) * (f2_sq ** 2)
    denominator = ((f2_sq + f1 ** 2) *
                   np.sqrt((f2_sq + f2 ** 2) * (f2_sq + f3 ** 2)) *
                   (f2_sq + f4 ** 2))
    a_response = numerator / denominator
    weighted_fft = fft_data * a_response
    return np.real(np.fft.ifft(weighted_fft))

def _calculate_spl(audio_data: np.ndarray, offset_db: float = 20.0) -> float:
    """Return A-weighted SPL + calibration offset. Returns NaN on silence."""
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms <= 0 or not np.isfinite(rms):
        return float('nan')
    base_spl = 20 * np.log10(rms / 20e-6)
    return base_spl + offset_db

def wav_to_dataframe(wav_path: Path, calibration_offset: float = 20.0) -> pd.DataFrame:
    """
    Convert a WAV file to a DataFrame with columns:
      level            – frequency label matching FREQUENCY_LABELS (e.g. '1 kHz')
      LAeq [dB]        – A-weighted SPL + calibration offset

    Also saves a CSV to wav2csv/<stem>_<timestamp>.csv.
    """
    sample_rate, audio_data = _load_audio(wav_path)
    print(f"  Sample rate : {sample_rate} Hz")
    print(f"  Duration    : {len(audio_data) / sample_rate:.2f} s")

    center_freqs = _get_third_octave_frequencies()
    print(f"  Analysing {len(center_freqs)} third-octave bands...")

    # Use FREQUENCY_LABELS so labels match the plot's frequency_index exactly.
    # FREQUENCY_LABELS has 33 entries (12.5 Hz … 20 kHz), same order as center_freqs.
    freq_labels = FREQUENCY_LABELS[:len(center_freqs)]

    col_name = f"LAeq [dB] - {wav_path.stem}"
    rows = []
    for label, freq in zip(freq_labels, center_freqs):
        filtered = _third_octave_filter(audio_data, sample_rate, freq)
        a_weighted = _apply_a_weighting(filtered, sample_rate)
        if np.all(filtered == 0):
            spl = float('nan')
        else:
            spl = _calculate_spl(a_weighted, offset_db=calibration_offset)
        rows.append({"level": label, col_name: spl})

    df = pd.DataFrame(rows)

    # Save CSV to wav2csv/
    WAV2CSV_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = WAV2CSV_DIR / f"{wav_path.stem}_{timestamp}.csv"
    df.rename(columns={"level": "Frequency"}).to_csv(out_csv, index=False)
    print(f"  ✓ WAV→CSV saved: {out_csv}")

    return df


def load_csv(path: Path) -> pd.DataFrame:
    try:
        # WAV file — route through audio analysis pipeline
        if path.suffix.lower() == '.wav':
            print(f"  Detected WAV file — running audio analysis...")
            return wav_to_dataframe(path)

        # Check file extension
        if path.suffix.lower() in ['.xlsx', '.xls']:
            # Load Excel file
            df = pd.read_excel(path)
            # Check if first row contains headers (common in acoustic data)
            if df.iloc[0, 0] == 'Frequency' or (isinstance(df.iloc[0, 0], str) and 'frequency' in df.iloc[0, 0].lower()):
                # Use first row as headers
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
        elif path.suffix.lower() == '.csv':
            # Load CSV file
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv, .xlsx, .xls, or .wav")
        
        # Check for the acoustic data format with Frequency column
        if "Frequency" in df.columns:
            # Rename for consistency
            df = df.rename(columns={"Frequency": "level"})
        elif "level" not in df.columns:
            raise ValueError("File must have 'Frequency' or 'level' column.")
        
        # Convert numeric columns to proper numeric type
        for col in df.columns:
            if col != 'level':
                converted = pd.to_numeric(df[col], errors='coerce')
                df[col] = converted.where(converted.notna(), df[col])
        
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading file: {e}")


def summarize_outliers(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Find all values above threshold across all measurement columns."""
    outliers = []
    
    # Get the level column and all numeric columns (measurements)
    level_col = "level"
    measurement_cols = [col for col in df.columns if col != level_col and df[col].dtype in ['float64', 'int64']]
    
    # Check each row
    for idx, row in df.iterrows():
        level = row[level_col]
        # Check each measurement column
        for col in measurement_cols:
            value = row[col]
            # Filter values above threshold
            if pd.notna(value) and value > threshold:
                outliers.append({
                    'level': level,
                    'measurement': col,
                    'noise': value
                })
    
    return pd.DataFrame(outliers)


def read_msr(register: str) -> str:
    try:
        result = subprocess.run(
            ["MSR_Register.exe", "read", register],
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception, check output manually
        )
        output = result.stdout.strip()
        
        # Check for failure messages
        if "failed" in output.lower() or "error" in output.lower():
            # Extract error details
            error_lines = [line for line in output.split('\n') if 'error' in line.lower() or 'failed' in line.lower()]
            return f"ERROR: {'; '.join(error_lines)}" if error_lines else f"ERROR: Read failed"
        
        # Try to extract the actual register value from output
        # Look for patterns like "0x0000000000001400" in the output
        import re
        hex_pattern = r'(0x[0-9A-Fa-f]{16}|0x[0-9A-Fa-f]{8})'
        matches = re.findall(hex_pattern, output)
        if matches:
            # Return the first hex value found (usually the register value)
            return matches[0]
        
        return output
    except FileNotFoundError:
        return "ERROR: MSR_Register.exe not found"
    except Exception as e:
        return f"ERROR: {str(e)}"


def write_msr(register: str, value: str) -> str:
    try:
        result = subprocess.run(
            ["MSR_Register.exe", "write", register, value],
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception, check output manually
        )
        output = result.stdout.strip()
        
        # Check for success flag
        if "Wrmsr SUCCESS" in output:
            return f"✓ SUCCESS: Wrote {value} to {register}"
        
        # Check for failure messages
        if "failed" in output.lower() or "error" in output.lower():
            # Extract error details
            error_lines = [line for line in output.split('\n') if 'error' in line.lower() or 'failed' in line.lower()]
            return f"✗ FAILED: {'; '.join(error_lines)}" if error_lines else f"✗ FAILED: Write operation failed"
        
        # If no clear success/failure indicator, return output
        return f"⚠ UNKNOWN: {output}"
        
    except FileNotFoundError:
        return "ERROR: MSR_Register.exe not found"
    except Exception as e:
        return f"ERROR: {str(e)}"


def append_log(log_path: Path, entry: dict) -> None:
    if not log_path.exists():
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(log_path, index=False)
    df = pd.read_csv(log_path)
    new_row = pd.DataFrame([entry])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(log_path, index=False)


def append_outliers(outlier_path: Path, outliers: pd.DataFrame) -> None:
    if not outlier_path.exists():
        pd.DataFrame(columns=OUTLIER_COLUMNS).to_csv(outlier_path, index=False)
    df = pd.read_csv(outlier_path)
    df = pd.concat([df, outliers], ignore_index=True)
    df.to_csv(outlier_path, index=False)


def get_next_log_id(log_path: Path) -> int:
    if not log_path.exists():
        return 1
    df = pd.read_csv(log_path)
    if df.empty:
        return 1
    return int(df["log_id"].max()) + 1


def save_formatted_log(formatted_log_path: Path, log_id: int, timestamp: str, target_msr: str, 
                       current_msr_value: str, threshold: float, outliers: pd.DataFrame) -> None:
    """Save a formatted log entry with MSR data and outlier table."""
    with open(formatted_log_path, 'a', encoding='utf-8') as f:
        # Write header for this run
        f.write("=" * 80 + "\n")
        f.write(f"Run #{log_id} - {timestamp}\n")
        f.write(f"Target MSR: {target_msr} | Current Value: {current_msr_value}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write("-" * 80 + "\n")
        
        # Write outlier table
        if outliers.empty:
            f.write("No outliers detected above threshold.\n")
        else:
            f.write(f"{'Frequency':<20} {'Measurement':<35} {'Noise (dB)':<15}\n")
            f.write("-" * 80 + "\n")
            for _, row in outliers.iterrows():
                f.write(f"{str(row['level']):<20} {str(row['measurement']):<35} {row['noise']:<15.2f}\n")
        
        f.write("=" * 80 + "\n")
        f.write("\n")


def normalize_hex_value(value: str) -> str:
    if not value:
        raise ValueError("Empty hex value.")
    try:
        normalized = hex(int(value, 0))
    except ValueError:
        raise ValueError(f"Invalid hex value: {value}")
    return normalized


def set_hysteresis_sequence(core: str, hysteresis_value: str) -> dict[str, str]:
    core = core.lower()
    if core not in CORE_BIT_MAP:
        raise ValueError("Core must be 'pcore' or 'ecore'.")
    bit = CORE_BIT_MAP[core]
    get_cmd = f"0x8000{bit}10A"
    set_cmd = f"0x8000{bit}00A"
    value = normalize_hex_value(hysteresis_value)

    sequence = [
        ("read", HYSTERESIS_CONTROL_REGISTER, None),
        ("write", HYSTERESIS_CONTROL_REGISTER, get_cmd),
        ("read", HYSTERESIS_REGISTER, None),
        ("write", HYSTERESIS_REGISTER, value),
        ("write", HYSTERESIS_CONTROL_REGISTER, set_cmd),
        ("write", HYSTERESIS_CONTROL_REGISTER, get_cmd),
        ("read", HYSTERESIS_REGISTER, None),
    ]
    results: dict[str, str] = {}
    for action, reg, val in sequence:
        if action == "read":
            result = read_msr(reg)
        else:
            result = write_msr(reg, val)
        print(f"{action.upper()} {reg} -> {result}")
        results[f"{action}_{reg}"] = result
    return results


def preload_runs(plot_window: PlotWindow, log_path: Path) -> int:
    """Load prior runs from log file to restore plot context."""
    if not log_path.exists():
        return 0
    try:
        df_logs = pd.read_csv(log_path)
    except Exception:
        return 0
    if df_logs.empty:
        return 0

    loaded = 0
    for _, row in df_logs.iterrows():
        details_path = str(row.get("details_path", "")).strip()
        if not details_path:
            continue
        try:
            run_df = load_csv(Path(details_path))
        except Exception:
            continue
        threshold_value = row.get("threshold")
        try:
            threshold_value = float(threshold_value)
        except (TypeError, ValueError):
            threshold_value = 0.0
        label = f"Run {int(row.get('log_id', loaded + 1))}"
        plot_window.add_run(label, run_df, threshold_value)
        loaded += 1
    return loaded


def main():
    parser = argparse.ArgumentParser(description="MSR outlier analyzer")
    parser.add_argument("--log", type=Path, default=Path("msr_log.csv"), help="Log file to update")
    parser.add_argument("--outliers", type=Path, default=Path("msr_outliers.csv"), help="Outliers log file")
    parser.add_argument("--formatted", type=Path, default=Path("msr_formatted_log.txt"), help="Formatted log file")
    args = parser.parse_args()

    print("=== MSR Outlier Analyzer ===\n")

    plot_window = PlotWindow()
    loaded_runs = preload_runs(plot_window, args.log)
    run_counter = loaded_runs + 1
    
    # Step 1: Input - Get CSV path dynamically
    while True:
        csv_path_input = input("Enter file path (.csv, .xlsx, .xls, .wav): ").strip()
        if not csv_path_input:
            print("File path cannot be empty. Please try again.")
            continue
        
        # Remove surrounding quotes if present
        csv_path_input = csv_path_input.strip('"').strip("'")
        csv_path = Path(csv_path_input)
        
        # Step 2: Load file with error handling
        try:
            print(f"Loading file: {csv_path}")
            df = load_csv(csv_path)
            print(f"✓ Successfully loaded {len(df)} rows")
            break
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                print("Exiting.")
                sys.exit(1)
        except (ValueError, Exception) as e:
            print(f"✗ Error: {e}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                print("Exiting.")
                sys.exit(1)
    
    # Step 3: Input - Get threshold dynamically
    while True:
        threshold_input = input("\nEnter threshold value: ").strip()
        try:
            threshold = float(threshold_input)
            break
        except ValueError:
            print("✗ Invalid threshold. Please enter a numeric value.")
    
    print(f"Threshold set to: {threshold}")
    
    # Step 4: Display - Current MSR
    print("\n--- Current MSR Status ---")
    current_msr = read_msr("0x1b")  # default register example
    print(f"Current MSR (0x1b) value: {current_msr}")
    
    # Step 5: Display - Outlier freq vs level
    print("\n--- Outlier Analysis ---")
    summary = summarize_outliers(df, threshold)
    if summary.empty:
        print("No outliers above threshold.")
    else:
        print(f"Found {len(summary)} outlier(s):\n")
        print(f"{'Frequency':<20} {'Measurement':<35} {'Noise (dB)':<15}")
        print("-" * 70)
        for _, row in summary.iterrows():
            print(f"{str(row['level']):<20} {str(row['measurement']):<35} {row['noise']:<15.2f}")

    plot_window.add_run(f"Run {run_counter}", df, threshold)
    
    # Step 6: Input - Hysteresis values for P-core and E-core
    skip_count = 0
    while skip_count < 2:
        print("\n--- Set Hysteresis Values ---")
        pcore_value_input = input("P-core value (decimal ms, or press Enter to skip): ").strip()
        
        if not pcore_value_input:
            skip_count += 1
            if skip_count == 1:
                confirm = input("Press Enter again to exit, or enter P-core value: ").strip()
                if confirm:
                    pcore_value_input = confirm
                    skip_count = 0  # Reset since user provided input
                else:
                    print("Exiting.")
                    sys.exit(0)
            else:
                print("Exiting.")
                sys.exit(0)
        
        if pcore_value_input:
            ecore_value_input = input("E-core value (decimal ms): ").strip()
            
            if not ecore_value_input:
                print("No E-core value provided. Skipping write.")
                continue
            
            try:
                # Convert decimal ms to hex
                pcore_ms = int(pcore_value_input)
                ecore_ms = int(ecore_value_input)
                pcore_hex = hex(pcore_ms)
                ecore_hex = hex(ecore_ms)
                
                print(f"\n=== Setting P-core Hysteresis to {pcore_ms} ms ({pcore_hex}) ===")
                
                # P-core sequence
                # Read MSR 0x607 (control register) - execute but don't display
                result = read_msr("0x607")
                
                print("\n1. Getting current P-core hysteresis value...")
                result = write_msr("0x607", "0x8000010A")
                print(f"   {result}")
                
                print("\n2. Reading MSR 0x608 (current P-core hysteresis)...")
                pcore_old = read_msr("0x608")
                print(f"   Current value: {pcore_old}")
                
                print(f"\n3. Writing new P-core hysteresis value ({pcore_hex})...")
                result = write_msr("0x608", pcore_hex)
                print(f"   {result}")
                
                print("\n4. Setting new P-core hysteresis value...")
                result = write_msr("0x607", "0x8000000A")
                print(f"   {result}")
                
                print("\n5. Verifying new P-core hysteresis value...")
                result = write_msr("0x607", "0x8000010A")
                print(f"   {result}")
                
                print("\n6. Reading MSR 0x608 (verify P-core hysteresis)...")
                pcore_new = read_msr("0x608")
                print(f"   New value: {pcore_new}")
                
                print(f"\n=== Setting E-core Hysteresis to {ecore_ms} ms ({ecore_hex}) ===")
                
                # E-core sequence
                # Read MSR 0x607 (control register) - execute but don't display
                result = read_msr("0x607")
                
                print("\n1. Getting current E-core hysteresis value...")
                result = write_msr("0x607", "0x8001010A")
                print(f"   {result}")
                
                print("\n2. Reading MSR 0x608 (current E-core hysteresis)...")
                ecore_old = read_msr("0x608")
                print(f"   Current value: {ecore_old}")
                
                print(f"\n3. Writing new E-core hysteresis value ({ecore_hex})...")
                result = write_msr("0x608", ecore_hex)
                print(f"   {result}")
                
                print("\n4. Setting new E-core hysteresis value...")
                result = write_msr("0x607", "0x8001000A")
                print(f"   {result}")
                
                print("\n5. Verifying new E-core hysteresis value...")
                result = write_msr("0x607", "0x8001010A")
                print(f"   {result}")
                
                print("\n6. Reading MSR 0x608 (verify E-core hysteresis)...")
                ecore_new = read_msr("0x608")
                print(f"   New value: {ecore_new}")
                
                # Summary
                print("\n" + "=" * 70)
                print("SUMMARY:")
                print(f"P-core: {pcore_old} → {pcore_new} (target: {pcore_hex} / {pcore_ms} ms)")
                print(f"E-core: {ecore_old} → {ecore_new} (target: {ecore_hex} / {ecore_ms} ms)")
                print("=" * 70)
                
                # Use the new values for logging
                target_msr = "0x607/0x608 (Hysteresis)"
                current_msr = f"P-core: {pcore_new}, E-core: {ecore_new}"
                
            except ValueError:
                print("✗ Invalid input. Please enter numeric values for milliseconds.")
                continue
            
            # Get next log ID
            log_id = get_next_log_id(args.log)
            timestamp = datetime.utcnow().isoformat()
            
            # Log the main operation
            entry = {
                "log_id": log_id,
                "timestamp": timestamp,
                "target_msr": target_msr,
                "threshold": threshold,
                "current_msr_value": current_msr,
                "details_path": str(csv_path),
            }
            append_log(args.log, entry)
            print(f"✓ Logged to {args.log} (ID: {log_id})")
            
            # Log outliers to separate table
            outliers_to_save = pd.DataFrame()
            if not summary.empty:
                outliers_to_save = summary.copy()
                outliers_to_save["log_id"] = log_id
                outliers_to_save = outliers_to_save[["log_id", "level", "measurement", "noise"]]
                append_outliers(args.outliers, outliers_to_save)
                print(f"✓ Logged {len(outliers_to_save)} outliers to {args.outliers}")
            
            # Save formatted log for better viewing
            save_formatted_log(args.formatted, log_id, timestamp, target_msr, 
                             current_msr, threshold, summary)
            print(f"✓ Formatted log saved to {args.formatted}")
            
            # After writing MSR, ask if user wants to test with new configuration
            print("\n" + "=" * 70)
            print("MSR write completed. Test with new configuration?")
            continue_choice = input(
                "Load new acoustic data file? (y/n or paste path): "
            ).strip()

            if not continue_choice:
                print("\nExiting. Review outliers and run again if needed.")
                sys.exit(0)

            if continue_choice.lower() == 'n':
                print("\nExiting. Review outliers and run again if needed.")
                sys.exit(0)

            # Load new file for next test
            while True:
                if continue_choice.lower() == 'y':
                    csv_path_input = input("\nEnter new file path (.csv, .xlsx, .xls, .wav): ").strip()
                else:
                    csv_path_input = continue_choice

                if not csv_path_input:
                    print("File path cannot be empty. Please try again.")
                    continue

                # Remove surrounding quotes if present
                csv_path_input = csv_path_input.strip('"').strip("'")
                csv_path = Path(csv_path_input)

                try:
                    print(f"Loading file: {csv_path}")
                    df = load_csv(csv_path)
                    print(f"✓ Successfully loaded {len(df)} rows")
                    break
                except FileNotFoundError as e:
                    print(f"✗ Error: {e}")
                except (ValueError, Exception) as e:
                    print(f"✗ Error: {e}")

                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("Exiting.")
                    sys.exit(0)
                continue_choice = 'y'
            
            # Reuse initial threshold for all runs
            print(f"Using initial threshold: {threshold}")
            
            # Analyze new outliers
            print("\n--- Outlier Analysis (New Configuration) ---")
            summary = summarize_outliers(df, threshold)
            if summary.empty:
                print("No outliers above threshold.")
            else:
                print(f"Found {len(summary)} outlier(s):\n")
                print(f"{'Frequency':<20} {'Measurement':<35} {'Noise (dB)':<15}")
                print("-" * 70)
                for _, row in summary.iterrows():
                    print(f"{str(row['level']):<20} {str(row['measurement']):<35} {row['noise']:<15.2f}")

            run_counter += 1
            plot_window.add_run(f"Run {run_counter}", df, threshold)
            
            skip_count = 0  # Reset counter to continue loop
    
    # Step 8: Exit
    print("\nExiting.")


if __name__ == "__main__":
    main()
