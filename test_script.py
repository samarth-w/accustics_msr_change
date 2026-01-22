import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

#!/usr/bin/env python3

# Check and install required dependencies
def check_and_install_dependencies():
    """Check for required packages and install if missing."""
    required_packages = {
        'pandas': 'pandas',
        'openpyxl': 'openpyxl',  # For Excel support
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


def load_csv(path: Path) -> pd.DataFrame:
    try:
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
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv, .xlsx, or .xls")
        
        # Check for the acoustic data format with Frequency column
        if "Frequency" in df.columns:
            # Rename for consistency
            df = df.rename(columns={"Frequency": "level"})
        elif "level" not in df.columns:
            raise ValueError("File must have 'Frequency' or 'level' column.")
        
        # Convert numeric columns to proper numeric type
        for col in df.columns:
            if col != 'level':
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
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


def main():
    parser = argparse.ArgumentParser(description="MSR outlier analyzer")
    parser.add_argument("--log", type=Path, default=Path("msr_log.csv"), help="Log file to update")
    parser.add_argument("--outliers", type=Path, default=Path("msr_outliers.csv"), help="Outliers log file")
    parser.add_argument("--formatted", type=Path, default=Path("msr_formatted_log.txt"), help="Formatted log file")
    args = parser.parse_args()

    print("=== MSR Outlier Analyzer ===\n")
    
    # Step 1: Input - Get CSV path dynamically
    while True:
        csv_path_input = input("Enter file path (.csv, .xlsx, .xls): ").strip()
        if not csv_path_input:
            print("File path cannot be empty. Please try again.")
            continue
        
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
    
    # Step 6: Input - Target MSR (with double confirmation for exit)
    skip_count = 0
    while skip_count < 2:
        target_msr = input("\nEnter target MSR register (hex, e.g. 0x1b, or press Enter to skip): ").strip()
        
        if not target_msr:
            skip_count += 1
            if skip_count == 1:
                confirm = input("Press Enter again to exit, or type MSR address: ").strip()
                if confirm:
                    target_msr = confirm
                    skip_count = 0  # Reset since user provided input
                else:
                    print("Exiting.")
                    sys.exit(0)
            else:
                print("Exiting.")
                sys.exit(0)
        
        if target_msr:
            # Step 7: Write into MSR
            target_value = input("Enter value to write (hex): ").strip()
            
            if not target_value:
                print("No value provided. Skipping write.")
                continue
            
            print(f"\nWriting {target_value} to MSR {target_msr}...")
            result = write_msr(target_msr, target_value)
            print(f"Write result: {result}")
            
            # Read back the value
            current_msr = read_msr(target_msr)
            print(f"MSR {target_msr} current value: {current_msr}")
            
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
            continue_choice = input("Load new acoustic data file? (y/n): ").strip().lower()
            
            if continue_choice != 'y':
                print("\nExiting. Review outliers and run again if needed.")
                sys.exit(0)
            
            # Load new file for next test
            while True:
                csv_path_input = input("\nEnter new file path (.csv, .xlsx, .xls): ").strip()
                if not csv_path_input:
                    print("File path cannot be empty. Please try again.")
                    continue
                
                csv_path = Path(csv_path_input)
                
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
                        sys.exit(0)
                except (ValueError, Exception) as e:
                    print(f"✗ Error: {e}")
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        print("Exiting.")
                        sys.exit(0)
            
            # Get new threshold
            while True:
                threshold_input = input("\nEnter threshold value: ").strip()
                try:
                    threshold = float(threshold_input)
                    break
                except ValueError:
                    print("✗ Invalid threshold. Please enter a numeric value.")
            
            print(f"Threshold set to: {threshold}")
            
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
            
            skip_count = 0  # Reset counter to continue loop
    
    # Step 8: Exit
    print("\nExiting.")


if __name__ == "__main__":
    main()
