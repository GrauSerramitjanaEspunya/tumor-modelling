from pathlib import Path
import pandas as pd
import numpy as np
import sys

def rtv(df: pd.DataFrame) -> None:
    """Adds a Relative Tumor Volume column to the dataset."""
    baseline = df["VOL"].iloc[0] # All mice in the same dataset have the same baseline
    df["RTV"] = df["VOL"] / baseline

    return df


def logvol(df: pd.DataFrame) -> None:
    """Adds logVolume column to the dataset."""
    df["LOGVOL"] = np.log(df["VOL"])

    return df


def process_dataset(input_filepath: str, output_filepath: str) -> None:
    """Preprocesses the file located at 'input_filepath' and exports the dataset as a .csv to 'output_filepath'"""
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    df = pd.read_csv(input_path)

    df = rtv(df)
    df = logvol(df)

    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure existance of the output directory

    df.to_csv(output_filepath, index=False)
    print(f"Successfully processed: {input_filepath} -> {output_filepath}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input_filepath> <output_filepath>")
        sys.exit(1)

    process_dataset(sys.argv[1], sys.argv[2])