from pathlib import Path
import pandas as pd
import sys

def rtv(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a Relative Tumor Volume column to the dataset."""
    df["Baseline"] = df.groupby("ID")["Size"].transform("first")
    df["RTV"] = df["Size"] / df["Baseline"]

    return df


def process_dataset(input_filepath: str, output_filepath: str) -> None:
    """
    Preprocesses the file located at 'input_filepath' and exports the dataset as a .csv to 'output_filepath'.
    """
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    # Read dataset and adjust type
    df = pd.read_csv(input_path, sep=";", decimal=",")
    df["Size"] = df["Size"].astype(float)

    # Preprocess
    df = rtv(df)
    
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure existance of the output directory

    df.to_csv(output_filepath, index=False)
    print(f"Successfully processed: {input_filepath} -> {output_filepath}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input_filepath> <output_filepath> <T/F>")
        sys.exit(1)

    process_dataset(sys.argv[1], sys.argv[2])