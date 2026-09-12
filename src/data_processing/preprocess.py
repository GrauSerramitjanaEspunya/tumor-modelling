from pathlib import Path
import pandas as pd
import numpy as np
import sys

def eliminate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Eliminates the duplicates on the datasets."""
    df = df.drop_duplicates(subset=["TRT", "DAYS"]).copy()
    df["ID"] = df.groupby("TRT").ngroup() + 1

    return df

def rtv(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a Relative Tumor Volume column to the dataset."""
    baseline = df["VOL"].iloc[0] # All mice in the same dataset have the same baseline
    df["RTV"] = df["VOL"] / baseline

    return df


def logvol(df: pd.DataFrame) -> pd.DataFrame:
    """Adds logVolume column to the dataset."""
    df["LOGVOL"] = np.log(df["VOL"])

    return df


def add_dose(df: pd.DataFrame) -> pd.DataFrame:
    """Adds dose information to the CDKi+Gemcitabine dataset"""
    df["DOSE"] = np.array([0,0,0,0,0,0,0,0,0,0,0,
                           50,50,50,50,50,50,0,0,0,0,0,
                           60,0,60,60,60,60,60,0,60,60,60,
                           50,50,0,0,0,0,0,0,0,0,0,
                           110,0,110,110,110,110,60,0,0,0,0])

    return df

def process_dataset(input_filepath: str, output_filepath: str, dose: str) -> None:
    """
    Preprocesses the file located at 'input_filepath' and exports the dataset as a .csv to 'output_filepath'.
    If the dataset treats the mice with CDKi+Gemcitabine (dose=T) it adds the information about the dose to the dataset.
    """
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    # Read dataset eliminate duplicates
    df = pd.read_csv(input_path)
    df = eliminate_duplicates(df)

    # Preprocess
    df = rtv(df)
    df = logvol(df)
    if dose == 'T':
        df = add_dose(df)
    
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure existance of the output directory

    df.to_csv(output_filepath, index=False)
    print(f"Successfully processed: {input_filepath} -> {output_filepath}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python preprocess.py <input_filepath> <output_filepath> <T/F>")
        sys.exit(1)
    if sys.argv[3] not in 'TF':
        print("The third argument must be T or F")
        sys.exit(1)

    process_dataset(sys.argv[1], sys.argv[2], sys.argv[3])