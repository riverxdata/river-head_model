import pandas as pd
import argparse
from pathlib import Path

LABELS_MAPPING = {
    "VOL": "Healthy Controls",
    "LK": "Lung (LU)",
    "CC": "Colorectal (CR)",
    "GC": "Gastric (GA)",
    "PR": "Prostate (PR)",
    "PC": "Pancreatic (PA)",
    "BC": "Breast (BR)",
    "EC": "Esophageal Squamous Cell (ES)",
    "BT": "Biliary Tract (BT)",
    "OV": "Ovarian (OV)",
    "BL": "Bladder (BL)",
    "HC": "Hepatocellular (HC)",
    "SA_benign": "Benign Bone/Soft Tissue (SA_N)",
    "SA": "Bone and Soft Tissue Sarcomas (SA)",
    "GL": "Gliomas/Brain Tumors (GL)",
    "PR_benign": "Benign Prostate Disease (PR_N)",
    "BC_benign": "Benign Breast Disease (BR_N)",
    "OV_benign": "Benign Ovarian Disease (OV_N)",
    "GL_benign": "Benign Brain Disease (GL_N)",
}

CATEGORY_MAPPING = {
    "VOL": "Control",
    "LK": "Cancer",
    "CC": "Cancer",
    "GC": "Cancer",
    "PR": "Cancer",
    "PC": "Cancer",
    "BC": "Cancer",
    "EC": "Cancer",
    "BT": "Cancer",
    "OV": "Cancer",
    "BL": "Cancer",
    "HC": "Cancer",
    "SA_benign": "Benign",
    "SA": "Cancer",
    "GL": "Cancer",
    "PR_benign": "Benign",
    "BC_benign": "Benign",
    "OV_benign": "Benign",
    "GL_benign": "Benign",
}


def load_data(work_dir, split):
    """Load features and labels for a given split (train/test)."""
    work_dir = Path(work_dir)

    features = pd.read_csv(work_dir / "feature_names.txt", header=None, sep="\t")[0].to_list()

    df = pd.read_csv(work_dir / split / "feature_vectors.csv", header=None, names=features)

    labels_value_list = pd.read_csv(work_dir / "label_names.txt", header=None)[0].to_list()

    labels_value_dict = {i: label for i, label in enumerate(labels_value_list)}

    labels_index = pd.read_csv(work_dir / split / "labels.txt", header=None)[0].to_list()

    labels = [labels_value_dict[i] for i in labels_index]
    df["target"] = labels
    df["label"] = df["target"].map(LABELS_MAPPING)
    df["category"] = df["target"].map(CATEGORY_MAPPING)

    return df


def main():
    parser = argparse.ArgumentParser(description="Combine train and test data into a single dataframe")
    parser.add_argument(
        "--work_dir", type=str, required=True, help="Path to work directory containing feature and label files"
    )
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")

    args = parser.parse_args()

    # Load train and test data
    train_df = load_data(args.work_dir, "train")
    test_df = load_data(args.work_dir, "test")

    # Combine dataframes
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save combined dataframe
    combined_df.to_csv(args.output, index=False)
    print(f"Combined dataframe saved to {args.output}")
    print(f"Total rows: {len(combined_df)} (train: {len(train_df)}, test: {len(test_df)})")


if __name__ == "__main__":
    main()
