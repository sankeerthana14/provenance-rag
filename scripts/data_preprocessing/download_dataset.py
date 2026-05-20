#===========================================================================================
# DOWNLOADING THE DATASETS
#---------------------------
# This file contains the code to download the HotpotQA, MusiQue and the FEVER datasets 
# locally.
#===========================================================================================

# 1. Imports
#------------
from datasets import load_dataset
import datasets
from pathlib import Path


def main():
    # 2. Downloading the HotPotQA dataset
    #----------------------------------------
    print("INFO: Downloading HotpotQA distractor split...")

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

    print(dataset)

    # Save locally inside repo
    dataset.save_to_disk("data/raw/hotpotqa_distractor")

    print("INFO: Saved HotpotQA to: data/raw/hotpotqa_distractor")


    # 3. Downloading the MuSiQue dataset
    #------------------------------------------
    output_dir = Path("data/raw/musique")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    print("INFO: Downloading MuSiQue from Hugging Face...")

    dataset = load_dataset("bdsaglam/musique")

    print("\nDataset:")
    print(dataset)

    dataset.save_to_disk(str(output_dir))

    print(f"\nSaved MuSiQue to: {output_dir}")


    # 4. Downloading the FEVER dataset
    #-------------------------------------
    output_dir = Path("data/raw/fever_v1")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print("INFO: Downloading FEVER v1.0...")

    print("NOTE: this requires datasets<4.0.0 because FEVER uses an older loading script.")

    major_version = 0
    try:
        major_version = int(datasets.__version__.split(".")[0])
    except ValueError:
        pass

    if major_version >= 4:
        print("ERROR: FEVER cannot be downloaded with datasets>=4 because dataset scripts are no longer supported.")
        print("Install an older version with: pip install 'datasets<4.0.0' and rerun this script.")
        return

    dataset = load_dataset(

        "fever/fever",
        "v1.0",
        trust_remote_code=True,
    )

    print("\nDataset:")
    print(dataset)

    dataset.save_to_disk(str(output_dir))
    print(f"\nINFO: Saved FEVER to: {output_dir}")


if __name__ == "__main__":
    main()
