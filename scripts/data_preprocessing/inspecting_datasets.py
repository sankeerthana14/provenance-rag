#===============================================================================
# INSPECTING THE DATASETS
#---------------------------
# This script allows you to load the datasets from disk and inspect their 
# structure and contents in order to understand what all fields to add in the
# metadata.
#===============================================================================

# 1. Imports
#---------------
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
import json
from pathlib import Path
import shutil


def _replace_legacy_list_type(obj):
    if isinstance(obj, dict):
        # Legacy HF schema pattern: struct-of-lists used to represent list-of-struct.
        # Convert:
        # {"field_a": {"feature": ..., "_type": "List"}, ...}
        # into:
        # {"feature": {"field_a": ..., ...}, "_type": "Sequence"}
        if "_type" not in obj and obj:
            looks_like_struct_of_lists = True
            for value in obj.values():
                if not (
                    isinstance(value, dict)
                    and value.get("_type") in {"List", "Sequence"}
                    and "feature" in value
                    and len(value.keys()) == 2
                ):
                    looks_like_struct_of_lists = False
                    break
            if looks_like_struct_of_lists:
                struct_feature = {
                    key: _replace_legacy_list_type(value["feature"])
                    for key, value in obj.items()
                }
                return {"feature": struct_feature, "_type": "Sequence"}

        updated = {}
        for key, value in obj.items():
            if key == "_type" and value == "List":
                updated[key] = "Sequence"
            else:
                updated[key] = _replace_legacy_list_type(value)
        return updated
    if isinstance(obj, list):
        return [_replace_legacy_list_type(item) for item in obj]
    return obj


def _fix_legacy_feature_metadata(dataset_path: Path) -> int:
    patched_files = 0
    for info_path in dataset_path.rglob("dataset_info.json"):
        with info_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        fixed_data = _replace_legacy_list_type(data)
        if fixed_data != data:
            with info_path.open("w", encoding="utf-8") as f:
                json.dump(fixed_data, f, indent=2)
                f.write("\n")
            patched_files += 1
    return patched_files


def _rebuild_hotpotqa_dataset(original_path: Path) -> Path:
    rebuilt_path = original_path.parent / f"{original_path.name}_rebuilt"
    cache_dir = original_path.parent / "_hf_cache_hotpotqa_rebuild"

    print("INFO: Rebuilding HotPotQA dataset into a compatibility-safe format...")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        cache_dir=str(cache_dir),
        download_mode="force_redownload",
    )
    ds.save_to_disk(str(rebuilt_path))
    print(f"INFO: Rebuilt dataset saved to: {rebuilt_path}")
    return rebuilt_path


def _rebuild_musique_dataset(original_path: Path) -> Path:
    rebuilt_path = original_path.parent / f"{original_path.name}_rebuilt"
    cache_dir = original_path.parent / "_hf_cache_musique_rebuild"

    print("INFO: Rebuilding MuSiQue dataset into a compatibility-safe format...")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    ds = load_dataset(
        "bdsaglam/musique",
        cache_dir=str(cache_dir),
        download_mode="force_redownload",
    )
    ds.save_to_disk(str(rebuilt_path))
    print(f"INFO: Rebuilt dataset saved to: {rebuilt_path}")
    return rebuilt_path


def _maybe_rebuild_known_dataset(dataset_path: Path, error_message: str):
    dataset_name = dataset_path.name.lower()
    if "hotpotqa" in dataset_name and "Feature type 'List' not found" in error_message:
        print("INFO: Dataset still uses legacy Arrow metadata. Trying automatic HotPotQA rebuild...")
        rebuilt_path = _rebuild_hotpotqa_dataset(dataset_path)
        return load_from_disk(str(rebuilt_path)), rebuilt_path

    if "musique" in dataset_name and (
        "Feature type 'List' not found" in error_message or "Type mismatch:" in error_message
    ):
        print("INFO: MuSiQue metadata/schema mismatch detected. Trying automatic MuSiQue rebuild...")
        rebuilt_path = _rebuild_musique_dataset(dataset_path)
        return load_from_disk(str(rebuilt_path)), rebuilt_path

    return None, dataset_path


# 2. Function to load and print an example from the dataset
#-------------------------------------------------------------
def _print_example(dataset_path: str, split: str = "train", index: int = 0):
    # Converting to a Path Object for better handling of file paths
    dataset_path = Path(dataset_path) 

    # Directory Absense Check and Handling
    if not dataset_path.exists():
        raise FileNotFoundError(f"INFO: Dataset path does not exist: {dataset_path}")
    print(f"INFO: Loading dataset from: {dataset_path}")

    # Loading the file and error handling
    try:
        ds = load_from_disk(str(dataset_path))
    except Exception as exc:
        error_msg = str(exc)
        if "Feature type 'List' not found" in error_msg:
            print("INFO: Detected legacy feature metadata. Applying compatibility fix...")
            patched_files = _fix_legacy_feature_metadata(dataset_path)
            print(f"INFO: Patched {patched_files} dataset_info.json file(s). Retrying load...")
            try:
                ds = load_from_disk(str(dataset_path))
            except Exception as retry_exc:
                retry_msg = str(retry_exc)
                try:
                    rebuilt_ds, rebuilt_path = _maybe_rebuild_known_dataset(dataset_path, retry_msg)
                    if rebuilt_ds is not None:
                        ds = rebuilt_ds
                        dataset_path = rebuilt_path
                    else:
                        raise RuntimeError(
                            f"ERROR: Failed to load dataset from '{dataset_path}' after metadata fix: {retry_exc}"
                        ) from retry_exc
                except Exception as rebuild_exc:
                    raise RuntimeError(
                        "ERROR: Failed to auto-rebuild dataset. "
                        f"Please rerun scripts/data_preprocessing/download_dataset.py. Details: {rebuild_exc}"
                    ) from rebuild_exc
        elif "Type mismatch:" in error_msg:
            try:
                print("INFO: Detected metadata/schema type mismatch. Applying compatibility normalization...")
                patched_files = _fix_legacy_feature_metadata(dataset_path)
                print(f"INFO: Patched {patched_files} dataset_info.json file(s). Retrying load...")
                try:
                    ds = load_from_disk(str(dataset_path))
                except Exception as retry_exc:
                    rebuilt_ds, rebuilt_path = _maybe_rebuild_known_dataset(dataset_path, str(retry_exc))
                    if rebuilt_ds is not None:
                        ds = rebuilt_ds
                        dataset_path = rebuilt_path
                    else:
                        raise RuntimeError(
                            f"ERROR: Failed to load dataset from '{dataset_path}' after type-mismatch normalization: {retry_exc}"
                        ) from retry_exc
            except Exception as rebuild_exc:
                raise RuntimeError(
                    "ERROR: Failed to auto-rebuild dataset after type mismatch. "
                    f"Please rerun scripts/data_preprocessing/download_dataset.py. Details: {rebuild_exc}"
                ) from rebuild_exc
        else:
            raise RuntimeError(f"ERROR:Failed to load dataset from '{dataset_path}': {exc}") from exc

    print("\nDataset object:")
    print(ds)

    # Checks for DatasetDict (multiple splits) vs Dataset (single split) and split existence
    if isinstance(ds, DatasetDict):
        print("\nINFO: Available splits:")
        print(ds.keys())
        if split not in ds:
            raise ValueError(
                f"ERROR: Split '{split}' not found. Available splits: {list(ds.keys())}"
            )
        target_ds = ds[split]
    elif isinstance(ds, Dataset):
        print("\nLoaded a single split Dataset (not DatasetDict).")
        target_ds = ds
    else:
        raise TypeError(f"Unsupported dataset type: {type(ds)}")

    if index < 0 or index >= len(target_ds):
        raise IndexError(
            f"Index {index} is out of range for dataset with {len(target_ds)} rows."
        )

    # Picks an example to inspect 
    example = target_ds[index]

    print("\nExample keys:")
    print(example.keys())

    print("\nExample:")

    # Keep terminal output ASCII-safe to avoid UnicodeEncodeError on Windows consoles.
    print(json.dumps(example, indent=2, ensure_ascii=True, default=str)[:5000])


# 3. Separate functions for inspecting each dataset
#------------------------------------------------------------
def inspect_hotpotqa(
    split: str = "train",
    index: int = 0,
    dataset_path: str = "data/raw/hotpotqa_distractor",
):
    _print_example(dataset_path=dataset_path, split=split, index=index)


def inspect_musique(
    split: str = "train",
    index: int = 0,
    dataset_path: str = "data/raw/musique",
):
    _print_example(dataset_path=dataset_path, split=split, index=index)


def inspect_fever(
    split: str = "train",
    index: int = 0,
    dataset_path: str = "data/raw/fever_v1",
):
    _print_example(dataset_path=dataset_path, split=split, index=index)


if __name__ == "__main__":
    inspect_hotpotqa(split="train", index=0)
    inspect_musique(split="train", index=0)
    inspect_fever(split="train", index=0)
