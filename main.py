import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import typer
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import snapshot_download

app = typer.Typer(help="CLI utilities for ModelRetrieval Subtask A")
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root_logger.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)


@app.callback()
def entrypoint() -> None:
    """Root CLI callback to keep explicit subcommand behavior."""
    configure_logging()


def parse_model_repo_id(model_hf_url: str) -> str:
    """Extract a Hugging Face model repo id from a model URL."""
    parsed = urlparse(model_hf_url)
    parts = [segment for segment in parsed.path.split("/") if segment]
    if not parts:
        raise ValueError(f"Cannot parse model repo from URL: {model_hf_url}")

    # Handle URLs like /owner/model/resolve/main/... by trimming at route markers.
    route_markers = {"resolve", "blob", "tree", "commit", "raw", "discussions"}
    for idx, part in enumerate(parts):
        if part in route_markers:
            parts = parts[:idx]
            break

    if parts and parts[0] == "models":
        parts = parts[1:]

    if not parts:
        raise ValueError(f"Cannot parse model repo from URL: {model_hf_url}")

    return "/".join(parts[:2])


def _format_task_id(task_id: int) -> str:
    return str(task_id).zfill(4)


def _read_tasks_table() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent
    tasks_csv_path = project_root / "data" / "tasks.csv"
    return pd.read_csv(tasks_csv_path)


def _load_hf_task_dataset(row: pd.Series) -> DatasetDict | Dataset:
    dataset_name = str(row["task_hf_dataset_name"]).strip()
    subset_name = row.get("task_hf_subset_name")

    logger.info("Loading dataset: %s subset=%s", dataset_name, subset_name)

    if pd.isna(subset_name) or not str(subset_name).strip():
        return load_dataset(dataset_name)

    return load_dataset(dataset_name, str(subset_name).strip())


def _ensure_dataset_dict(ds: DatasetDict | Dataset) -> DatasetDict:
    if isinstance(ds, DatasetDict):
        return ds
    return DatasetDict({"train": ds})


def preserve_features_from_df(df: pd.DataFrame, template_ds: Dataset) -> Dataset:
    """Create a Dataset from pandas DataFrame and attempt to preserve template features."""
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    try:
        ds = ds.cast(template_ds.features)
    except Exception:
        pass
    return ds


def _split_train_dataset(train_dataset: Dataset, seed: int = 0) -> tuple[Dataset, Dataset, Dataset]:
    first_split = train_dataset.train_test_split(test_size=0.2, seed=seed)
    second_split = first_split["test"].train_test_split(test_size=0.5, seed=seed)
    return first_split["train"], second_split["train"], second_split["test"]


def _get_task_splits(
    ds: DatasetDict | Dataset, seed: int = 0, label_col: str | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    ds_dict = _ensure_dataset_dict(ds)
    split_names = set(ds_dict.keys())

    if {"train", "validation", "test"}.issubset(split_names):
        return ds_dict["train"], ds_dict["validation"], ds_dict["test"]

    if "train" not in split_names:
        raise ValueError(f"train split not found in dataset splits: {sorted(split_names)}")

    train_dataset = ds_dict["train"]

    if "validation" not in split_names and "test" not in split_names:
        df_full = pd.DataFrame(train_dataset)
        if label_col and label_col in df_full.columns:
            stratified = safe_stratified_resplit(df_full, label_col, seed, 0.2, 0.5, logger)
            if stratified is not None:
                df_train, df_val, df_test = stratified
                return (
                    preserve_features_from_df(df_train, train_dataset),
                    preserve_features_from_df(df_val, train_dataset),
                    preserve_features_from_df(df_test, train_dataset),
                )

        ds_split_1 = train_dataset.train_test_split(test_size=0.2, seed=seed)
        ds_train = ds_split_1["train"]

        ds_split_2 = ds_split_1["test"].train_test_split(test_size=0.5, seed=seed)
        ds_val = ds_split_2["train"]
        ds_test = ds_split_2["test"]

        return ds_train, ds_val, ds_test

    if "validation" in split_names and "test" not in split_names:
        ds_train_val = train_dataset.train_test_split(test_size=0.1, seed=seed)
        return ds_train_val["train"], ds_dict["validation"], ds_train_val["test"]

    if "test" in split_names and "validation" not in split_names:
        ds_train_val = train_dataset.train_test_split(test_size=0.1, seed=seed)
        return ds_train_val["train"], ds_train_val["test"], ds_dict["test"]

    if "validation" in split_names and "test" in split_names:
        ds_train_val = train_dataset.train_test_split(test_size=0.2, seed=seed)
        return ds_train_val["train"], ds_dict["validation"], ds_dict["test"]

    return _split_train_dataset(train_dataset, seed=seed)


def _feature_names_for_column(ds: Dataset, column_name: str) -> list[str] | None:
    feature = ds.features.get(column_name)
    if hasattr(feature, "names"):
        return list(feature.names)
    return None


def _split_column_names(column_spec: str) -> list[str]:
    return [column.strip() for column in str(column_spec).split(",") if column.strip()]


def _combine_text_columns(df: pd.DataFrame, text_columns: list[str]) -> pd.Series:
    missing_columns = [column for column in text_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"text_col(s) not found in dataset columns: {missing_columns}; available: {list(df.columns)}")

    if len(text_columns) == 1:
        return df[text_columns[0]].astype(str)

    return df[text_columns].astype(str).agg(". ".join, axis=1)


def limit_rows(df: pd.DataFrame, max_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Limit DataFrame to maximum number of rows via random sampling."""
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=seed)
    return df


def _one_hot_label_to_id(df: pd.DataFrame, label_columns: list[str]) -> pd.Series:
    missing_columns = [column for column in label_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"label_col(s) not found in dataset columns: {missing_columns}; available: {list(df.columns)}"
        )

    label_frame = df[label_columns]

    def _row_to_label(row: pd.Series) -> object:
        non_null_row = row.dropna()
        if non_null_row.empty:
            return pd.NA

        try:
            numeric_row = pd.to_numeric(non_null_row)
        except Exception:
            active_labels = [column for column, value in non_null_row.items() if bool(value)]
            if not active_labels:
                return pd.NA
            return active_labels[0] if len(active_labels) == 1 else active_labels[0]

        if (numeric_row > 0).any():
            return numeric_row.idxmax()

        return pd.NA

    return label_frame.apply(_row_to_label, axis=1)


def safe_stratified_resplit(
    df_full: pd.DataFrame, label_col: str, seed: int, t1: float, t2: float, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Attempt sklearn stratified 2-stage split. Return None on failure."""
    try:
        from sklearn.model_selection import train_test_split

        df_train_val, df_test = train_test_split(df_full, test_size=t1, random_state=seed, stratify=df_full[label_col])
        df_train, df_val = train_test_split(
            df_train_val, test_size=t2, random_state=seed, stratify=df_train_val[label_col]
        )
        return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
    except Exception as exc:
        logger.warning("Stratified resplit failed (%s). Falling back to non-stratified split.", exc)
        return None


def _standardize_split(
    ds_split: Dataset, text_column: str, label_column: str, max_rows: int = 5000, seed: int = 0
) -> pd.DataFrame:
    df = pd.DataFrame(ds_split)
    if df.empty:
        return df

    text_columns = _split_column_names(text_column)
    label_columns = _split_column_names(label_column)

    text_series = _combine_text_columns(df, text_columns)
    label_feature_names = _feature_names_for_column(ds_split, label_columns[0]) if len(label_columns) == 1 else None
    label_series = df[label_columns[0]] if len(label_columns) == 1 else _one_hot_label_to_id(df, label_columns)

    if label_feature_names is not None:

        def _normalize_label(value: object) -> object:
            if pd.isna(value):
                return value
            if isinstance(value, str):
                return value
            return label_feature_names[int(value)]

        label_series = label_series.map(_normalize_label)

    if len(label_columns) > 1:
        label_series = label_series.astype("string")

    standardized = pd.DataFrame({"text": text_series, "labels": label_series})
    standardized = standardized.dropna(subset=["text", "labels"]).reset_index(drop=True)
    return limit_rows(standardized, max_rows=max_rows, seed=seed)


def _write_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    logger.info("Saved %s rows to %s", len(df), output_path)


def _process_single_task(row: pd.Series, output_root: Path, seed: int = 0, max_rows: int = 5000) -> None:
    ds = _load_hf_task_dataset(row)
    label_column = str(row["label_col"]).strip()
    ds_train, ds_val, ds_test = _get_task_splits(ds, seed=seed, label_col=label_column)

    text_column = str(row["text_col"]).strip()

    task_id = int(row["task_id"])
    task_dir = output_root / _format_task_id(task_id)
    if task_dir.exists():
        logger.info("Removing existing task directory: %s", task_dir)
        shutil.rmtree(task_dir)

    _write_jsonl(
        _standardize_split(ds_train, text_column, label_column, max_rows=max_rows, seed=seed), task_dir / "train.jsonl"
    )
    _write_jsonl(
        _standardize_split(ds_val, text_column, label_column, max_rows=max_rows, seed=seed), task_dir / "val.jsonl"
    )


@app.command("download_test_tasks")
def download_test_tasks(
    task_ids: list[int] = typer.Option(
        None,
        "--task-id",
        help="Test task ID to download. Repeat the option for multiple IDs.",
    ),
    all_tasks: bool = typer.Option(
        False,
        "--all",
        help="Download all test tasks listed in data/tasks.csv.",
    ),
) -> None:
    """Download test task datasets into data/task-data/test-tasks/{task_id}."""
    if all_tasks and task_ids:
        raise typer.BadParameter("Use either --all or --task-id, not both.")
    if not all_tasks and not task_ids:
        raise typer.BadParameter("Provide --all or at least one --task-id.")

    project_root = Path(__file__).resolve().parent
    output_root = project_root / "data" / "task-data" / "test-tasks"
    output_root.mkdir(parents=True, exist_ok=True)

    df_tasks = _read_tasks_table()
    df_test_tasks = df_tasks[df_tasks["task_type"] == "test"].copy()
    test_task_ids = set(int(task_id) for task_id in df_test_tasks["task_id"].tolist())

    requested_ids = sorted(test_task_ids) if all_tasks else sorted(set(task_ids))
    missing_ids = [task_id for task_id in requested_ids if task_id not in test_task_ids]
    if missing_ids:
        raise typer.BadParameter(f"Unknown or non-test task_id(s): {missing_ids}")

    for task_id in requested_ids:
        try:
            row = df_test_tasks[df_test_tasks["task_id"] == task_id].iloc[0]
            logger.info("Downloading task_id=%s -> %s", task_id, output_root / _format_task_id(task_id))
            _process_single_task(row, output_root)

        except Exception as e:
            logger.error("Error occurred while downloading task_id=%s: %s", task_id, str(e))
            continue

    logger.info("Done")


@app.command("download_models")
def download_models(
    model_ids: list[int] = typer.Option(
        None,
        "--model-id",
        help="Model ID to download. Repeat the option for multiple IDs.",
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        help="Download all models listed in data/models.csv.",
    ),
) -> None:
    """Download selected models from data/models.csv into data/models/{model_id}."""
    if all_models and model_ids:
        raise typer.BadParameter("Use either --all or --model-id, not both.")
    if not all_models and not model_ids:
        raise typer.BadParameter("Provide --all or at least one --model-id.")

    project_root = Path(__file__).resolve().parent
    models_csv_path = project_root / "data" / "models.csv"
    output_root = project_root / "data" / "models"
    output_root.mkdir(parents=True, exist_ok=True)

    df_models = pd.read_csv(models_csv_path)
    model_rows = df_models.set_index("model_id")
    csv_model_ids = set(model_rows.index.tolist())

    requested_ids = sorted(csv_model_ids) if all_models else sorted(set(model_ids))
    missing_ids = [model_id for model_id in requested_ids if model_id not in csv_model_ids]
    if missing_ids:
        raise typer.BadParameter(f"Unknown model_id(s): {missing_ids}")

    id_width = max(2, len(str(max(csv_model_ids))))

    for model_id in requested_ids:
        try:
            model_row = model_rows.loc[model_id]
            model_hf_url = str(model_row["model_hf_url"])
            repo_id = parse_model_repo_id(model_hf_url)
            model_output_dir = output_root / str(model_id).zfill(id_width)

            logger.info("Downloading model_id=%s (%s) -> %s", model_id, repo_id, model_output_dir)
            snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(model_output_dir))

        except Exception as e:
            logger.error("Error occurred while downloading model_id=%s: %s", model_id, str(e))
            continue

    logger.info("Done")


if __name__ == "__main__":
    app()
