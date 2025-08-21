# Config Class Documentation

The `Config` class manages all project paths in a structured and reproducible way. By default, it organizes your data, models, logs, and other assets into versioned directories under a base folder. You can override any of these defaults or extend the config with custom paths.

Except of creating and choosing only dedault (`base_dir`) directory for the config you can use the following:

## Usage
```
cfg = Config(
    base_dir='data',
    raw_dir='external_data/raw',
    models_dir ='tmp/models'
)

or more explicit:

cfg = Config(
    base_dir=Path("my_project_data"),
    logs_dir=Path("/tmp/my_logs"),
    raw_dir=Path("/mnt/raw_datasets"),
    models_dir=Path("/tmp/metrics")  # это будет в extra
)

```


## Default Directories (auto-created if not overridden)
- `raw_dir`: base/01_raw – contains raw CSVs and submission files
- `cleaned_dir`: base/02_cleaned – cleaned dataset
- `interim_dir`: base/03_interim – intermediate/debugging data
- `features_dir`: base/04_features – final features
- `processed_dir`: base/05_processed – train/test files
- `predict_dir`: base/06_predictions – model predictions
- `logs_dir`: base/07_logs – all log files
- `models_dir`: models folder (separate from base)

## Key Methods
- `cfg.get("train_x")`: access a known path
- `cfg.get_model_param("xgb_params")`: access previously saved best found model params
- `cfg["log_file_etl"]`: dictionary-like access (only read)
- `cfg.set(key, value)`: manually update a non-directory path (means names of files and sub paths). Set all values excepts key with "_dir" string in name
- `cfg.keys()`: list all standard keys
- `cfg.as_dict()`: export config as a dictionary (for vivid examinitation of object instance)

## Notes
- All known directories and files are created on `.get` method which envoked in every `.run` method of main modules classes.
- You  can manually store best models params and use different key. [this point is for for developers / doesn't supported in package API yet]
- Changing path logic is allowed, changing key names is discouraged.
