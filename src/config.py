from pathlib import Path

class Config:

    def __init__(self, base_dir: Path = None, **custom_paths):
        self.base_dir = base_dir or Path(__file__).resolve().parents[1] / 'data' # store all data in dedicated folder ⚠️
        self.models_dir = base_dir or Path(__file__).resolve().parents[1] / 'models'

        '''
        Rules of devine config:
        1. Don't change variables names, only path.
        2. If you change path, it's up to you, but remember it's never late to turn back
        3. Configs sorted vice versa on purpose.
        Due to expecation you will like to change them as less
        as you get closer to raw data.

        Instance of usage:
        cfg = Config(
            base_dir=Path("my_data_folder"),
            raw_dir=Path("/mnt/drive/datasets/raw"),
            logs_dir=Path("/tmp/logs")
        )
        config.get('')
        '''

        def override_or_default(key: str, default: Path):
            path =  Path(custom_paths.get(key, default)) # if key exists in custom_path - return it as path, else take 'defatult' var
            path.mkdir(parents=True, exist_ok=True)
            return path


        # Sub data/ folders
        logs_dir      = override_or_default('logs_dir', self.base_dir / '07_logs')
        predict_dir   = override_or_default('predict_dir', self.base_dir / '06_predictions')
        processed_dir = override_or_default('processed_dir', self.base_dir / '05_processed')
        features_dir  = override_or_default('features_dir', self.base_dir / '04_features')
        interim_dir   = override_or_default('interim_dir', self.base_dir / '03_interim')
        cleaned_dir   = override_or_default('cleaned_dir', self.base_dir / '02_cleaned')
        raw_dir       = override_or_default('raw_dir', self.base_dir / '01_raw')
        models_dir    = override_or_default('models_dir', self.models_dir)
        
        kaggle_download = override_or_default('kaggle_download', raw_dir / 'datasets/arshkon/linkedin-job-postings/versions/13') 
        '''
        When downloading from kaggle to 01_raw it's created own subpath, we also can explicitly transfer
        it's subpath as config parameter if we want to take raw data outside of project directory.
        Otherwise we use in code:

        import os
        os.environ["KAGGLEHUB_CACHE"] = str(config.get('raw_dir'))

        Now no need to redefine kaggle_download variable when initing config bc it's stored at raw dir. 
        '''
        self._config = {
            # Logs _07
            'logs_dir':                 logs_dir,
            'log_file_etl':             logs_dir / 'etl_pipeline.log',
            'log_file_build_features':  logs_dir / 'build_features.log',
            'log_file_split':           logs_dir / 'split.log',
            'log_file_model':           logs_dir / 'model.log',

            # Expirements logging
            'log_ml_flow':              logs_dir / 'ml_flow.log',

            # Models
            'models_dir':               models_dir,
            'xgb_model':                models_dir / 'xgb_model.pkl',

            # Validation Schems _07
            'validation_schema_cleaned':   logs_dir / 'validation_schema_cleaned.log',
            'validation_schema_features':   logs_dir / 'validation_schema_features.log',

            # Predict _06
            'predict_dir':              predict_dir,
            'predict_base':             predict_dir / 'base_predicted.csv', # basic predictions
            'predict':                  predict_dir / 'predicted.csv',

            # Processed _05
            'processed_dir':            processed_dir,
            'train_x':                  processed_dir / 'train_x.parquet',
            'train_y':                  processed_dir / 'train_y.parquet',
            'inference':                processed_dir / 'inference.parquet',

            # Features _04
            'features_dir':             features_dir,
            'features':                 features_dir / 'full_features.parquet',

            # Interim _03
            'interim_dir':              interim_dir,
            'sales_for_eda':            interim_dir / 'sales_for_eda.parquet', # DQC Notebook output with merged dicts
            'interim_parquet':          interim_dir / 'data_checkpoint_full_df_feature_engineering.parquet', # Feature engineering debugging
            'full_df_test_csv':         interim_dir / 'full_df_test_csv.csv', # Test for features validation schema
            'cleaned_test_schema_csv':  interim_dir / 'cleaned_test_schema.csv', # Test for cleaned validation schema
            
            # Cleaned _02
            'cleaned_dir':              cleaned_dir,
            'cleaned_parquet':          cleaned_dir / 'sales_cleaned.parquet',

            # Raw _01
            'raw_dir':                  raw_dir,
            'postings':                  kaggle_download / 'postings.csv',
            'industries_id':            kaggle_download / 'jobs/job_industries.csv',
            'mapping_industries':       kaggle_download / 'mappings/industries.csv',
            'skills_id':                kaggle_download / 'jobs/job_skills.csv',
            'mapping_skills':           kaggle_download / 'mappings/skills.csv',
            'companies':                kaggle_download /  'companies/companies.csv'
        }
        self._model={

        }

    def get_model_param(self, key: str):
         return self._model[key]

    def get(self, key : str) -> Path:
            if key not in self._config:
                raise KeyError(f"Config key '{key}' not found.\nPossible keys: {self.keys()}")
            return self._config[key]

    def set(self, key: str, value: Path | str) -> None:
            if 'dir' in key:
                raise KeyError(f'You can set base directories only when creating config object')
            self._config[key] = Path(value)

    def as_dict(self) -> dict[str, Path]:
          return self._config.copy()

    def keys(self) -> list[str]:
          return list(self._config.keys())

    def __getitem__(self, key: str) -> Path:
        return self.get(key)                                             # To have possibility call keys through dict syntax e.g. config['key']

    def __contains__(self, key: str) -> bool:
        return key in self._config                                       # To support expressions as " if 'key' in config "

    def __repr__(self) -> str:
          return '\n'.join(f"{k}: {v}" for k, v in self._config.items()) # To use syntax as "print(config)"
