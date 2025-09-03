# System/env config
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

parent_dir = Path.cwd().resolve().parent
sys.path.append(str(parent_dir))
print('Current dir for import:', parent_dir)

from config import Config
config = Config()
print('Config initialized')


from datasets import load_dataset

# Modules for data 
import pandas as pd
import re, json


class ETL:
    def __init__(self, config):
        self.config = config if config else Config()
        self.synonym_map = {
            
            'Administration': 'Administrative',
            'Administrative': 'Administrative',
            'Receptionists': 'Administrative',
            'Assistants': 'Administrative',

            
            'Accountancy': 'Accounting',
            'Accounting': 'Accounting',

            
            'Telecommunication': 'Telecommunications',
            'Telecommunications': 'Telecommunications',

            
            'Executive': 'Management',
            'Managerial': 'Management',
            'Management': 'Management',

            
            'IT': 'IT',
            'Computer': 'IT',
            'Technology': 'IT',
            'Developers': 'IT',
            'Web': 'IT',
            'Internet': 'IT',

            
            'Medical': 'Healthcare',
            'Healthcare': 'Healthcare',
            'Nursing': 'Healthcare',  

            
            'Scientific': 'Science',
            'Science': 'Science',
            'Research': 'Science',  

            
            'Restaurant': 'Hospitality',
            'Hospitality': 'Hospitality',
            'Housekeeping': 'Hospitality',

            
            'Consultancy': 'Consulting',
            'Consulting': 'Consulting',

            
            'QA': 'Quality Assurance',  
            'Manufacturing': 'Manufacturing',  
            'Production': 'Manufacturing',     

            
            'Graduate': 'Training',
            'Training': 'Training',
            'Education': 'Education',  

            
            'Purchasing': 'Procurement',  
        }

 
        self.bad_categories = {
            'ONET', 'NHS', 'Careers', 'Workplace', 'Other', 'Others', 'Multilingual', 'Seasonal', 'Temporary'
        }
        pass

    def _size_memory_info(self, df: pd.DataFrame, name: str = 'current df'):
        size_in_bytes = df.memory_usage(deep=True).sum()
        size_in_megabytes = size_in_bytes / (1024 ** 2)
        size_in_gigabytes = size_in_bytes / (1024 ** 3)

        print(f"\nMemory usage of {name}: {size_in_megabytes:.2f} MB ~ {size_in_gigabytes:.2f} GB\
                    \nNumber of rows in this table: {df.shape[0]}\
                    \nNumber of columns in this table: {df.shape[1]}\n")
        
    def _clean_text(self, x):
        if pd.isna(x):
            return x
        x = str(x).strip()
        
        return x
    
    def extract(self):
        # Loading Data
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        os.environ["HF_DATASETS_CACHE"] = str(config.get('raw_dir'))
        data = load_dataset("2024-mcm-everitt-ryan/job-postings-english-clean", token = token)
        df = data["train"].select_columns(['category','job_posting']).to_pandas()
        return df

    def transform(self, df, top_cats = 25): # top_cats - amount of categories from top of those with observations

        # Category - unkown values remove
        df = df[~(df['category'].str.contains('UNKNOWN', na=False))]
        # Vacancy Description - both duplicates removing
        df = df[~(df.duplicated(subset=["job_posting"], keep=False))]

        # Remove categories with few rows
        counts = df["category"].value_counts()
        max_count = counts.max()
        valid_categories = counts[counts >= 0.01 * max_count].index
        df = df[df["category"].isin(valid_categories)].copy()

        # Delete trailing spaces for categories
        df['category'] = df['category'].apply(self._clean_text)

        # Delete not real categories as 'other'
        df = df.loc[~(df['category'].isin(self.bad_categories))].copy()

        # Normalise categories names
        df['category'] = df['category'].map(lambda x: self.synonym_map.get(x, x))

        # Vacancy Description -  remove trailing spaces
        df['job_posting'] = df['job_posting'].apply(self._clean_text)

        # Vacancy Description - Remove too long and too short descriptions
        df = df[
            df["job_posting"].astype(str).str.split().str.len().between(100, 1000)
        ].copy()

        print("Cut length, now it's:", len(df))

        # Leave only top categories with more data
        top_categories = df['category'].value_counts().head(top_cats).index
        df = df[df['category'].isin(top_categories)].copy()
        return df

    def load(self, df, path):
        print('Starting saving data')
        df.to_parquet(path)
        print('Data Saved')
        # After that we usually save it's dataframe. 
        # We should note it contains phone numbers and emails. While we do use TF-IDF which preprocess it for LogReg.

    def bert_processing(self, df):
        def clean_for_bert(text: str) -> str:
            text = str(text)
            text = re.sub(r"<.*?>", " ", text)                     # HTML
            text = re.sub(r"http\S+|www\.\S+", " ", text)          # URL
            text = re.sub(r"\S+@\S+", " ", text)                   # email
            text = re.sub(r"\+?\d[\d\-\(\) ]{7,}\d", " ", text)    # phones
            text = re.sub(r"\s+", " ", text).strip()
            return text

        assert {"job_posting", "category"}.issubset(df.columns), df.columns

        df["job_posting"] = df["job_posting"].apply(clean_for_bert)


        labels = sorted(df["category"].unique())
        label2id = {lbl:i for i,lbl in enumerate(labels)}
        id2label = {i:lbl for lbl,i in label2id.items()}
        df["category"] = df["category"].map(label2id).astype(int)

        with open(self.config['label2id'], "w") as f: json.dump(label2id, f)
        with open(self.config['id2label'], "w") as f: json.dump(id2label, f)

        df = df.rename(columns={'job_posting':'text', 'category': 'labels'})

        df.to_parquet(self.config.get('cleaned_for_bert_training'))

    def run(self, top_cats=25, bert=True):
        df = self.extract()
        df = self.transform(df, top_cats = top_cats)
        self.load(df, self.config.get('cleaned_parquet'))
        if bert:
            df = self.bert_processing(df)
        self.load(df, self.config.get('cleaned_for_bert_training'))

if __name__ == "__main__":
    config = Config()
    etl = ETL(config)
    etl.run(bert = False, top_cats = 25)


"""
Example of usage:

config = Config()
etl = ETL(config)
etl.run(bert = True, top_cats = 25)


Uses config variables named:
'cleaned_parquet' - to save final processed data with 'top_cats' number of most informative categories in trainin examples
'cleaned_for_bert_training' - to clean data for bert training with less altering data itself but removing redudntant "number" form text e.g. phones, personal id numbers etc. 

"""
