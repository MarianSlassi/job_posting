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


import kagglehub
from kagglehub import KaggleDatasetAdapter
from datasets import load_dataset

# Modules for data 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def size_memory_info(df: pd.DataFrame, name: str = 'current df'):
    size_in_bytes = df.memory_usage(deep=True).sum()
    size_in_megabytes = size_in_bytes / (1024 ** 2)
    size_in_gigabytes = size_in_bytes / (1024 ** 3)

    print(f"\nMemory usage of {name}: {size_in_megabytes:.2f} MB ~ {size_in_gigabytes:.2f} GB\
                \nNumber of rows in this table: {df.shape[0]}\
                \nNumber of columns in this table: {df.shape[1]}\n")
    
def _clean_text(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    
    return x

# Loading Data
load_dotenv()
token = os.getenv("HF_TOKEN")
os.environ["HF_DATASETS_CACHE"] = str(config.get('raw_dir'))

data = load_dataset("2024-mcm-everitt-ryan/job-postings-english-clean", token = token)
df = data["train"].select_columns(['category','job_posting']).to_pandas()

# Category - unkown values remove
df = df[~(df['category'].str.contains('UNKNOWN', na=False))]
# Vacancy Description - both duplicates removing
df = df[~(df.duplicated(subset=["job_posting"], keep=False))]

# Remove categories with few rows
counts = df["category"].value_counts()
max_count = counts.max()
valid_categories = counts[counts >= 0.01 * max_count].index
df = df[df["category"].isin(valid_categories)].copy()

# Normalise Categories names
synonym_map = {
    
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


bad_categories = {
    'ONET', 'NHS', 'Careers', 'Workplace', 'Other', 'Others', 'Multilingual', 'Seasonal', 'Temporary'
}

# Delete trailing spaces for categories
df['category'] = df['category'].apply(_clean_text)

# Delete not real categories as 'other'
df = df.loc[~(df['category'].isin(bad_categories))].copy()

# Normalise categories names
df['category'] = df['category'].map(lambda x: synonym_map.get(x, x))

# Vacancy Description -  remove trailing spaces
df['job_posting'] = df['job_posting'].apply(_clean_text)

# Vacancy Description - Remove too long and too short descriptions
df = df[
    df["job_posting"].astype(str).str.split().str.len().between(100, 1000)
].copy()

print("Cut length, now it's:", len(df))

# Leave only top categories with more data
top_categories = df['category'].value_counts().head(25).index
df = df[df['category'].isin(top_categories)].copy()

print('Starting saving data')
df.to_parquet(config.get('cleaned_parquet'))
print('Data Saved')
# After that we usually save it's dataframe. 
# We should note it contains phone numbers and emails. While we do use TF-IDF which preprocess it for LogReg.
