import os
from enum import Enum
import pandas as pd
from fastparquet import write

def directory_up(path: str, n: int):
    for _ in range(n):
        path = directory_up(path.rpartition("/")[0], 0)
    return path


root_path = os.path.dirname(os.path.realpath(__file__))
# Change working directory to root of the project.
os.chdir(directory_up(root_path, 1))

class Filenames(Enum):
    train = 'train'
    test = 'test'

def read_csv(filename: Filenames):
    return pd.read_csv(f'./data/{filename.value}.csv')

def save_csv(data, filename:Filenames):
    data.to_csv(f"./data/output_{filename.value}.csv", index = False, sep = ',')

def save_parquet(filename: Filenames, data):
    write(f'./data/{filename.value}.parquet', data)

def read_parquet(filename: Filenames):
    return pd.read_parquet(f'data/{filename.value}.parquet', engine='fastparquet')
