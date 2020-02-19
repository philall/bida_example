import os

def get_project_dir(start_point: str) -> str:
    return os.path.basename(os.path.abspath(start_point))

def get_data_source(path: str, dir: str='data', file: str='glass.csv') -> str:
    return os.path.join(path, dir, file_name)
