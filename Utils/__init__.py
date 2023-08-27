import numpy as np

def read_npz_file(file_path):
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None