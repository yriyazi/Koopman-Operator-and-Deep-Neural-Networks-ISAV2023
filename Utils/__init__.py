import                                  numpy               as np
from .utils                     import  *
from .Plot.Koopman_Eigenvalue   import  Koopman_Eigenvalue
from .Eigen_values              import  *
from .Plot.Plot                 import  *
from .configuration             import  *

def read_npz_file(file_path):
    return np.load(file_path)
    # try:
    #     data = np.load(file_path)
    #     return data
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return None