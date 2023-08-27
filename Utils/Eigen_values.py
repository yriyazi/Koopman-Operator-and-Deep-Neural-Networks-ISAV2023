import torch

def Eigen_value_calculator(data:torch.tensor)->tuple:
    # Compute the eigenvalues
    eigenvalues ,eigenvectors = torch.linalg.eig(data.clone().detach())
    eigenvectors = None
    
    # Extract the real and imaginary parts of the eigenvalues
    real_parts = torch.real(eigenvalues)
    imaginary_parts = torch.imag(eigenvalues)
    
    # Compute the magnitude of the eigenvalues
    magnitudes = torch.sqrt(torch.pow(real_parts, 2) + torch.pow(imaginary_parts, 2))
    
    # Sort the eigenvalues by magnitude
    sorted_indices = torch.argsort(magnitudes, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    
    real_parts = torch.real(sorted_eigenvalues)
    imaginary_parts = torch.imag(sorted_eigenvalues)
    
    return real_parts,imaginary_parts,magnitudes