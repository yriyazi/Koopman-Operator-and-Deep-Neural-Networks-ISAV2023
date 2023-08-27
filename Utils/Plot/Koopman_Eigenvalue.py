import os
import torch
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec

def Koopman_Eigenvalue(
                       real_parts           : torch.tensor,
                       imaginary_parts      : torch.tensor,
                       magnitudes           : torch.tensor,
                       _index               : int = 256,
                       DPI                  : int = 400,
                       save_image           : bool= None,
                       neural_net_name      : str= None,
                       dataset_parameters   : str= None,
                       save_dir             : str = os.path.join('utils','Plot','Koopman_Eigenvalues')
                        )->None:
    
    real_parts_cpu      = real_parts.cpu().numpy()
    imaginary_parts_cpu = imaginary_parts.cpu().numpy()
    magnitudes_cpu      = magnitudes.cpu().numpy()
    
    fig , ax = plt.subplots(1,figsize=(15,10) , dpi=DPI)
    
    sc = ax.scatter(real_parts_cpu[:_index], imaginary_parts_cpu[:_index], c=magnitudes_cpu[:_index], cmap='viridis', marker='o')
    plt.colorbar(sc, label="Magnitude")
       
    ax.hlines(y=0,xmin=real_parts_cpu.min()     ,xmax=real_parts_cpu.max()      , color='black', linestyle='dashed')
    ax.vlines(x=0,ymin=imaginary_parts_cpu.min(),ymax=imaginary_parts_cpu.max() , color='black', linestyle='dashed')
    ax.set_xlabel('Real part ($\mu$)')
    ax.set_ylabel('Imaginary part ($\mu$)')
    ax.set_title(f"neural net = {neural_net_name} dataset parameters ={dataset_parameters} index={_index}")
    plt.grid(True)
    
    if save_image:
            plt.savefig(os.path.join(save_dir,f"neural net = {neural_net_name} dataset parameters ={dataset_parameters} index={_index}.png"), bbox_inches='tight')
            
            
    plt.show()