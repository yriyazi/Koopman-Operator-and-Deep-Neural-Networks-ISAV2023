import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def plot_XandV_t(model_name : str,
                 
                 time, 
                 displacement,
                 velocity,
                 
                 delta  :int,
                 alpha  :int,
                 beta   :int,
                 gamma  :int,
                 omega  :int,
                 x_lim  = None,
                 
                 save_iamge : bool = False,
                 
                 x_grid :int = 0.1,
                 y_grid :int = 0.5,
                 DPI    :int = 400)->None:
    
    
        # Create a 1x2 grid of subplots with specified width ratios
        fig = plt.figure(figsize=(20,10) , dpi=DPI)
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 3])
        # Plot phase space on the first subplot (larger width)
        ax1 = plt.subplot(gs[0])
        # Plot velocity vs. time on the second subplot (smaller width)
        ax2 = plt.subplot(gs[1])



        # fig , (ax1, ax2) = plt.subplots(nrows = 1 ,ncols = 2,figsize=(20,10) , dpi=DPI)
        fig.suptitle(f"Duffing Oscillator (δ={delta}, α={alpha}, β={beta}, γ={gamma}, ω={omega})",y=0.95 , fontsize=20)
        ax1.plot(time, displacement , label="Displacement")
        ax1.plot(time, velocity     , label="Velocity")
        
        if not x_lim:
            ax1.set_xlim((0,x_lim))
            
        ax1.set_xlabel("Time"            ,fontsize=15)
        ax1.set_ylabel("Amplitude"       ,fontsize=15)
        
        if x_grid:
            ax1.grid(axis="x",alpha=0.1)
        if y_grid:
            ax1.grid(axis="y",alpha=0.5)
        
        ax1.legend(loc=0,prop={"size":9})
        
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        
        
        

        
        # Plot phase space
        sc = ax2.scatter(displacement, velocity, c=time, cmap='plasma', label="Phase Space Trajectory")
        plt.colorbar(sc, label="Time")

        ax2.set_title('Phase space')
        # ax2.plot(displacement, velocity, label="Phase Space Trajectory", color='b')
        ax2.scatter(displacement[0], velocity[0], color='r', label='Initial Condition')
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_xlabel("Displacement (x)")
        ax2.set_ylabel("Velocity (v)")
        ax1.set_title("Phase Space of the Duffing Oscillator")
        ax2.legend()
        ax2.grid(True)       
            
        if save_iamge:
            plt.savefig(os.path.join('results',model_name+'.png'), bbox_inches='tight')
        
        plt.show()