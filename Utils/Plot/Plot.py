import numpy as np
import matplotlib.pyplot as plt

def plot_signal_and_prediction(signal,
                               prediction,
                               title,
                               time = (0,10),
                               signal_label="Signal",
                               prediction_label="Prediction"):
    """
    Plot a signal and its corresponding prediction side by side using subplots.

    Args:
    signal (array-like): The original signal data.
    prediction (array-like): The predicted signal data.
    title (str): The title of the plot.
    signal_label (str): Label for the original signal.
    prediction_label (str): Label for the prediction.

    Returns:
    None
    """
    # Create a time array (assuming the signals are sampled evenly)
    time = np.linspace(time[0], time[1], (time[1]-time[0])*10)


    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(1, 1, figsize=(15, 5),dpi = 300)

    # Plot the original signal
    axs.plot(time, signal, label=signal_label, color='blue')#,linestyle='dotted'
    axs.plot(time, prediction, label=prediction_label, color='orange')
    axs.set_title("Original Signal")
    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Amplitude")
    axs.legend()

    # Add a title for the entire plot
    fig.suptitle(title)

    # Adjust layout to prevent overlap of labels
    plt.tight_layout()
    
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()



def zero_meaner(inp_vec,indexer = 250):
    for i in range(inp_vec.shape[0]//indexer):
        inp_vec[i*indexer:(i+1)*indexer] -= inp_vec[i*indexer:(i+1)*indexer].mean()
    return inp_vec

def integrate_and_plot( vector,
                        Result,
                        dt=0.1,
                        time = (0,10),
                                                
                        ):
    vector = zero_meaner(vector)
    # V = np.cumsum((vector - vector.mean())* dt) 
    V = np.cumsum((vector)* dt) 
    V = zero_meaner(V)

    # X = np.cumsum((V-V.mean())* dt) 
    X = np.cumsum((V)* dt) 
    X = zero_meaner(X)
    
    time = np.linspace(time[0], time[1], (time[1]-time[0])*10)
    fig, axs = plt.subplots(4,1,figsize=(10,25),dpi=300)

    # Plot the original signal
    axs[0].plot(time, vector, label="a_label", color='blue')
    axs[0].set_title("Acceleration")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # Plot the prediction
    axs[1].plot(time, V, label="v_label", color='blue')
    axs[1].set_title("Velocity")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    axs[2].plot(time, X, label="x_label", color='blue')
    axs[2].set_title("Position")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Amplitude")
    axs[2].legend()

    axs[3].plot(X,V, label="X_label", color='blue')
    axs[3].set_title("Phase Plane")
    axs[3].set_xlabel("Position")
    axs[3].set_ylabel("Velocity")
    axs[3].legend()
    
    fig.suptitle(Result)

    # Adjust layout to prevent overlap of labels
    fig.tight_layout()
    
    fig.savefig(f"{Result}.png", dpi=300, bbox_inches='tight')

    # Display the plot
    fig.show()