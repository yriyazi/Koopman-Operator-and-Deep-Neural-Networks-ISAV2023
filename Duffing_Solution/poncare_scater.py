import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

adress = 'Frames=600 points=800 All=True gamma=0.37 omega=1.2 beta=1 alpha=-1.0 delta=0.3.pkl'
title = 'Poincaré Map of the Duffing Oscillator' + adress[:-4]

# Load the list from the file
with open(adress, 'rb') as file:
    mapss = pickle.load(file)
x_min, x_max = -2, 2
y_min, y_max = -2, 2


# Sample data for demonstration
num_frames = mapss[0].shape[0]
DPI = 300
# Step 1: Create a figure and axis
fig, ax = plt.subplots(figsize=(15, 10))#,dpi=DPI

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Step 2: Set up the initial scatter plot (empty in this case)
scatter = ax.scatter([], [])
# frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                     bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.4))

ax.grid(True)

# Set axis labels and title
ax.set_xlabel("Displacement (x)")
ax.set_ylabel("Velocity (v)")
ax.set_title(title)    

    
# Step 3: Define the update function for the animation
def update(frame):
    # Get the x and y data for the current frame
    frame_text.set_text('$\phi$: {}'.format(frame))
    
    
    datas = []
    for item in range(len(mapss)):
        datas.append([mapss[item][frame,0],mapss[item][frame,1]])
    # Update the scatter plot data
    scatter.set_offsets(np.array(datas))
    return scatter,frame_text

# Step 4: Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=10)
# animation.save(title+".gif", writer="ffmpeg", dpi=DPI)

# Step 5: Display the animation
plt.show()
