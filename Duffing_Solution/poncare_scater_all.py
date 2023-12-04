import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import matplotlib.colors as mcolors
adress = 'Frames=600 points=800 All=True gamma=0.37 omega=1.2 beta=1 alpha=-1.0 delta=0.3.pkl'
title = 'Poincaré Map of the Duffing Oscillator' + adress[:-4]

# Load the list from the file
with open(adress, 'rb') as file:
    mapss = pickle.load(file)
x_min, x_max = -2, 2
y_min, y_max = -2, 2




# Sample data for demonstration
num_frames = mapss[0].shape[1]
frame_scale_down = 100
num_frames = int(num_frames/frame_scale_down)

colors = np.random.rand(num_frames, 800)

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
    frame = frame*frame_scale_down
    # Get the x and y data for the current frame
    frame_text.set_text('$\phi$: {:.3f}'.format(frame*0.01/1.2/np.pi))
        
    datas = []
    for item in range(len(mapss)):
        datas.append([mapss[item][0,frame],mapss[item][1,frame]])
    # Update the scatter plot data
    scatter.set_offsets(np.array(datas))
    
    
    
    current_colors = colors[0]
    # Update the scatter plot data and colors
    scatter.set_array(current_colors)
    
    return scatter,frame_text

# Step 4: Create the animation
animation = FuncAnimation(fig, update, frames=num_frames, interval=100)
animation.save(title+".gif", writer="ffmpeg")

# Step 5: Display the animation
plt.show()
