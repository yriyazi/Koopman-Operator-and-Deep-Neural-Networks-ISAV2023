import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import dataloaders

# Parameters
δ = 0.3    # Damping coefficient
α = -1.0    # Linear stiffness coefficient
β = 1   # Non-linear stiffness coefficient
ω = 1.2    # Angular frequency of the external driving force

γ = 0.5    # Amplitude of the external driving force

# Initial conditions
x0 = 1.0   # Initial displacement
v0 = 0.0   # Initial velocity

t_span = (0, 500)
dt = 0.01

# Numerically solve the Duffing equation using Runge-Kutta method
time,x, v = dataloaders.runge_kutta_solve(δ, α, β, γ, ω, x0, v0, t_span, dt)


#######################################################################################################################
#######################################################################################################################
time = np.cos(ω*time)
#######################################################################################################################
#######################################################################################################################


# Create a 3D phase space plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, v, time, c=time, cmap='plasma', label="Phase Space Trajectory")
ax.set_xlabel("Displacement (x)")
ax.set_ylabel("Velocity (v)")
ax.set_zlabel("Time")
ax.set_title("3D Phase Space of the Duffing Oscillator")
ax.legend()
ax.grid(True)

# Add a colorbar
cbar = fig.colorbar(sc, label="Time")

def update(angle):
    # ax.view_init(elev=10, azim=angle)
    pass

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), interval=100)
ani.save_count = 0  # Set save_count to 0 for interactive rotation

# Display the interactive 3D phase space plot
plt.show()