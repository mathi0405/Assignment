import numpy as np
import matplotlib.pyplot as plt
from controllers import kinematic_controller

# Initial and goal states
state = [0.0, 0.0, 0.0]
goal = [2.0, 2.0, 0.0]
dt = 0.1
steps = 100

trajectory = [state.copy()]
errors = []
controls = []

for _ in range(steps):
    v, w = kinematic_controller(state, goal)
    
    # Log control and error
    controls.append([v, w])
    error = np.linalg.norm(np.array(state[:2]) - np.array(goal[:2]))
    errors.append(error)

    # Simple simulation (Euler integration)
    state[0] += v * np.cos(state[2]) * dt
    state[1] += v * np.sin(state[2]) * dt
    state[2] += w * dt
    trajectory.append(state.copy())

trajectory = np.array(trajectory)
controls = np.array(controls)

# === Plot Trajectory ===
plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Robot Trajectory')
plt.plot(goal[0], goal[1], 'rx', label='Goal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kinematic Controller Trajectory')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig("kinematic_trajectory.png")

# === Plot Tracking Error ===
plt.figure()
plt.plot(errors)
plt.xlabel('Time step')
plt.ylabel('Tracking Error')
plt.title('Kinematic Controller Tracking Error')
plt.grid(True)
plt.savefig("kinematic_error.png")

# === Plot Control Inputs ===
plt.figure()
plt.plot(controls[:, 0], label='Linear Velocity (v)')
plt.plot(controls[:, 1], label='Angular Velocity (w)')
plt.xlabel('Time step')
plt.ylabel('Control Inputs')
plt.title('Kinematic Controller Control Inputs')
plt.legend()
plt.grid(True)
plt.savefig("kinematic_controls.png")

plt.show()