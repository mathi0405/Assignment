import numpy as np
import matplotlib.pyplot as plt
from controllers import lqr_controller

# Discrete-time system
A = np.array([[1, 1], [0, 1]])
B = np.array([[0], [1]])
Q = np.eye(2)
R = np.array([[0.1]])

# Compute LQR gain
K = lqr_controller(A, B, Q, R)

# Initial state and goal
x = np.array([[10], [0]])
goal = np.array([[0], [0]])

trajectory = [x.flatten()]
errors = []
controls = []

# Simulate system
for _ in range(50):
    u = -K @ (x - goal)
    x = A @ x + B @ u

    controls.append(u.item())
    errors.append(np.linalg.norm(x - goal))
    trajectory.append(x.flatten())

trajectory = np.array(trajectory)

# === Plot State Trajectories ===
plt.figure()
plt.plot(trajectory[:, 0], label='Position')
plt.plot(trajectory[:, 1], label='Velocity')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.title('LQR Controlled State Evolution')
plt.legend()
plt.grid(True)
plt.savefig("lqr_trajectory.png")

# === Plot Tracking Error ===
plt.figure()
plt.plot(errors)
plt.xlabel('Time Step')
plt.ylabel('Tracking Error')
plt.title('LQR Tracking Error')
plt.grid(True)
plt.savefig("lqr_error.png")

# === Plot Control Inputs ===
plt.figure()
plt.plot(controls)
plt.xlabel('Time Step')
plt.ylabel('Control Input (u)')
plt.title('LQR Control Input')
plt.grid(True)
plt.savefig("lqr_controls.png")

plt.show()
