import numpy as np
import matplotlib.pyplot as plt
from systems import unicycle_model
from controllers import mpc_stage_cost, mpc_terminal_cost

# === Simulation Parameters ===
dt = 0.1       # Time step
N = 10         # Prediction horizon
T = 50         # Total simulation steps

# === Cost Matrices ===
Q = np.diag([10, 10, 1])        # State cost
R = np.diag([0.1, 0.1])         # Input cost
Qf = np.diag([20, 20, 5])       # Terminal cost

# === Initial and Target State ===
x = np.array([0.0, 0.0, 0.0])               # Start
reference = np.array([1.0, 1.0, 0.0])       # Target
trajectory = [x.copy()]

# === MPC Controller ===
def simple_mpc_controller(x, ref, Q, R, Qf, N):
    best_cost = float("inf")
    best_u = None
    for v in np.linspace(-1.0, 1.0, 5):
        for w in np.linspace(-1.0, 1.0, 5):
            x_sim = x.copy()
            cost = 0
            for _ in range(N):
                x_sim = unicycle_model(x_sim, [v, w], dt)
                cost += mpc_stage_cost(x_sim, [v, w], ref, Q, R)
            cost += mpc_terminal_cost(x_sim, ref, Qf)
            if cost < best_cost:
                best_cost = cost
                best_u = [v, w]
    return np.array(best_u)

# === Simulation Loop ===
for _ in range(T):
    u = simple_mpc_controller(x, reference, Q, R, Qf, N)
    x = unicycle_model(x, u, dt)
    trajectory.append(x.copy())

trajectory = np.array(trajectory)

# === Plot Result ===
plt.plot(trajectory[:, 0], trajectory[:, 1], label="MPC Trajectory")
plt.plot(reference[0], reference[1], 'ro', label="Reference Point")
plt.title("MPC Controller Trajectory")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
