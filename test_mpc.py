import numpy as np
from controllers import mpc_stage_cost, mpc_terminal_cost

# Define example state and input
x = np.array([[1.0], [0.5]])
u = np.array([[0.1]])

Q = np.eye(2)
R = np.array([[0.01]])
Qf = np.eye(2) * 10

print("Stage cost:", mpc_stage_cost(x, u, Q, R))
print("Terminal cost:", mpc_terminal_cost(x, Qf))