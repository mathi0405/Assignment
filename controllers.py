import numpy as np
from scipy.linalg import solve_discrete_are, LinAlgError
from scipy.optimize import minimize

class KinematicController:
    def __init__(self, k_rho, k_alpha, k_beta):
        self.k_rho   = k_rho
        self.k_alpha = k_alpha
        self.k_beta  = k_beta

    def compute_control(self, state, goal):
        x, y, theta       = state
        x_goal, y_goal, th_goal = goal

        dx  = x_goal - x
        dy  = y_goal - y
        rho = np.hypot(dx, dy)

        alpha = np.arctan2(dy, dx) - theta
        beta  = th_goal - np.arctan2(dy, dx)

        alpha = (alpha + np.pi) % (2*np.pi) - np.pi
        beta  = (beta  + np.pi) % (2*np.pi) - np.pi

        v = self.k_rho   * rho
        w = self.k_alpha * alpha + self.k_beta * beta

        return np.array([v, w], dtype=float)

class LQRController:
    def __init__(self, A, B, Q, R, x_ref=None, u_ref=None):
        # Solve discrete-time Algebraic Riccati equation
        try:
            P = solve_discrete_are(A, B, Q, R)
            self.K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        except LinAlgError:
            # Fallback: zero gain
            print("Warning: discrete LQR ARE failed, using zero-gain fallback.")
            self.K = np.zeros((B.shape[1], A.shape[0]))
        self.x_ref = x_ref if x_ref is not None else np.zeros((A.shape[0],))
        self.u_ref = u_ref if u_ref is not None else np.zeros((B.shape[1],))

    def compute_control(self, state, goal=None):
        x_err = state - self.x_ref
        return -self.K @ x_err + self.u_ref


def mpc_stage_cost(x, u, Q, R):
    return float(x.T @ Q @ x + u.T @ R @ u)


def mpc_terminal_cost(x, Qf):
    return float(x.T @ Qf @ x)

class MPCController:
    def __init__(self,
                 dyn_func,
                 N, dt,
                 Q, R, Qf,
                 x_ref_traj=None,
                 u_ref_traj=None):
        self.f       = dyn_func
        self.N       = N
        self.dt      = dt
        self.Q, self.R, self.Qf = Q, R, Qf
        self.x_ref = x_ref_traj
        self.u_ref = u_ref_traj
        self.nu = 2

    def compute_control(self, state, goal):
        u0 = np.zeros(self.N * self.nu)

        def cost(u_seq):
            u_seq = u_seq.reshape(self.N, self.nu)
            xk = state.copy()
            total = 0.0
            for k in range(self.N):
                uk      = u_seq[k]
                x_ref_k = self.x_ref[k] if self.x_ref is not None else goal
                u_ref_k = self.u_ref[k] if self.u_ref is not None else np.zeros(self.nu)
                total += (xk - x_ref_k).T @ self.Q @ (xk - x_ref_k)
                total += (uk - u_ref_k).T @ self.R @ (uk - u_ref_k)
                # discrete update with dt
                xk = self.f(xk, uk, self.dt)
            x_ref_N = self.x_ref[self.N] if self.x_ref is not None else goal
            total += (xk - x_ref_N).T @ self.Qf @ (xk - x_ref_N)
            return total

        res = minimize(cost, u0, method='L-BFGS-B')
        u_opt = res.x.reshape(self.N, self.nu)[0]
        return u_opt
