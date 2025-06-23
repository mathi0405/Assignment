import numpy as np
from scipy.integrate import odeint

class Simulator:
    def __init__(self,
                 system,
                 controller,
                 dt,
                 mode='diff_eqn',
                 saturate=False,
                 v_max=None,
                 w_max=None):
        """
        system    : instance of Sys3WRobotNI
        controller: implements compute_control(state, goal)
        dt        : simulation time step
        mode      : 'diff_eqn' | 'discr_fnc' | 'discr_prob'
        saturate  : whether to clamp control inputs
        v_max     : max linear velocity
        w_max     : max angular velocity
        """
        self.sys        = system
        self.controller = controller
        self.dt         = dt
        self.mode       = mode
        self.saturate   = saturate
        self.v_max      = v_max
        self.w_max      = w_max
        self.state      = None

    def reset(self, x0):
        """
        Reset the simulator state to initial condition x0.
        """
        self.state = np.array(x0, dtype=float)

    def sim_step(self, goal):
        """
        Perform one simulation step towards the given goal state.

        Returns
        -------
        state : ndarray
            Updated state after the step.
        control : ndarray
            Control input [v, w] applied at this step.
        """
        # 1) compute control input
        u = self.controller.compute_control(self.state, goal)
        v, w = u

        # 2) apply actuator saturation if enabled
        if self.saturate and self.v_max is not None and self.w_max is not None:
            v = np.clip(v, -self.v_max, self.v_max)
            w = np.clip(w, -self.w_max, self.w_max)
            u = np.array([v, w], dtype=float)

        # 3) propagate system dynamics
        if self.mode == 'diff_eqn':
            # Continuous-time integration over dt
            tspan = [0, self.dt]
            # Call _state_dyn(t, state, action)
            sol = odeint(lambda state, t: self.sys._state_dyn(t, state, u),
                         self.state,
                         tspan)
            self.state = sol[-1]
        elif self.mode == 'discr_fnc':
            # Discrete update function (requires dt argument)
            self.state = self.sys.step(self.state, u, self.dt)
        elif self.mode == 'discr_prob':
            # Probabilistic discrete step (also uses dt)
            self.state = self.sys.step(self.state, u, self.dt)
            # Probabilistic discrete step with dt
            self.state = self.sys.step(self.state, u, self.dt)
        else:
            raise ValueError(f"Unknown simulation mode '{self.mode}'")

        return self.state.copy(), np.array(u, dtype=float)