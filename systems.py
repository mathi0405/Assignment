#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains a generic interface for systems (environments) as well as concrete systems as realizations of the former

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
from numpy.random import randn

class System:
    """
    Interface class of dynamical systems a.k.a. environments.
    Concrete systems should be built upon this class.
    To design a concrete system: inherit this class, override:
        | :func:`~systems.system._state_dyn` :
        | right-hand side of system description (required)
        | :func:`~systems.system._disturb_dyn` :
        | right-hand side of disturbance model (if necessary)
        | :func:`~systems.system._ctrl_dyn` :
        | right-hand side of controller dynamical model (if necessary)
        | :func:`~systems.system.out` :
        | system out (if not overridden, output is identical to state)
      
    Attributes
    ----------
    sys_type : : string
        Type of system by description:
            
        | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, action, disturb)`
        | ``discr_fnc`` : difference equation :math:`state^+ = f(state, action, disturb)`
        | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, action, disturb)`
    
    where:
        
        | :math:`state` : state
        | :math:`action` : input
        | :math:`disturb` : disturbance
        
    The time variable ``t`` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is non-autonomous.
    For the latter case, however, you already have the input and disturbance at your disposal.
    
    Parameters of the system are contained in ``pars`` attribute.
    
    dim_state, dim_input, dim_output, dim_disturb : : integer
        System dimensions 
    pars : : list
        List of fixed parameters of the system
    ctrl_bnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default)
    is_dyn_ctrl : : 0 or 1
        If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
    is_disturb : : 0 or 1
        If 0, no disturbance is fed into the system
    pars_disturb : : list
        Parameters of the disturbance model
        
   Each concrete system must realize ``System`` and define ``name`` attribute.   
        
    """
    def __init__(self,
                 sys_type,
                 dim_state,
                 dim_input,
                 dim_output,
                 dim_disturb,
                 pars=[],
                 ctrl_bnds=[],
                 is_dyn_ctrl=0,
                 is_disturb=0,
                 pars_disturb=[]):
        
        """
        Parameters
        ----------
        sys_type : : string
            Type of system by description:
                
            | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, action, disturb)`
            | ``discr_fnc`` : difference equation :math:`state^+ = f(state, action, disturb)`
            | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, action, disturb)`
        
        where:
            
            | :math:`state` : state
            | :math:`action` : input
            | :math:`disturb` : disturbance
            
        The time variable ``t`` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is non-autonomous.
        For the latter case, however, you already have the input and disturbance at your disposal.
        
        Parameters of the system are contained in ``pars`` attribute.
        
        dim_state, dim_input, dim_output, dim_disturb : : integer
            System dimensions 
        pars : : list
            List of fixed parameters of the system
        ctrl_bnds : : array of shape ``[dim_input, 2]``
            Box control constraints.
            First element in each row is the lower bound, the second - the upper bound.
            If empty, control is unconstrained (default)
        is_dyn_ctrl : : 0 or 1
            If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
        is_disturb : : 0 or 1
            If 0, no disturbance is fed into the system
        pars_disturb : : list
            Parameters of the disturbance model        
        """
        
        self.sys_type = sys_type
        
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_disturb = dim_disturb   
        self.pars = pars
        self.ctrl_bnds = ctrl_bnds
        self.is_dyn_ctrl = is_dyn_ctrl
        self.is_disturb = is_disturb
        self.pars_disturb = pars_disturb
        
        # Track system's state
        self._state = np.zeros(dim_state)
        
        # Current input (a.k.a. action)
        self.action = np.zeros(dim_input)
        
        if is_dyn_ctrl:
            if is_disturb:
                self._dim_full_state = self.dim_state + self.dim_disturb + self.dim_input
            else:
                self._dim_full_state = self.dim_state
        else:
            if is_disturb:
                self._dim_full_state = self.dim_state + self.dim_disturb
            else:
                self._dim_full_state = self.dim_state
            
    def _state_dyn(self, t, state, action, disturb):
        """
        Description of the system internal dynamics.
        Depending on the system type, may be either the right-hand side of the respective differential or difference equation, or a probability distribution.
        As a probability disitribution, ``_state_dyn`` should return a number in :math:`[0,1]`
        
        """
        pass

    def _disturb_dyn(self, t, disturb):
        """
        Dynamical disturbance model depending on the system type:
            
        | ``sys_type = "diff_eqn"`` : :math:`\mathcal D disturb = f_q(disturb)`    
        | ``sys_type = "discr_fnc"`` : :math:`disturb^+ = f_q(disturb)`
        | ``sys_type = "discr_prob"`` : :math:`disturb^+ \sim P_Q(disturb^+|disturb)`
        
        """       
        pass

    def _ctrl_dyn(self, t, action, observation):
        """
        Dynamical controller. When ``is_dyn_ctrl=0``, the controller is considered static, which is to say that the control actions are
        computed immediately from the system's output.
        In case of a dynamical controller, the system's state vector effectively gets extended.
        Dynamical controllers have some advantages compared to the static ones.
        
        Depending on the system type, can be:
            
        | ``sys_type = "diff_eqn"`` : :math:`\mathcal D action = f_u(action, observation)`    
        | ``sys_type = "discr_fnc"`` : :math:`action^+ = f_u(action, observation)`  
        | ``sys_type = "discr_prob"`` : :math:`action^+ \sim P_U(action^+|action, observation)`        
        
        """
        Daction = np.zeros(self.dim_input)
    
        return Daction 

    def out(self, state, action=[]):
        """
        System output.
        This is commonly associated with signals that are measured in the system.
        Normally, output depends only on state ``state`` since no physical processes transmit input to output instantly.       
        
        See also
        --------
        :func:`~systems.system._state_dyn`
        
        """
        # Trivial case: output identical to state
        observation = state
        return observation
    
    def receive_action(self, action):
        """
        Receive exogeneous control action to be fed into the system.
        This action is commonly computed by your controller (agent) using the system output :func:`~systems.system.out`. 

        Parameters
        ----------
        action : : array of shape ``[dim_input, ]``
            Action
            
        """
        self.action = action
        
    def closed_loop_rhs(self, t, state_full):
        """
        Right-hand side of the closed-loop system description.
        Combines everything into a single vector that corresponds to the right-hand side of the closed-loop system description for further use by simulators.
        
        Attributes
        ----------
        state_full : : vector
            Current closed-loop system state        
        
        """
        rhs_full_state = np.zeros(self._dim_full_state)
        
        state = state_full[0:self.dim_state]
        
        if self.is_disturb:
            disturb = state_full[self.dim_state:]
        else:
            disturb = []
        
        if self.is_dyn_ctrl:
            action = state_full[-self.dim_input:]
            observation = self.out(state)
            rhs_full_state[-self.dim_input:] = self._ctrlDyn(t, action, observation)
        else:
            # Fetch the control action stored in the system
            action = self.action
        
        if self.ctrl_bnds.any():
            for k in range(self.dim_input):
                action[k] = np.clip(action[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])
        
        rhs_full_state[0:self.dim_state] = self._state_dyn(t, state, action, disturb)
        
        if self.is_disturb:
            rhs_full_state[self.dim_state:] = self._disturb_dyn(t, disturb)
        
        # Track system's state
        self._state = state
        
        return rhs_full_state    
    
from systems import System  # make sure System is imported at top

class Sys3WRobotNI(System):
    def __init__(self,
                 sys_type    = 'discr_fnc',
                 dim_state   = 3,
                 dim_input   = 2,
                 dim_output  = 3,
                 dim_disturb  = 0,
                 pars        = None,
                 ctrl_bnds   = None,
                 is_dyn_ctrl = 0,
                 is_disturb   = 0,
                 pars_disturb= None):
        """
        A 3-wheel robot (nonholonomic integrator) with convenient defaults
        so you can just call Sys3WRobotNI() without args.
        """
        super().__init__(
            sys_type,
            dim_state,
            dim_input,
            dim_output,
            dim_disturb,
            pars or [],
            ctrl_bnds or [],
            is_dyn_ctrl,
            is_disturb,
            pars_disturb or []
        )

        self.name = '3wrobotNI'

        if self.is_disturb:
            self.sigma_disturb = self.pars_disturb[0]
            self.mu_disturb    = self.pars_disturb[1]
            self.tau_disturb   = self.pars_disturb[2]

    # … keep _state_dyn, _disturb_dyn, out, and step as before
    
    def _state_dyn(self, t, state, action, disturb=[]):   
        Dstate = np.zeros(self.dim_state)
        # Write the unicycle model here
        v, omega = action
        theta = state[2]
        Dstate[0] = v * np.cos(theta)
        Dstate[1] = v * np.sin(theta)
        Dstate[2] = omega

        return Dstate    
 
    def _disturb_dyn(self, t, disturb):
        """
        
        
        """       
        Ddisturb = np.zeros(self.dim_disturb)
        
        for k in range(0, self.dim_disturb):
            Ddisturb[k] = - self.tau_disturb[k] * ( disturb[k] + self.sigma_disturb[k] * (randn() + self.mu_disturb[k]) )
                
        return Ddisturb   
    
    def out(self, state, action=[]):
        observation = np.zeros(self.dim_output)
        observation = state
        return observation
    def step(self, state, action, dt):
        # Unicycle discrete-time model
        x_new = np.zeros_like(state)
        x_new[0] = state[0] + action[0] * np.cos(state[2]) * dt  # x
        x_new[1] = state[1] + action[0] * np.sin(state[2]) * dt  # y
        x_new[2] = state[2] + action[1] * dt                     # theta
        return x_new
    
    def linearize_discrete(self, dt: float):
        """
        Compute the discrete‐time linearization of the unicycle
        about the current state and zero input:
          x_{k+1} = f(x_k, u_k)
        Returns:
          A, B  where
            A = ∂f/∂x |_{x,0}  ≃ I + A_c * dt
            B = ∂f/∂u |_{x,0}  dt
        """
        theta = self._state[2]
        # continuous‐time Jacobians at u=0
        A_c = np.array([
            [0, 0, 0],   # since v=0 → ∂(v cosθ)/∂θ = 0
            [0, 0, 0],   # likewise for ∂(v sinθ)/∂θ
            [0, 0, 0]
        ])
        # discretize
        A = np.eye(3) + A_c * dt
        B = np.array([
            [np.cos(theta) * dt, 0.0],
            [np.sin(theta) * dt, 0.0],
            [0.0,                 dt  ]
        ])
        return A, B

def unicycle_model(x, u, dt):
    """
    Unicycle model for mobile robot.

    x: current state [x, y, theta]
    u: control input [v, omega]
    dt: time step
    """
    x_new = np.zeros_like(x)
    x_new[0] = x[0] + u[0] * np.cos(x[2]) * dt
    x_new[1] = x[1] + u[0] * np.sin(x[2]) * dt
    x_new[2] = x[2] + u[1] * dt
    return x_new