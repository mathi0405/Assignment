# run_benchmark.py

import os
import numpy as np
import argparse

from systems     import Sys3WRobotNI
from utilities   import generate_circular_trajectory
from controllers import (
    KinematicController,
    LQRController,
    MPCController,
    mpc_stage_cost,
    mpc_terminal_cost
)
from simulator   import Simulator
from loggers     import Logger3WRobotNI

def run_experiment_A(sys, x0, traj, dt, log_dir):
    print("\n=== Experiment A: Kinematic ===")
    gains = [
        (1.0, 4.0, -1.5),
        (0.5, 3.0, -1.0),
        (2.0, 5.0, -2.0)
    ]
    for kr, ka, kb in gains:
        ctl = KinematicController(kr, ka, kb)
        sim = Simulator(sys, ctl, dt, mode='diff_eqn')
        log = Logger3WRobotNI()
        sim.reset(x0)
        for k, goal in enumerate(traj):
            x, u = sim.sim_step(goal)
            log.log(k * dt, x, u, goal)
        fname = f"A_kin_{kr:.1f}_{ka:.1f}_{kb:.1f}.csv"
        dest = os.path.join(log_dir, fname)
        log.save_csv(dest)
        print(f"  • Saved {fname} → {os.path.abspath(dest)}")

def run_experiment_B(sys, x0, traj, dt, log_dir):
    print("\n=== Experiment B: LQR ===")
    A, B = sys.linearize_discrete(dt)
    QRs = [
        (np.diag([1.0, 1.0, 0.01]), np.diag([1.0, 1.0])),
        (np.diag([10.0,10.0,1.0]), np.diag([1.0, 1.0]))
    ]
    for Q, R in QRs:
        for sat in (False, True):
            ctl = LQRController(A, B, Q, R)
            sim = Simulator(sys, ctl, dt,
                            mode='discr_fnc',
                            saturate=sat,
                            v_max=1.0,
                            w_max=1.0)
            log = Logger3WRobotNI()
            sim.reset(x0)
            for k, goal in enumerate(traj):
                x, u = sim.sim_step(goal)
                log.log(k * dt, x, u, goal)
            tag = 'sat' if sat else 'nosat'
            qv  = int(Q[0,0])
            fname = f"B_lqr_Q{qv}_{tag}.csv"
            dest  = os.path.join(log_dir, fname)
            log.save_csv(dest)
            print(f"  • Saved {fname} → {os.path.abspath(dest)}")

def run_experiment_C(sys, x0, traj, dt, log_dir):
    print("\n=== Experiment C: MPC ===")
    horizons    = [5, 10, 20]
    weight_sets = [
        (np.diag([1,1,0]), np.diag([1,1]), np.diag([10,10,1]))
    ]
    for N in horizons:
        for Q, R, Qf in weight_sets:
            ctl = MPCController(
                dyn_func=sys.step,
                N=N,
                dt=dt,
                Q=Q, R=R, Qf=Qf,
                x_ref_traj=traj
            )
            sim = Simulator(sys, ctl, dt, mode='discr_fnc')
            log = Logger3WRobotNI()
            sim.reset(x0)
            for k, goal in enumerate(traj):
                x, u = sim.sim_step(goal)
                stage = mpc_stage_cost(x, u, Q, R)
                log.log(k * dt, x, u, goal, extra={'stage_cost': stage})
            # use sim.state (not sim._state) for terminal cost
            term = mpc_terminal_cost(sim.state, Qf)
            log.log_terminal(term)
            fname = f"C_mpc_N{N}.csv"
            dest  = os.path.join(log_dir, fname)
            log.save_csv(dest)
            print(f"  • Saved {fname} → {os.path.abspath(dest)}")

def main():
    p = argparse.ArgumentParser(description="Run Experiments A, B, and C.")
    p.add_argument('--dt',      type=float, default=0.1, help="Simulation time step")
    p.add_argument('--T',       type=float, default=20.0, help="Total simulation time")
    p.add_argument('--log_dir', type=str,   default='logs', help="Directory to save CSV logs")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    sys  = Sys3WRobotNI()
    x0   = sys._state
    traj = generate_circular_trajectory(x0, args.T, args.dt)

    run_experiment_A(sys, x0, traj, args.dt, args.log_dir)
    run_experiment_B(sys, x0, traj, args.dt, args.log_dir)
    run_experiment_C(sys, x0, traj, args.dt, args.log_dir)

if __name__ == '__main__':
    main()