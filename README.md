# Assignment
# 3-Wheel Robot Tracking — Experiments A,B,C

## 📖 Overview
This project implements and compares three feedback control strategies for a non-holonomic 3-wheel robot (unicycle model):

1. **Kinematic Controller**  
   A closed-form polar-coordinate law:  
   \[
     v = k_\rho\,\rho,\quad
     \omega = k_\alpha\,\alpha + k_\beta\,\beta
   \]
2. **LQR Controller**  
   Discrete-time Linear Quadratic Regulator via the Riccati equation.
3. **MPC Controller**  
   Finite-horizon Model Predictive Control solved as a QP at each step.

We run three experiments:
- **A**: Sweep of kinematic gains  
- **B**: Sweep of LQR weight matrices, with and without actuator saturation  
- **C**: Sweep of MPC horizons & cost weights  

## 📁 Repository Structure

rcognita-edu-main/
├── controllers.py # Kinematic, LQR, MPC implementations
├── systems.py # 3-wheel robot model & linearization
├── simulator.py # Closed-loop simulation engine
├── loggers.py # CSV data logger
├── utilities.py # Helpers (trajectory generator, angle wrap)
├── visuals.py # Plotting routines + CLI entrypoint
├── run_benchmark.py # Batch runner for Experiments A–C
├── logs/ # Generated CSV logs (after running)
├── figures/ # Generated plots (after running)
└── README.md # ← you are here


## 🚀 Installation

```bash
git clone <your-repo-url>
cd rcognita-edu-main
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib pandas

▶️ Usage

    Run the experiments (will produce CSVs in logs/):

python run_benchmark.py \
  --dt 0.1 \
  --T 20.0 \
  --log_dir logs

Generate plots (reads logs/, writes PNGs to figures/):

    python visuals.py \
      --log_dir logs \
      --fig_dir figures

    Inspect results:

        CSV files in logs/

        Plots in figures/

📊 Sample Results


Figure 1: Robot trajectories under Kinematic, LQR, and MPC controllers.


Figure 2: Tracking error vs. time.


Figure 3: Control inputs (v, ω) vs. time.


Figure 4: Total cost (stage + terminal) for MPC runs.
🛠️ Code Details
controllers.py

    KinematicController(k_rho, k_alpha, k_beta)

    LQRController(A, B, Q, R)

    MPCController(dyn_func, N, dt, Q, R, Qf, x_ref_traj)

systems.py

    Sys3WRobotNI

        Continuous dynamics: ._state_dyn(t, x, u)

        Discrete step: step(x, u, dt)

        Linearization: linearize_discrete(dt) → (A, B)

simulator.py

    Simulator(system, controller, dt, mode, saturate, v_max, w_max)

    Supports mode='diff_eqn' | 'discr_fnc' | 'discr_prob'

loggers.py

    Logger3WRobotNI

        .log(t, state, control, goal, extra)

        .log_terminal(cost)

        .save_csv(filename)

utilities.py

    generate_circular_trajectory(x0, T, dt)

        Creates a full-circle path of length T with time step dt.

visuals.py

    plot_trajectory(...), plot_error(...), plot_controls(...), plot_costs(...)

    CLI entrypoint: python visuals.py --log_dir logs --fig_dir figures
