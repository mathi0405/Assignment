# visuals.py

import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

def plot_trajectory(log_files, out_file='trajectory.png'):
    plt.figure()
    for f in log_files:
        df = pd.read_csv(f)
        plt.plot(df['x'], df['y'], label=os.path.basename(f))
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend(); plt.title('Trajectories')
    plt.savefig(out_file)

def plot_error(log_files, out_file='error.png'):
    plt.figure()
    for f in log_files:
        df = pd.read_csv(f)
        plt.plot(df['time'], df['error'], label=os.path.basename(f))
    plt.xlabel('time'); plt.ylabel('tracking error')
    plt.legend(); plt.title('Error vs Time')
    plt.savefig(out_file)

def plot_controls(log_files, out_file='controls.png'):
    plt.figure()
    for f in log_files:
        df = pd.read_csv(f)
        plt.plot(df['time'], df['v'], label=f"{os.path.basename(f)} v")
        plt.plot(df['time'], df['w'], label=f"{os.path.basename(f)} w")
    plt.xlabel('time'); plt.ylabel('control')
    plt.legend(); plt.title('Control Signals')
    plt.savefig(out_file)

def plot_costs(log_files, out_file='costs.png'):
    labels, costs = [], []
    for f in log_files:
        df = pd.read_csv(f)
        stage = df['stage_cost'].sum()    if 'stage_cost' in df else 0
        term  = df['terminal_cost'].iloc[-1] if 'terminal_cost' in df else 0
        labels.append(os.path.basename(f))
        costs.append(stage + term)
    plt.figure()
    plt.bar(labels, costs)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Total Cost'); plt.title('Cost Comparison')
    plt.tight_layout()
    plt.savefig(out_file)
def main():
    parser = argparse.ArgumentParser(description="Generate plots from log CSVs.")
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory containing CSV logs')
    parser.add_argument('--fig_dir', type=str, default='figures', help='Directory to save figures')
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    # Gather log files
    kin = sorted(glob.glob(os.path.join(args.log_dir, 'A_kin_*.csv')))
    lqr = sorted(glob.glob(os.path.join(args.log_dir, 'B_lqr_*.csv')))
    mpc = sorted(glob.glob(os.path.join(args.log_dir, 'C_mpc_*.csv')))

    # Generate and save plots
    plot_trajectory(kin + lqr + mpc,
                    out_file=os.path.join(args.fig_dir, 'trajectory.png'))
    print(f"Saved trajectory plot to {args.fig_dir}/trajectory.png")

    plot_error(kin + lqr + mpc,
               out_file=os.path.join(args.fig_dir, 'error.png'))
    print(f"Saved error plot to {args.fig_dir}/error.png")

    plot_controls(kin + lqr + mpc,
                  out_file=os.path.join(args.fig_dir, 'controls.png'))
    print(f"Saved control signals plot to {args.fig_dir}/controls.png")

    plot_costs(mpc,
               out_file=os.path.join(args.fig_dir, 'costs.png'))
    print(f"Saved cost comparison plot to {args.fig_dir}/costs.png")


if __name__ == '__main__':
    import argparse
    main()