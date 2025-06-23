import csv
import numpy as np

class Logger3WRobotNI:
    def __init__(self):
        self.rows = []

    def log(self, t, state, control, goal, extra=None):
        x, y, th = state
        v, w     = control
        xg, yg, _= goal
        err      = np.hypot(x - xg, y - yg)

        row = {
            'time': t,
            'x': x, 'y': y, 'theta': th,
            'v': v, 'w': w,
            'error': err
        }
        if extra:
            row.update(extra)
        self.rows.append(row)

    def log_terminal(self, term_cost):
        if self.rows:
            self.rows[-1]['terminal_cost'] = term_cost

    def save_csv(self, filename):
        # Collect all field names from all rows to handle extra fields
        all_keys = []
        seen = set()
        for r in self.rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        # Write CSV with uniform columns
        with open(filename, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in self.rows:
                # ensure all keys present in row
                for k in all_keys:
                    if k not in r:
                        r[k] = ''
                w.writerow(r)
