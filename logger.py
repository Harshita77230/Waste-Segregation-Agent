import csv
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "scans.csv")
FIELDS = ["time", "model_guess", "confidence", "final_label", "bin", "agent_asked"]


def log_scan(model_guess, confidence, final_label, bin_name, agent_asked):
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_guess, f"{confidence:.3f}", final_label, bin_name, agent_asked,
        ])