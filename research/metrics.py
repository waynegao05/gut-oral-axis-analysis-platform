from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def concordance_index(time, event, risk) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    concordant = 0.0
    permissible = 0.0

    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue
            if event[i] == 0 and event[j] == 0:
                continue

            if time[i] < time[j] and event[i] == 1:
                permissible += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
            elif time[j] < time[i] and event[j] == 1:
                permissible += 1
                if risk[j] > risk[i]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    if permissible == 0:
        return 0.0
    return float(concordant / permissible)
