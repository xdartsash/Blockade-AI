"""Bounded, escalating recovery; no game dependencies, clock supplied by caller."""
class RecoveryPlanner:
    def __init__(self):
        self.last_time = None
        self.attempt = 0

    def next(self, now, base_turn, base_hold):
        if self.last_time is None or now - self.last_time > 20:
            self.attempt = 0
        self.last_time = now
        # Repeated failures change both turn size and side, instead of undoing
        # the previous identical turn forever. This is a heuristic, not mapping.
        turns = (1.0, -1.5, -2.0, 2.5)
        i = min(self.attempt, 3)
        self.attempt += 1
        return int(base_turn * turns[i % 4] * (-1 if self.attempt > 4 and self.attempt % 2 else 1)), min(3.0, base_hold * (1 + i * .4))
