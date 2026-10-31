"""Thread-safe velocity command input for the native MuJoCo viewer."""

from threading import Lock

from .config import Stage


class KeyboardVelocityControl:
    """One key press changes the requested body-frame velocity by one increment."""

    def __init__(self, stage: Stage):
        self._stage = stage
        self._lock = Lock()
        self._command = (0.0, 0.0, 0.0)

    def on_key(self, key: int) -> None:
        delta = {
            ord("W"): (0.1, 0.0, 0.0),
            ord("S"): (-0.1, 0.0, 0.0),
            ord("J"): (0.0, 0.1, 0.0),
            ord("L"): (0.0, -0.1, 0.0),
            ord("Q"): (0.0, 0.0, 0.15),
            ord("E"): (0.0, 0.0, -0.15),
        }.get(key)
        if delta is None and key != ord("X"):
            return
        with self._lock:
            if key == ord("X"):
                self._command = (0.0, 0.0, 0.0)
            else:
                vx, vy, yaw = self._command
                self._command = (
                    round(min(self._stage.forward, max(self._stage.backward, vx + delta[0])), 3),
                    round(min(self._stage.lateral, max(-self._stage.lateral, vy + delta[1])), 3),
                    round(min(self._stage.yaw, max(-self._stage.yaw, yaw + delta[2])), 3),
                )
            print(f"keyboard command (body frame): vx={self._command[0]:+.2f} m/s, "
                  f"vy={self._command[1]:+.2f} m/s, yaw={self._command[2]:+.2f} rad/s", flush=True)

    def snapshot(self) -> tuple[float, float, float]:
        with self._lock:
            return self._command
