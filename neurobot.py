import numpy as np
import neurobot_cffi 

class Neurobot():
    def __init__(self, pwm_max=20):
        self._nb = neurobot_cffi.lib
        self._nb.set_pwm_max(pwm_max)
        self.DT = self._nb.DT

    def __enter__(self):
        self._nb.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_val, exc_tb)
        else:
            self._nb.print_final_time()

        self._nb.cleanup()

    def event_loop(self):
        while not self._nb.g_please_die_kthxbai:
            yield self._nb.get_current_time()
            self._nb.synchronize_loop()

    def read_adcs(self):
        return np.array([self._nb.read_adc(i) for i in range(4)])

    def apply_actuators(self, control):
        for i,c in enumerate(control):
            self._nb.apply_actuator(i, c)

