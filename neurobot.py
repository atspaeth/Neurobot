import numpy as np
import neurobot_cffi 

class Neurobot():
    def __init__(self, dt_ms=None, pwm_max=20, datalog=None):
        self._nb = neurobot_cffi.lib
        self._nb.set_pwm_max(pwm_max)

        if dt_ms is not None:
            self._nb.g_dt_us = int(dt_ms * 1e3)

        if datalog is None:
            self._logfile = None
            self.log = lambda _: 0
        else:
            self._logfile = open(datalog, 'w')
            self.log = self._logfile.write

        self.dt_ms = self._nb.dt_ms

    def __enter__(self):
        self.log('t')
        self._nb.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_val, exc_tb)
        else:
            self._nb.print_final_time()

        if self.log('\n'):
            self._logfile.close()

        self._nb.cleanup()

    def event_loop(self):
        while not self._nb.g_please_die_kthxbai:
            t = self._nb.get_current_time()
            # self.log(f'\n{t}')
            yield self._nb.get_current_time()
            self._nb.synchronize_loop()

    def read_adcs(self):
        return np.array([self._nb.read_adc(i) for i in range(4)])

    def apply_actuators(self, control):
        for i,c in enumerate(control):
            self._nb.apply_actuator(i, c)

