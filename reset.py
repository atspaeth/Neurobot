#! /usr/bin/env python3
import numpy as np
from neurobot import Neurobot

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
            description='Zero out the actuator positions.')
    parser.add_argument('-p', '--pwm-max', default=20, type=float,
            help='maximum PWM percentage')
    parser.add_argument('-k', '--proportional', default=3, type=float,
            help='proportional control gain')
    parser.add_argument('-i', '--integral', default=6, type=float,
            help='integral control gain')
    parser.add_argument('-t', '--time-constant', default=1000,
            type=float, help='integral decay time constant (ms)')
    parser.add_argument('logfile', nargs='?', 
            help='optional file to log to')
    args = parser.parse_args()

    kp = args.proportional
    ki = args.integral
    tau = args.time_constant

    with Neurobot(pwm_max=args.pwm_max) as nb:

        interr = np.zeros(4)

        for t in nb.event_loop():
            pos = nb.read_adcs()

            err = pos - 0.5
            control = -kp*err - ki*interr
            interr += nb.DT/tau*(err - interr)

            # nb.apply_actuators(control)

            if int(t/nb.DT) % 1000 == 0:
                print(f'At time {t}, position = {pos}')
                print(f'  Control: {control}')
                print(f'  Error:   {err}')
                print(f'  Int Err: {interr}')

