#! /usr/bin/env python3
import numpy as np
from neurobot import Neurobot
import os
from itertools import chain

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
    parser.add_argument('datalog', nargs='?', default=os.devnull,
            help='optional file to datalog to')
    args = parser.parse_args()

    kp = args.proportional
    ki = args.integral
    tau = args.time_constant

    with Neurobot(pwm_max=args.pwm_max, 
            datalog=args.datalog, dt_ms=1) as nb:

        # Emit a list of the variables to be logged.
        nb.log(',A0,A1,A2,A3,C0,C1,C2,C3')

        print('Starting simulation...')
        interr = np.zeros(4)
        for t in nb.event_loop():

            pos = nb.read_adcs()

            err = pos - 0.5
            control = -kp*err - ki*interr
            interr += nb.dt_ms()/tau*(err - interr)

            # Emit the actual values of the variables.
            nb.log(f',{pos[0]},{pos[1]},{pos[2]},{pos[3]},{control[0]},{control[1]},{control[2]},{control[3]}')

            # nb.apply_actuators(control)

