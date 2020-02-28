#! /usr/bin/env python3
import os, sys

import numpy as np

from neurobot import Neurobot
from cpgwalker import *

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
            description='Run the default forward CPG.')
    parser.add_argument('-p', '--pwm-max', default=20, type=float,
            help='maximum PWM percentage')
    parser.add_argument('datalog', nargs='?', default=os.devnull,
            help='optional file to datalog to')
    parser.add_argument('-k', '--feedback', default=5, type=float,
            help='strength of the proprioceptive feedback current')
    args = parser.parse_args()

    cpg = SingleCPG()

    print(cpg.V.dtype)
    sys.exit(0)

    with Neurobot(pwm_max=args.pwm_max, 
            datalog=args.datalog, dt_ms=0.5) as nb:
        # Emit a list of the variables to be logged: all four actuator
        # positions, twelve neuron voltages, and four muscle voltages.
        nb.log(',A0,A1,A2,A3,V0,V1,V2,V3,V4,V5,V6,V7')
        nb.log(',V8,V9,V10,V11,M0,M1,M2,M3')
        print('Starting simulation...', file=sys.stderr)

        for t in nb.event_loop():
            pos = nb.read_adcs()

            cpg.step(dt=nb.dt, pos=pos)

            # Actually log the variables.
            nb.log(f',{pos[0]},{pos[1]},{pos[2]},{pos[3]}'
                   f',{cpg.V[0]},{cpg.V[1]},{cpg.V[2]},{cpg.V[3]}'
                   f',{cpg.V[4]},{cpg.V[5]},{cpg.V[6]},{cpg.V[7]}'
                   f',{cpg.V[8]},{cpg.V[9]},{cpg.V[10]},{cpg.V[11]}'
                   f',{cpg.V[12]},{cpg.V[13]},{cpg.V[14]},{cpg.V[15]}')

