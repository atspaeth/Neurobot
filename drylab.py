import numpy as np
from functools import partial


# A map from neuron type abbreviation to ordered list of parameters
# a, b, c, d, C, k, Vr, Vt, Vp, Vn, and tau from Dynamical Systems in
# Neuroscience.  NB: many of these models have some extra bonus
# features in the book, used to more accurately reproduce traces from
# electrophysiological experiments in the appropriate model
# organisms. In particular,
#  - LTS caps the value of u but (along with a few other types) allows
#     it to influence the effective value of spike threshold and c.
#  - Several other types have PWL u nullclines.
NEURON_TYPES = {
    'rs':  [0.03, -2, -50, 100, 100, 0.7, -60, -40, 35,   0,  5],
    'ib':  [0.01,  5, -56, 130, 150, 1.2, -75, -45, 50,   0,  5],
    'ch':  [0.03,  1, -40, 150,  50, 1.5, -60, -40, 25,   0,  5],
    'lts': [0.03,  8, -53,  20, 100, 1.0, -56, -42, 20, -70, 20],
    'ls':  [0.17,  5, -45, 100,  20, 0.3, -66, -40, 30, -70, 20]}


class Organoid():
    """
    A simulated 2D culture of cortical cells using models from
    Dynamical Systems in Neuroscience, with synapses implemented as
    exponential PSPs for both excitatory and inhibitory cells.

    The model represents the excitability of a neuron using three
    phase variables: the membrane voltage v : mV, the "recovery" or
    "leakage" current u : pA, and the synaptic activation at each
    cell, a unitless A.

    The synapses in the updated model are conductance-type, with
    synaptic conductance following the alpha function of the synaptic
    actiavtion times the peak conductance G[i,j]. This adds another
    parameter to each cell: the Nernst reversal potential Vn of its
    neurotransmitter. Synaptic activation pulls the membrane voltage
    of the postsynaptic cell towards the reversal potential of the
    presynaptic cell.

    Additionally, the excitation model contains the following static
    parameters, on a per cell basis by providing arrays of size (N,):
     a : 1/ms time constant of recovery current
     b : nS steady-state conductance for recovery current
     c : mV membrane voltage after a downstroke
     d : pA bump to recovery current after a downstroke
     C : pF membrane capacitance
     k : nS/mV voltage-gated Na+ channel conductance
     Vr: mV resting membrane voltage when u=0
     Vt: mV threshold voltage when u=0
     Vp: mV action potential peak, after which reset happens
    tau: ms time constant for synaptic activation
     Vn: mV Nernst potential of the cell's neurotransmitter

    Finally, there is an optional triplet STDP rule for unsupervised
    learning, from Pfister and Gerstner, J. Neurosci. 26(38):9673.
    An arbitrary weight maximum has been introduced, but the proper
    way to do this is through synaptic scaling or homeostatic
    modulation of intrinsic excitability (TODO).

    The default parameters for STDP are taken from the same source,
    which derived them from a fit to V1 data recorded by Sjoström.
    """
    def __init__(self, *, XY=None, G,
                 a, b, c, d, C, k, Vr, Vt, Vp, Vn, tau,
                 # STDP parameters.
                 do_stdp=False, stdp_tau_plus=15, stdp_tau_minus=35,
                 stdp_tau_y=115, stdp_Aplus=6.5e-3, stdp_Aminus=7e-3):

        conv = partial(np.asarray, dtype=np.float32)
        self.G = conv(G)
        self.N = G.shape[0]
        if XY is not None:
            self.XY = conv(XY)
        self.a = conv(a)
        self.b = conv(b)
        self.c = conv(c)
        self.d = conv(d)
        self.C = conv(C)
        self.k = conv(k)
        self.Vr = conv(Vr)
        self.Vt = conv(Vt)
        self.Vp = conv(Vp)
        self.Vn = conv(Vn)
        self.tau = conv(tau)
        self.VUA = conv(np.zeros((4,self.N)))

        # STDP by the triplet model of Pfister and Gerstner (2006).
        # We store three synaptic traces at three different time
        # constants.
        self.do_stdp = do_stdp
        self.traces = conv(np.zeros((3,self.N)))
        stdp_taus = [stdp_tau_plus, stdp_tau_minus, stdp_tau_y]
        self.tau_stdp = conv([[tau] for tau in stdp_taus])
        self.Aplus = stdp_Aplus
        self.Aminus = stdp_Aminus

        self.reset()

    def reset(self):
        self.VUA[0,:] = self.Vr
        self.fired = self.V >= self.Vp
        self.VUA[1:,:] = 0

    def VUAdot(self, Iin):
        NAcurrent = self.k*(self.V - self.Vr)*(self.V - self.Vt)
        syncurrent = self.G@(self.A * self.Vn) - (self.G@self.A) * self.V
        Vdot = (NAcurrent - self.U + syncurrent + Iin) / self.C
        Udot = self.a * (self.b*(self.V - self.Vr) - self.U)
        Adot = self.Adot / self.tau
        Addot = -(self.A + 2*self.Adot) / self.tau
        return np.vstack([Vdot, Udot, Adot, Addot])

    def step(self, dt, Iin):
        """
        Simulate the organoid for a time dt, subject to an input
        current Iin.
        """

        # Apply the correction to any cells that crossed the AP peak
        # in the last update step, so that this step puts them into
        # the start of the refractory period.
        fired = self.fired
        self.V[fired] = self.c[fired]
        self.U[fired] += self.d[fired]
        self.Adot[fired] += 1

        if self.do_stdp:
            any = fired.any()

            if any:
                # Update for presynaptic spikes.
                pre_mod = self.Aminus*self.traces[1,fired]
                self.G[:,fired] -= pre_mod

                # Update for postsynaptic spikes.
                post_mod = self.traces[0,:] * self.traces[2,fired,None]
                self.G[fired,:] += self.Aplus * post_mod

            # Even if no cells fired, the traces decay
            self.traces *= np.exp(dt / self.tau_stdp)

            # Cells which fired increment their traces.
            if any: self.traces[:,fired] += 1

        # Actually do the stepping, using the midpoint method for
        # integration. This costs as much as halving the timestep
        # would in forward Euler, but increases the order to 2.
        Iin = np.asarray(Iin, dtype=np.float32)
        k1 = self.VUAdot(Iin)
        self.VUA += k1 * dt/2
        k2 = self.VUAdot(Iin)
        self.VUA += k2*dt - k1*dt/2

        # Make a note of which cells this step has caused to fire,
        # then correct their membrane voltages down to the peak.  This
        # can make some of the traces look a little weird; it may be
        # prettier to adjust the previous point UP to self.Vp and set
        # this point to self.c, but that's not possible here since we
        # don't save all states.
        self.fired = self.V >= self.Vp
        self.V[self.fired] = self.Vp[self.fired]


    @property
    def V(self):
        return self.VUA[0,:]

    @V.setter
    def V(self, value):
        self.VUA[0,:] = value

    @property
    def U(self):
        return self.VUA[1,:]

    @U.setter
    def U(self, value):
        self.VUA[1,:] = value

    @property
    def A(self):
        return self.VUA[2,:]

    @A.setter
    def A(self, value):
        self.VUA[2,:] = value

    @property
    def Adot(self):
        return self.VUA[3,:]

    @Adot.setter
    def Adot(self, value):
        self.VUA[3,:] = value

