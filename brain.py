"""Leaky integrate-and-fire model of the traced MaleCNS v1.0 neurons with the Shiu et al. 2024
parameters, and an untrained escape-reflex test: Poisson drive to LPLC2 (or, as a control,
to R7 photoreceptors) while counting spikes in DNp01, the giant fiber."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import scipy.sparse as sp

DATA = Path(__file__).parent / 'data'
ANNOTATIONS = DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather'
NEUROTRANSMITTERS = DATA / 'body-neurotransmitters-male-cns-v1.0.feather'
WEIGHTS = DATA / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
SIGNED_WEIGHTS = DATA / 'signed-weights-traced-male-cns-v1.0.npz'

# Shiu et al. 2024 parameters (mV, ms)
V_REST = -52.0      # resting and reset potential
V_TH = -45.0
TAU_M = 20.0
TAU_S = 5.0
T_REF = 2.2
DELAY = 1.8
W_SYN = 0.275       # per synapse
F_POI = 250         # a Poisson event adds F_POI * W_SYN to v, forcing a spike (Shiu et al.)
DT = 0.5

INHIBITORY = ['gaba', 'glutamate']
R7_TYPES = ['R7p', 'R7y', 'R7d', 'R7_unclear']
RATES = (0, 20, 50, 100, 150)   # Hz
TRIAL = 300                     # ms


def save_matrix(path, W, ids):
    np.savez(path, data=W.data, indices=W.indices, indptr=W.indptr, shape=W.shape, body_ids=ids)


def load_matrix(path):
    f = np.load(path)
    return sp.csr_matrix((f['data'], f['indices'], f['indptr']), shape=tuple(f['shape'])), f['body_ids']


def cell_types(ids):
    return pd.read_feather(ANNOTATIONS, columns=['bodyId', 'type']).set_index('bodyId')['type'].reindex(ids)


def stimuli(types):
    """LPLC2 and R7 indices, and for each rate the Poisson input trains shared by both groups."""
    lplc2 = np.flatnonzero(types == 'LPLC2')
    rng = np.random.default_rng(0)
    r7 = np.sort(rng.choice(np.flatnonzero(types.isin(R7_TYPES)), len(lplc2), replace=False))
    steps = round(TRIAL / DT)
    trains = {rate: rng.random((steps, len(lplc2))) < rate * DT / 1000 for rate in RATES}
    return lplc2, r7, trains


def signed_weights():
    """Signed synapse counts between traced neurons (rows presynaptic, columns postsynaptic)
    and the bodyId of each row. Built once from the flat connectome, then loaded from disk."""
    if SIGNED_WEIGHTS.exists():
        return load_matrix(SIGNED_WEIGHTS)

    ann = pd.read_feather(ANNOTATIONS, columns=['bodyId', 'status'])
    ids = np.sort(ann.loc[ann.status == 'Traced', 'bodyId'].to_numpy())
    # Read one column at a time to keep the 150M-row table out of memory.
    index = {col: pc.index_in(feather.read_table(WEIGHTS, columns=[col])[col], value_set=pa.array(ids))
             for col in ('body_pre', 'body_post')}
    keep = pc.and_(pc.is_valid(index['body_pre']), pc.is_valid(index['body_post']))
    pre = pc.filter(index['body_pre'], keep).to_numpy()
    post = pc.filter(index['body_post'], keep).to_numpy()
    count = pc.filter(feather.read_table(WEIGHTS, columns=['weight'])['weight'], keep).to_numpy()

    nt = pd.read_feather(NEUROTRANSMITTERS, columns=['body', 'consensus_nt']).set_index('body')['consensus_nt']
    sign = np.where(nt.reindex(ids).isin(INHIBITORY), -1, 1)
    W = sp.csr_matrix(((count * sign[pre]).astype(np.int32), (pre, post)), shape=(len(ids), len(ids)))
    save_matrix(SIGNED_WEIGHTS, W, ids)
    return W, ids


class LIF:
    """The network at rest, advanced one time step at a time. W holds synaptic weights in mV
    (rows presynaptic); driven are the neurons that receive Poisson input."""

    def __init__(self, W, driven):
        n = W.shape[0]
        self.W, self.driven = W, driven
        self.a, self.c = np.exp(-DT / TAU_M), np.exp(-DT / TAU_S)
        self.b = TAU_S / (TAU_S - TAU_M) * (self.c - self.a)   # exact integration of dv/dt = (g - (v - V_REST)) / TAU_M
        self.delay = round(DELAY / DT)
        self.ref_steps = np.full(n, round(T_REF / DT))
        self.ref_steps[driven] = 0          # Shiu et al.: Poisson targets have no refractory period
        self.v = np.full(n, V_REST)
        self.g = np.zeros(n)
        self.free_at = np.zeros(n, dtype=np.int64)
        self.queue = [np.empty(0, dtype=np.int64)] * self.delay
        self.t = 0

    def step(self, kicks):
        """Advance one time step; kicks marks the driven neurons that get a Poisson event.
        Returns the indices of the neurons that spiked."""
        t, n = self.t, self.W.shape[0]
        free = self.free_at <= t
        self.v = np.where(free, V_REST + (self.v - V_REST) * self.a + self.g * self.b, self.v)
        self.g = np.where(free, self.g * self.c, self.g)
        spiking = np.flatnonzero((self.v > V_TH) & free)

        arriving = self.queue[t % self.delay]   # spikes emitted `delay` steps ago
        if arriving.size:
            rows = self.W[arriving]
            self.g += np.bincount(rows.indices, weights=rows.data, minlength=n)
        self.queue[t % self.delay] = spiking
        self.v[self.driven[kicks]] += F_POI * W_SYN

        self.v[spiking] = V_REST
        self.g[spiking] = 0
        self.free_at[spiking] = t + self.ref_steps[spiking]
        self.t += 1
        return spiking


def run(W, driven, poisson, record):
    """Simulate one trial from rest. poisson is a (steps, len(driven)) boolean array of input
    events. Returns the spike count of each neuron in record."""
    net = LIF(W, driven)
    counts = np.zeros(len(record), dtype=np.int64)
    for kicks in poisson:
        counts += np.isin(record, net.step(kicks))
    return counts


def main():
    W, ids = signed_weights()
    W = W * W_SYN
    types = cell_types(ids)
    dnp01 = np.flatnonzero(types == 'DNp01')
    lplc2, r7, trains = stimuli(types)

    rows = []
    start = time.perf_counter()
    for rate, poisson in trains.items():
        rows.append((rate, run(W, lplc2, poisson, dnp01).sum(), run(W, r7, poisson, dnp01).sum()))
    elapsed = time.perf_counter() - start

    print(pd.DataFrame(rows, columns=['Poisson rate (Hz)',
                                      f'DNp01 spikes, {len(lplc2)} LPLC2 driven',
                                      f'DNp01 spikes, {len(r7)} R7 driven']).to_string(index=False))
    print(f'Whole-brain run: {elapsed:.1f} s for {len(rows) * 2} trials of {TRIAL} ms')


if __name__ == '__main__':
    main()
