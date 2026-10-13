"""Escape circuit carved from the whole brain: neurons on a path LPLC2 -> ... -> DNp01 of at most
four strong connections, and the strong connections between them. Runs the looming test on it
next to the whole brain and measures its speed relative to real time."""
import time
import hashlib
import json

import numpy as np
import pandas as pd

import brain

CIRCUIT = brain.DATA / 'escape-circuit-male-cns-v1.0.npz'
METADATA = brain.DATA / 'escape-circuit-male-cns-v1.0.metadata.json'
MIN_SYNAPSES = 5
MAX_STEPS = 4


def ids_digest(ids):
    """Stable fingerprint of the ordered body IDs used by a cached matrix."""
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()


def save_metadata(ids, types, path=None):
    """Save only the cell labels needed by the demo, with their data provenance."""
    path = METADATA if path is None else path
    metadata = {
        'schema_version': 1,
        'dataset': 'MaleCNS v1.0',
        'source': 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather',
        'body_ids_sha256': ids_digest(ids),
        'neuron_count': len(ids),
        'cell_types': {name: np.asarray(ids)[np.asarray(types) == name].tolist()
                       for name in ('LPLC2', 'DNp01')},
    }
    path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def cell_types(ids):
    """Read demo labels from the matching small cache, or the full annotations."""
    if not METADATA.exists():
        if brain.ANNOTATIONS.exists():
            return brain.cell_types(ids)
        raise FileNotFoundError('Missing circuit metadata. Restore data/' + METADATA.name
                                + ' from the repository, or download the neuron annotations.')
    metadata = json.loads(METADATA.read_text(encoding='utf-8'))
    if (metadata.get('schema_version') != 1 or metadata.get('neuron_count') != len(ids)
            or metadata.get('body_ids_sha256') != ids_digest(ids)):
        raise ValueError('Circuit metadata does not match the cached neuron IDs; rebuild the circuit.')
    types = pd.Series(None, index=np.asarray(ids), dtype=object)
    seen = set()
    for name in ('LPLC2', 'DNp01'):
        members = metadata.get('cell_types', {}).get(name, [])
        if not members or len(set(members)) != len(members) or seen.intersection(members):
            raise ValueError('Invalid or missing ' + name + ' labels in circuit metadata.')
        if not np.isin(members, ids).all():
            raise ValueError('Circuit metadata contains unknown neuron IDs.')
        seen.update(members)
        types.loc[members] = name
    return types.reset_index(drop=True)


def steps(A, sources):
    """Fewest steps from sources to each neuron, where A[i, j] != 0 means j reaches i in one step
    (inf beyond MAX_STEPS)."""
    dist = np.full(A.shape[0], np.inf)
    dist[sources] = 0
    frontier = np.zeros(A.shape[0])
    frontier[sources] = 1
    for k in range(1, MAX_STEPS + 1):
        reached = (A @ frontier > 0) & np.isinf(dist)
        dist[reached] = k
        frontier = reached.astype(float)
    return dist


def escape_circuit():
    """Signed synapse counts of the escape circuit (rows presynaptic) and the bodyId of each row."""
    if CIRCUIT.exists():
        return brain.load_matrix(CIRCUIT)
    W, ids = brain.signed_weights()
    types = brain.cell_types(ids)
    strong = W.copy()
    strong.data[np.abs(strong.data) < MIN_SYNAPSES] = 0
    strong.eliminate_zeros()
    adjacency = (strong != 0).astype(np.int8)
    from_lplc2 = steps(adjacency.T, np.flatnonzero(types == 'LPLC2'))
    to_dnp01 = steps(adjacency, np.flatnonzero(types == 'DNp01'))
    keep = np.flatnonzero(from_lplc2 + to_dnp01 <= MAX_STEPS)
    C = strong[keep][:, keep]
    brain.save_matrix(CIRCUIT, C, ids[keep])
    save_metadata(ids[keep], types.iloc[keep])
    return C, ids[keep]


def main():
    C, circuit_ids = escape_circuit()
    W, ids = brain.signed_weights()
    types = brain.cell_types(ids)
    lplc2, _, trains = brain.stimuli(types)
    dnp01 = np.flatnonzero(types == 'DNp01')
    print(f'Escape circuit: {C.shape[0]:,} neurons, {C.nnz:,} connections')

    circuit_types = brain.cell_types(circuit_ids)
    circuit_lplc2 = np.flatnonzero(circuit_types == 'LPLC2')
    circuit_dnp01 = np.flatnonzero(circuit_types == 'DNp01')
    kept_lplc2 = np.isin(ids[lplc2], circuit_ids)   # both sorted by bodyId, so train columns line up
    W, C = W * brain.W_SYN, C * brain.W_SYN

    whole = {rate: brain.run(W, lplc2, poisson, dnp01).sum() for rate, poisson in trains.items()}
    start = time.perf_counter()
    carved = {rate: brain.run(C, circuit_lplc2, poisson[:, kept_lplc2], circuit_dnp01).sum()
              for rate, poisson in trains.items()}
    elapsed = time.perf_counter() - start

    print(pd.DataFrame([(rate, whole[rate], carved[rate]) for rate in brain.RATES],
                       columns=['Poisson rate (Hz)', 'DNp01 spikes, whole brain',
                                'DNp01 spikes, escape circuit']).to_string(index=False))
    simulated = len(brain.RATES) * brain.TRIAL / 1000
    print(f'Escape circuit: {simulated:.1f} s simulated in {elapsed:.2f} s, '
          f'{simulated / elapsed:.1f}x faster than real time')


if __name__ == '__main__':
    main()
