# fruitfly-escape

A fruit fly's escape reflex, run straight from its wiring diagram, with no training.

This project takes the **male fruit fly central nervous system connectome** (MaleCNS v1.0, released by Google and HHMI Janelia), turns every traced neuron into a simulated spiking neuron, and checks whether the fly's looming detectors can make it take off. They can. Then it carves out the escape circuit and hooks it up to your webcam: bring your hand at the fly on screen, and a real connectome decides whether it jumps away in time.

## What's inside

| File | What it does |
|---|---|
| `brain.py` | Builds the signed weight matrix of all traced neurons and runs a leaky integrate-and-fire simulation of the whole brain. Tests the escape reflex by driving the LPLC2 looming detectors (and R7 photoreceptors as a control) and counting spikes in DNp01, the giant fiber. |
| `circuit.py` | Carves the escape circuit out of the whole brain and runs the same looming test on it, next to the whole brain. |
| `swat.py` | The webcam demo. |
| `looming-test.mp4` | A 6-second test clip of a dark circle rushing at the fly. |
| `data/escape-circuit-male-cns-v1.0.npz` | The carved escape circuit, so `swat.py` works right after cloning. |

## The model

A leaky integrate-and-fire network with the parameters of Shiu et al. 2024:

| Parameter | Value |
|---|---|
| Resting and reset potential | −52 mV |
| Threshold | −45 mV |
| Membrane time constant | 20 ms |
| Synaptic time constant | 5 ms |
| Refractory period | 2.2 ms |
| Synaptic delay | 1.8 ms |
| Weight per synapse | 0.275 mV |
| Time step | 0.5 ms |

- **Neurons:** only neurons with status `Traced` are kept (165,122 neurons, 25,563,197 connections, 124,025,046 synapses).
- **Signs:** each neuron's transmitter comes from the dataset's `consensus_nt` call. Acetylcholine is excitatory, GABA and glutamate are inhibitory, and everything else (including histamine and unlabeled neurons) is excitatory.
- **Input:** as in Shiu et al., each Poisson input event adds 250 × 0.275 mV to the driven neuron, so it fires, and driven neurons have no refractory period.
- **Time step rounding:** at a 0.5 ms time step, the delay and refractory period both round to 4 steps (2.0 ms), as the Brian2 simulator does.

## Results

DNp01 spikes (left and right giant fibers together) in 300 ms of Poisson drive to 185 neurons:

| Poisson rate (Hz) | LPLC2 driven, whole brain | R7 driven, whole brain | LPLC2 driven, escape circuit |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 20 | 45 | 0 | 73 |
| 50 | 48 | 0 | 117 |
| 100 | 62 | 0 | 200 |
| 150 | 163 | 1 | 222 |

The looming detectors drive the giant fiber; photoreceptors at the same rates barely reach it.

- **Whole brain:** 10 trials of 300 ms take about 81 s on a laptop CPU.
- **Escape circuit:** it keeps every neuron within four steps on a path from LPLC2 to DNp01 (steps from LPLC2 plus steps to DNp01 ≤ 4), counting only connections of 5 synapses or more. That leaves **12,510 neurons and 647,558 connections**, which run about **1.7× faster than real time**.

## The webcam demo

`swat.py` mirrors your webcam and draws a fly on the video.

1. **Looming:** it finds the biggest moving thing in the frame and measures its outward optical flow divided by its size. Motion that mostly hugs the edge of the frame is ignored.
2. **Input:** that looming becomes Poisson input to all LPLC2 neurons, capped at 100 Hz, and the escape circuit runs in real time.
3. **Takeoff:** when the giant fiber fires 3 spikes within 20 ms, the fly jumps away from your hand.
4. **Swat:** if your approaching hand covers the fly first, it gets swatted.

The side panel shows the threat level, LPLC2 spikes, a giant fiber raster, and counters for escapes, swats and spasms.

**Press S** to swap in a shuffled copy of the circuit, in which each neuron's connections are sent to random targets with the same weights. If the whole circuit fires more than 300 spikes per millisecond over 5 ms, the screen shows **SEIZURE**, and any takeoff counts as a spasm, not an escape. Press **Q** or **Esc** to quit.

On the test clip:

```
intact: 3 escapes, 0 swats, 0 spasms
shuffled: 0 escapes, 1 swats, 6 spasms
```

## Setup

Tested with Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy pandas pyarrow opencv-python
```

On macOS or Linux, use `.venv/bin/python` instead of `.venv/Scripts/python`.

### Run the demo

The escape circuit is included, so this works straight away:

```bash
.venv/Scripts/python swat.py
```

Options:

- `--source FILE`: use a video file instead of the webcam
- `--headless`: run without a window and print the counts at the end
- `--wiring intact|shuffled`: which circuit to start with

```bash
.venv/Scripts/python swat.py --source looming-test.mp4 --headless --wiring shuffled
```

### Rebuild everything from the connectome

`brain.py` and `circuit.py` need the connectome files, which are too big for GitHub (the weights file alone is about 1 GB). Download them into `data/`:

```bash
mkdir -p data
cd data
curl -LO https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -LO https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather
curl -LO https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
cd ..
```

Then run:

```bash
.venv/Scripts/python brain.py
.venv/Scripts/python circuit.py
```

- **`brain.py`:** the first run builds `data/signed-weights-traced-male-cns-v1.0.npz` (about 200 MB, roughly 40 s) and loads it on later runs.
- **`circuit.py`:** loads `data/escape-circuit-male-cns-v1.0.npz` if it exists. Delete that file to carve the circuit again.

Building the whole-brain matrix needs a few GB of free memory.

## Credits

- **Connectome:** MaleCNS v1.0, Google and HHMI Janelia (FlyEM).
- **Model parameters:** Shiu, P. K. et al. *A Drosophila computational brain model reveals sensorimotor processing.* Nature (2024).
