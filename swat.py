"""Swat the fly: looming seen by the webcam drives LPLC2 in the carved escape circuit, and the fly
takes off when the giant fiber (DNp01) fires 3 spikes within 20 ms."""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import scipy.sparse as sp

import brain
import circuit

PROC_WIDTH = 320                # optical flow is computed at this width
MOTION_PX = 1.0                 # flow (px per frame at PROC_WIDTH) that counts as moving
MIN_AREA = 0.002                # smallest moving thing, as a fraction of the frame
EDGE_BAND = 0.08                # border band width, as a fraction of the shorter side
EDGE_FRACTION = 0.5             # a moving thing with more of its area in the band is ignored
LOOM_MIN, LOOM_MAX = 0.5, 3.0   # looming (1/s) that maps to no threat and to full threat
MAX_RATE = 100.0                # LPLC2 input at full threat (Hz)
GF_SPIKES, GF_WINDOW = 3, 20.0  # takeoff: giant fiber spikes within this many ms
SEIZURE_RATE, SEIZURE_WINDOW = 300, 5.0   # whole-circuit spikes per ms, over this many ms
MAX_FRAME_MS = 100.0            # longest webcam frame interval simulated at once
FLY_RADIUS = 24
JUMP_MS = 300
DEAD_MS = 1000
PANEL_WIDTH = 320
RASTER_MS = 1000
HISTORY = 120                   # frames of LPLC2 spike counts shown in the panel


def shuffled(W, seed=0):
    """Copy of W with each neuron's outgoing connections sent to random targets, same weights."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    degree = np.diff(W.indptr)
    targets = np.concatenate([rng.choice(n, k, replace=False) for k in degree])
    return sp.csr_matrix((W.data, (np.repeat(np.arange(n), degree), targets)), shape=W.shape)


def biggest_mover(prev, gray, band):
    """Outward optical flow of the biggest moving thing divided by its size (per frame), its outline
    and its centroid, in PROC_WIDTH pixels. None if nothing moves or it mostly hugs the edge."""
    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    moving = (np.hypot(flow[..., 0], flow[..., 1]) > MOTION_PX).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(moving)
    if n < 2:
        return None
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[k, cv2.CC_STAT_AREA]
    blob = labels == k
    if area < MIN_AREA * gray.size or (blob & band).sum() > EDGE_FRACTION * area:
        return None
    ys, xs = np.nonzero(blob)
    cx, cy = centroids[k]
    dx, dy = xs - cx, ys - cy
    outward = np.mean((flow[ys, xs, 0] * dx + flow[ys, xs, 1] * dy) / np.maximum(np.hypot(dx, dy), 1))
    size = (stats[k, cv2.CC_STAT_WIDTH] + stats[k, cv2.CC_STAT_HEIGHT]) / 4   # radius
    contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return outward / size, max(contours, key=cv2.contourArea), (cx, cy)


class Fly:
    def __init__(self, home, frame_size):
        self.home, self.pos, self.frame_size = home, home, frame_size
        self.state, self.until = 'ground', 0.0
        self.start = self.end = home

    def update(self, now):
        if self.state == 'air':
            if now >= self.until:
                self.state, self.pos = 'ground', self.end
            else:
                f = 1 - (self.until - now) / JUMP_MS
                self.pos = tuple(s + f * (e - s) for s, e in zip(self.start, self.end))
        elif self.state == 'dead' and now >= self.until:
            self.state, self.pos = 'ground', self.home

    def jump(self, now, threat_at, rng):
        """Take off away from threat_at (or in a random direction) and land a third of the frame away."""
        w, h = self.frame_size
        if threat_at is None:
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            direction = np.subtract(self.pos, threat_at)
            direction = direction / max(np.hypot(*direction), 1e-6)
        end = np.array(self.pos) + direction * w / 3
        end = np.clip(end, [2 * FLY_RADIUS, 2 * FLY_RADIUS], [w - 2 * FLY_RADIUS, h - 2 * FLY_RADIUS])
        self.state, self.until, self.start, self.end = 'air', now + JUMP_MS, self.pos, tuple(end)

    def swat(self, now):
        self.state, self.until = 'dead', now + DEAD_MS


def draw_fly(img, fly):
    x, y = (int(round(p)) for p in fly.pos)
    if fly.state == 'dead':
        cv2.ellipse(img, (x, y), (30, 18), 20, 0, 360, (40, 70, 50), -1)
        for dx, dy in ((-30, -14), (28, 16), (6, -24), (-10, 22)):
            cv2.circle(img, (x + dx, y + dy), 6, (40, 70, 50), -1)
        cv2.circle(img, (x + 4, y - 2), 6, (30, 30, 190), -1)
        return
    s = 1.4 if fly.state == 'air' else 1.0
    r = lambda v: int(round(v * s))
    for side in (-1, 1):
        cv2.ellipse(img, (x + side * r(13), y + r(6)), (r(9), r(22)), -side * 25, 0, 360, (225, 225, 215), -1)
        cv2.ellipse(img, (x + side * r(13), y + r(6)), (r(9), r(22)), -side * 25, 0, 360, (140, 140, 140), 1)
        for dy in (-6, 2, 10):
            cv2.line(img, (x, y + r(dy)), (x + side * r(20), y + r(dy + 8)), (30, 30, 30), 2)
    cv2.ellipse(img, (x, y + r(4)), (r(10), r(20)), 0, 0, 360, (40, 70, 110), -1)
    cv2.circle(img, (x, y - r(20)), r(9), (30, 45, 70), -1)
    for side in (-1, 1):
        cv2.circle(img, (x + side * r(7), y - r(22)), r(5), (30, 30, 200), -1)


def draw_panel(h, wiring, threat, lplc2_history, raster, now, counts):
    panel = np.full((h, PANEL_WIDTH, 3), 32, np.uint8)
    text = lambda s, y, color=(230, 230, 230), size=0.6: cv2.putText(
        panel, s, (16, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    left, right = 16, PANEL_WIDTH - 16

    text(f'Wiring: {wiring} (S to swap)', 30)
    text(f'Threat level: {threat:.0%}', 70)
    cv2.rectangle(panel, (left, 80), (right, 100), (80, 80, 80), 1)
    cv2.rectangle(panel, (left, 80), (left + int(threat * (right - left)), 100),
                  (0, int(200 * (1 - threat)), int(255 * threat)), -1)

    text(f'LPLC2 spikes: {lplc2_history[-1] if lplc2_history else 0} this frame', 140)
    peak = max(max(lplc2_history, default=0), 1)
    for i, count in enumerate(lplc2_history):
        x = left + i * (right - left) // HISTORY
        cv2.line(panel, (x, 210), (x, 210 - int(55 * count / peak)), (80, 200, 255), 2)

    text('Giant fiber raster (last 1 s)', 250)
    for row in range(2):
        y = 275 + row * 30
        cv2.putText(panel, f'GF{row + 1}', (left, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.line(panel, (left + 40, y), (right, y), (70, 70, 70), 1)
    for t, row in raster:
        x = left + 40 + int((1 - (now - t) / RASTER_MS) * (right - left - 40))
        y = 275 + row * 30
        cv2.line(panel, (x, y - 10), (x, y + 10), (255, 255, 255), 1)

    escapes, swats, spasms = counts
    text(f'Escapes: {escapes}', 380, (120, 230, 120), 0.8)
    text(f'Swats: {swats}', 420, (120, 120, 240), 0.8)
    text(f'Spasms: {spasms}', 460, (80, 200, 255), 0.8)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='video file instead of the webcam')
    parser.add_argument('--headless', action='store_true', help='no window')
    parser.add_argument('--wiring', choices=['intact', 'shuffled'], default='intact')
    args = parser.parse_args()

    C, ids = circuit.escape_circuit()
    types = brain.cell_types(ids)
    lplc2 = np.flatnonzero(types == 'LPLC2')
    gf_row = np.full(C.shape[0], -1)
    gf_row[np.flatnonzero(types == 'DNp01')] = [0, 1]
    is_lplc2 = np.zeros(C.shape[0], bool)
    is_lplc2[lplc2] = True
    weights = {'intact': C * brain.W_SYN, 'shuffled': shuffled(C) * brain.W_SYN}
    wiring = args.wiring
    net = brain.LIF(weights[wiring], lplc2)
    rng = np.random.default_rng(0)

    webcam = args.source is None
    cap = cv2.VideoCapture(0 if webcam else args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, frame = cap.read()
    if not ok:
        raise SystemExit('Could not read from ' + ('the webcam' if webcam else args.source))
    if webcam:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    proc_size = (PROC_WIDTH, round(h * PROC_WIDTH / w))
    scale = w / PROC_WIDTH
    to_gray = lambda f: cv2.cvtColor(cv2.resize(f, proc_size), cv2.COLOR_BGR2GRAY)
    prev = to_gray(frame)
    band = np.ones(prev.shape, bool)
    m = round(EDGE_BAND * min(prev.shape))
    band[m:-m, m:-m] = False

    fly = Fly((w / 2, h / 2), (w, h))
    gf_recent, raster = deque(), deque()
    step_spikes = deque(maxlen=round(SEIZURE_WINDOW / brain.DT))
    lplc2_history = deque(maxlen=HISTORY)
    escapes = swats = spasms = 0
    last = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if webcam:
                frame = cv2.flip(frame, 1)
                wall = time.perf_counter()
                frame_ms = min((wall - last) * 1000, MAX_FRAME_MS)
                last = wall
            else:
                frame_ms = 1000 / fps

            gray = to_gray(frame)
            mover = biggest_mover(prev, gray, band)
            prev = gray
            loom = mover[0] * 1000 / frame_ms if mover else 0.0
            threat = float(np.clip((loom - LOOM_MIN) / (LOOM_MAX - LOOM_MIN), 0, 1))
            threat_at = (mover[2][0] * scale, mover[2][1] * scale) if mover else None
            p = MAX_RATE * threat * brain.DT / 1000

            seizure, lplc2_spikes = False, 0
            for _ in range(round(frame_ms / brain.DT)):
                spiking = net.step(rng.random(len(lplc2)) < p)
                now = net.t * brain.DT
                step_spikes.append(len(spiking))
                seizing = sum(step_spikes) > SEIZURE_RATE * SEIZURE_WINDOW
                seizure |= seizing
                lplc2_spikes += int(is_lplc2[spiking].sum())
                for row in gf_row[spiking][gf_row[spiking] >= 0]:
                    gf_recent.append(now)
                    raster.append((now, row))
                while gf_recent and gf_recent[0] <= now - GF_WINDOW:
                    gf_recent.popleft()
                fly.update(now)
                if len(gf_recent) >= GF_SPIKES and fly.state == 'ground':
                    if seizing:
                        spasms += 1
                    else:
                        escapes += 1
                    fly.jump(now, threat_at, rng)
                    gf_recent.clear()

            now = net.t * brain.DT
            fly.update(now)
            if mover and threat > 0 and fly.state == 'ground':   # only something approaching can swat
                outline = (mover[1] * scale).astype(np.float32)
                if cv2.pointPolygonTest(outline, fly.pos, True) >= FLY_RADIUS:
                    swats += 1
                    fly.swat(now)
            lplc2_history.append(lplc2_spikes)
            while raster and raster[0][0] <= now - RASTER_MS:
                raster.popleft()

            if args.headless:
                continue
            draw_fly(frame, fly)
            if seizure:
                cv2.putText(frame, 'SEIZURE', (w // 2 - 170, 90), cv2.FONT_HERSHEY_DUPLEX, 2.5,
                            (0, 0, 255), 5, cv2.LINE_AA)
            panel = draw_panel(h, wiring, threat, lplc2_history, raster, now, (escapes, swats, spasms))
            cv2.imshow('Swat the fly', np.hstack([frame, panel]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key in (ord('s'), ord('S')):
                wiring = 'shuffled' if wiring == 'intact' else 'intact'
                net = brain.LIF(weights[wiring], lplc2)
                gf_recent.clear()
                raster.clear()
                step_spikes.clear()
                fly.update(now)
    except KeyboardInterrupt:
        pass
    cap.release()
    cv2.destroyAllWindows()
    print(f'{wiring}: {escapes} escapes, {swats} swats, {spasms} spasms')


if __name__ == '__main__':
    main()
