#!/usr/bin/env python3
"""
mmwave_drone_detector.radar_driver
===================================
Serial driver for IWR6843ISK 3D People Tracking firmware.
Parses compressed TLV 1020 point cloud format.
Standalone — no ROS dependency.
"""

import serial
import struct
import math
import threading
import queue
import time
import os
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)

# ── Protocol ──────────────────────────────────────────────────────────────────
MAGIC_WORD        = b'\x02\x01\x04\x03\x06\x05\x08\x07'
FRAME_HEADER_FMT  = '<IIIIIIII'
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)   # 32 bytes
TLV_HEADER_FMT    = '<II'
TLV_HEADER_SIZE   = struct.calcsize(TLV_HEADER_FMT)     # 8 bytes

# Standard Cartesian TLV 1
POINT_FMT  = '<ffff'
POINT_SIZE = struct.calcsize(POINT_FMT)

# Compressed spherical TLV 1020
POINT_COMP_UNIT_FMT  = '<fffff'
POINT_COMP_UNIT_SIZE = struct.calcsize(POINT_COMP_UNIT_FMT)  # 20 bytes
POINT_COMP_FMT       = '<bbhHBB'
POINT_COMP_SIZE      = struct.calcsize(POINT_COMP_FMT)       # 8 bytes

# Target list TLVs
TARGET_SHORT_FMT  = '<I9f'
TARGET_SHORT_SIZE = struct.calcsize(TARGET_SHORT_FMT)
TARGET_3D_FMT     = '<I3f3f3f9fff'
TARGET_3D_SIZE    = struct.calcsize(TARGET_3D_FMT)

TLV_POINT_CLOUD  = (1, 1020)
TLV_TARGET_LIST  = (7, 11)
TLV_TARGET_INDEX = (8, 12, 1021)


@dataclass
class RadarPoint:
    x:         float
    y:         float
    z:         float
    doppler:   float
    snr_db:    float = 0.0
    target_id: int   = 255

    @property
    def range(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def azimuth_deg(self) -> float:
        return math.degrees(math.atan2(self.x, self.y))

    @property
    def elevation_deg(self) -> float:
        r = math.sqrt(self.x**2 + self.y**2)
        return math.degrees(math.atan2(self.z, r))


@dataclass
class RadarTarget:
    id:    int
    pos:   Tuple[float, float, float]
    vel:   Tuple[float, float, float]
    speed: float


@dataclass
class RadarFrame:
    frame_number: int
    timestamp:    float
    points:       List[RadarPoint]  = field(default_factory=list)
    targets:      List[RadarTarget] = field(default_factory=list)


class RadarDriver:

    def __init__(self, cli_port='/dev/ttyUSB0', data_port='/dev/ttyUSB1',
                 cfg_path=None, queue_size=16):
        self.cli_port_name  = cli_port
        self.data_port_name = data_port
        self.cfg_path       = cfg_path
        self.frame_queue    = queue.Queue(maxsize=queue_size)
        self._running       = False
        self._thread        = None
        self._cli           = None
        self._data          = None

    def _open_port(self, port, baud):
        s = serial.Serial()
        s.port = port; s.baudrate = baud
        s.timeout = 2; s.dtr = False; s.rts = False
        s.open()
        return s

    def _send_config(self):
        if not self.cfg_path or not os.path.exists(self.cfg_path):
            log.warning(f'Config not found: {self.cfg_path}')
            return
        log.info(f'Sending config: {self.cfg_path}')
        with open(self.cfg_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                self._cli.write((line + '\n').encode())
                time.sleep(0.08)
                deadline = time.time() + 0.6
                while time.time() < deadline:
                    if self._cli.in_waiting:
                        self._cli.read(self._cli.in_waiting)
                    time.sleep(0.01)
        log.info('Config sent.')

    def _safe_read(self, n, timeout=3.0):
        data = b''
        deadline = time.time() + timeout
        while len(data) < n:
            if time.time() > deadline:
                raise TimeoutError(f'Timeout: wanted {n}, got {len(data)}')
            chunk = self._data.read(n - len(data))
            if chunk:
                data += chunk
        return data

    def _sync_magic(self):
        buf = bytearray()
        while self._running:
            b = self._data.read(1)
            if not b:
                continue
            buf += b
            if bytes(buf[-8:]) == MAGIC_WORD:
                return True
        return False

    def _parse_compressed_points(self, payload):
        points = []
        if len(payload) < POINT_COMP_UNIT_SIZE:
            return points
        elev_res, azm_res, dopp_res, range_res, snr_res = struct.unpack(
            POINT_COMP_UNIT_FMT, payload[:POINT_COMP_UNIT_SIZE])
        if not (0 < range_res < 1.0 and 0 < azm_res < 1.0):
            return points
        off = POINT_COMP_UNIT_SIZE
        while off + POINT_COMP_SIZE <= len(payload):
            ei, ai, di, ri, si, ni = struct.unpack(
                POINT_COMP_FMT, payload[off:off+POINT_COMP_SIZE])
            off += POINT_COMP_SIZE
            rng  = ri * range_res
            azm  = ai * azm_res
            elev = ei * elev_res
            dopp = di * dopp_res
            if rng < 0.05 or rng > 50.0:
                continue
            x = rng * math.cos(elev) * math.sin(azm)
            y = rng * math.cos(elev) * math.cos(azm)
            z = rng * math.sin(elev)
            points.append(RadarPoint(
                x=x, y=y, z=z, doppler=dopp, snr_db=si * snr_res))
        return points

    def _parse_targets(self, payload):
        targets = []
        for fmt, sz in [(TARGET_3D_FMT, TARGET_3D_SIZE),
                        (TARGET_SHORT_FMT, TARGET_SHORT_SIZE)]:
            if len(payload) >= sz and len(payload) % sz == 0:
                for i in range(0, len(payload), sz):
                    f = struct.unpack(fmt, payload[i:i+sz])
                    vx, vy, vz = f[4], f[5], f[6]
                    targets.append(RadarTarget(
                        id=f[0], pos=(f[1], f[2], f[3]),
                        vel=(vx, vy, vz),
                        speed=math.sqrt(vx**2+vy**2+vz**2)))
                break
        return targets

    def _read_frame(self):
        try:
            hdr = self._safe_read(FRAME_HEADER_SIZE)
        except TimeoutError:
            return None

        (version, total_len, platform, frame_num,
         time_cycles, num_detected, num_tlvs, subframe) = \
            struct.unpack(FRAME_HEADER_FMT, hdr)

        remaining = total_len - 8 - FRAME_HEADER_SIZE
        if remaining < 0 or remaining > 65536:
            return None

        points, targets, tid_map = [], [], []
        bytes_read = 0

        for _ in range(num_tlvs):
            if bytes_read >= remaining:
                break
            try:
                th = self._safe_read(TLV_HEADER_SIZE)
            except TimeoutError:
                break
            tlv_type, tlv_len = struct.unpack(TLV_HEADER_FMT, th)
            bytes_read += TLV_HEADER_SIZE
            if tlv_len > 65536:
                break
            try:
                payload = self._safe_read(tlv_len)
            except TimeoutError:
                break
            bytes_read += tlv_len

            if tlv_type == 1:
                n = len(payload) // POINT_SIZE
                for i in range(n):
                    x, y, z, d = struct.unpack(
                        POINT_FMT, payload[i*POINT_SIZE:(i+1)*POINT_SIZE])
                    points.append(RadarPoint(x=x, y=y, z=z, doppler=d))
            elif tlv_type == 1020:
                points = self._parse_compressed_points(payload)
            elif tlv_type in TLV_TARGET_LIST:
                targets = self._parse_targets(payload)
            elif tlv_type in TLV_TARGET_INDEX:
                tid_map = list(payload[:tlv_len])

        for i, pt in enumerate(points):
            if i < len(tid_map):
                pt.target_id = tid_map[i]

        return RadarFrame(frame_number=frame_num,
                          timestamp=time.time(),
                          points=points, targets=targets)

    def _run(self):
        while self._running:
            try:
                if not self._sync_magic():
                    break
                frame = self._read_frame()
                if frame is not None:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put_nowait(frame)
            except Exception as e:
                log.debug(f'Frame error: {e}')

    def start(self):
        log.info(f'Opening {self.cli_port_name} @ 115200 and {self.data_port_name} @ 921600')
        self._cli  = self._open_port(self.cli_port_name,  115200)
        self._data = self._open_port(self.data_port_name, 921600)
        self._send_config()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info('Radar driver running.')

    def stop(self):
        self._running = False
        try:
            self._cli.write(b'sensorStop\n')
            time.sleep(0.1)
        except Exception:
            pass
        try:
            self._cli.close()
            self._data.close()
        except Exception:
            pass

    def get_frame(self, timeout=0.1):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *_):
        self.stop()
