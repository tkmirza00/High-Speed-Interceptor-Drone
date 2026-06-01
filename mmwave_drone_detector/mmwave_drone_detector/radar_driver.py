#!/usr/bin/env python3
"""
mmwave_drone_detector.radar_driver
===================================

Serial driver for TI IWR6843ISK / IWR6843ISK-ODS long-range people-detection
firmware variants.

Confirmed for your Long Range People Detection stream:
  - TLV 1000: float32 range, azimuth_rad, elevation_rad, doppler, 16 bytes/point
  - TLV 7:    uint16 snr_raw, uint16 noise_raw, 4 bytes/point
              SNR/noise are converted using 0.1 dB units.

ROS Cartesian convention after conversion:
  +X = radar boresight / forward range
  +Y = lateral left/right
  +Z = vertical
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
from typing import List, Tuple

log = logging.getLogger(__name__)

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
FRAME_HEADER_FMT = '<IIIIIIII'
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)
TLV_HEADER_FMT = '<II'
TLV_HEADER_SIZE = struct.calcsize(TLV_HEADER_FMT)

POINT_FMT = '<ffff'
POINT_SIZE = struct.calcsize(POINT_FMT)

POINT_COMP_UNIT_FMT = '<fffff'
POINT_COMP_UNIT_SIZE = struct.calcsize(POINT_COMP_UNIT_FMT)
POINT_COMP_FMT = '<bbhHBB'
POINT_COMP_SIZE = struct.calcsize(POINT_COMP_FMT)

# TARGET_SHORT_FMT: id(I), x(f), y(f), z(f), vx(f), vy(f), vz(f), ax(f), ay(f), az(f)
# indices:           0      1     2     3      4      5      6      7      8      9
TARGET_SHORT_FMT = '<I9f'
TARGET_SHORT_SIZE = struct.calcsize(TARGET_SHORT_FMT)
TARGET_SHORT_VEL_IDX = (4, 5, 6)   # vx, vy, vz

# TARGET_3D_FMT: id(I), x(f), y(f), z(f), vx(f), vy(f), vz(f), ax(f), ay(f), az(f),
#                cov[0..8](9f), ec0(f), ec1(f)
# indices:        0      1     2     3      4      5      6      7      8      9
#                 10..18                   19      20
TARGET_3D_FMT = '<I3f3f3f9fff'
TARGET_3D_SIZE = struct.calcsize(TARGET_3D_FMT)
TARGET_3D_VEL_IDX = (4, 5, 6)      # vx, vy, vz — same positional slots

TLV_SPHERICAL_POINTS = (1000,)
TLV_STANDARD_CARTESIAN_POINTS = (1,)
TLV_COMPRESSED_POINTS = (1020,)
TLV_SIDE_INFO = (7,)

# Map each TLV type that carries targets to its expected format.
# TLV 11 is the 3-D extended target list (TARGET_3D_FMT).
# TLV 1010 is also a 3-D list used by some SDK versions.
# TLV 12 / 1022 carry the compact short target list (TARGET_SHORT_FMT).
TLV_TARGET_LIST_3D    = (11, 1010)
TLV_TARGET_LIST_SHORT = (12, 1022)

TLV_TARGET_INDEX = (8, 12, 1011, 1021)


@dataclass
class RadarPoint:
    x: float
    y: float
    z: float
    doppler: float
    snr_db: float = 0.0
    noise_db: float = 0.0
    target_id: int = 255

    @property
    def range(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def azimuth_deg(self) -> float:
        # +X is radar boresight, +Y is lateral.
        return math.degrees(math.atan2(self.y, self.x))

    @property
    def elevation_deg(self) -> float:
        horizontal = math.sqrt(self.x * self.x + self.y * self.y)
        return math.degrees(math.atan2(self.z, horizontal))


@dataclass
class RadarTarget:
    id: int
    pos: Tuple[float, float, float]
    vel: Tuple[float, float, float]
    speed: float


@dataclass
class RadarFrame:
    frame_number: int
    timestamp: float
    points: List[RadarPoint] = field(default_factory=list)
    targets: List[RadarTarget] = field(default_factory=list)


class RadarDriver:
    def __init__(self, cli_port='/dev/ttyUSB0', data_port='/dev/ttyUSB1',
                 cfg_path=None, queue_size=16, debug=False):
        self.cli_port_name = cli_port
        self.data_port_name = data_port
        self.cfg_path = cfg_path
        self.debug = debug
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self._running = False
        self._thread = None
        self._cli = None
        self._data = None

    def _open_port(self, port, baud):
        s = serial.Serial()
        s.port = port
        s.baudrate = baud
        s.timeout = 0.2
        s.dtr = False
        s.rts = False
        s.open()
        return s

    def _send_config(self):
        if not self.cfg_path or not os.path.exists(self.cfg_path):
            log.warning(f'Config not found: {self.cfg_path}')
            return
        log.info(f'Sending config: {self.cfg_path}')
        with open(self.cfg_path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('%'):
                    continue
                self._cli.write((line + '\n').encode())
                self._cli.flush()
                time.sleep(0.08)
                resp = b''
                deadline = time.time() + 0.8
                while time.time() < deadline:
                    if self._cli.in_waiting:
                        resp += self._cli.read(self._cli.in_waiting)
                    time.sleep(0.01)
                if resp:
                    text = resp.decode(errors='ignore')
                    # Case-insensitive check: firmware may return ERROR / error / Error
                    text_lower = text.lower()
                    if self.debug or ('error' in text_lower) or ('not recognized' in text_lower):
                        print(f'CFG: {line}')
                        print(text)
        log.info('Config sent.')

    def _safe_read(self, n, timeout=2.0):
        out = b''
        deadline = time.time() + timeout
        while len(out) < n:
            if time.time() > deadline:
                raise TimeoutError(f'Timeout: wanted {n}, got {len(out)}')
            chunk = self._data.read(n - len(out))
            if chunk:
                out += chunk
        return out

    def _sync_magic(self):
        buf = bytearray()
        while self._running:
            b = self._data.read(1)
            if not b:
                continue
            buf += b
            if len(buf) > 8:
                buf = buf[-8:]
            if bytes(buf) == MAGIC_WORD:
                return True
        return False

    def _parse_spherical_points(self, payload):
        """Parse TLV 1000: range, azimuth_rad, elevation_rad, doppler."""
        points = []
        n = len(payload) // POINT_SIZE
        for i in range(n):
            off = i * POINT_SIZE
            rng, az, el, doppler = struct.unpack(POINT_FMT, payload[off:off + POINT_SIZE])
            if not all(math.isfinite(v) for v in (rng, az, el, doppler)):
                continue
            if rng <= 0.0 or rng > 120.0:
                continue
            if abs(az) > math.pi or abs(el) > (math.pi / 2.0):
                continue

            x = rng * math.cos(el) * math.cos(az)
            y = rng * math.cos(el) * math.sin(az)
            z = rng * math.sin(el)
            if not all(math.isfinite(v) for v in (x, y, z, doppler)):
                continue
            points.append(RadarPoint(x=float(x), y=float(y), z=float(z), doppler=float(doppler)))
        return points

    def _parse_standard_cartesian_points(self, payload):
        """Fallback for standard OOB TLV 1: x, y, z, doppler."""
        points = []
        n = len(payload) // POINT_SIZE
        for i in range(n):
            off = i * POINT_SIZE
            x, y, z, doppler = struct.unpack(POINT_FMT, payload[off:off + POINT_SIZE])
            if not all(math.isfinite(v) for v in (x, y, z, doppler)):
                continue
            points.append(RadarPoint(x=float(x), y=float(y), z=float(z), doppler=float(doppler)))
        return points

    def _parse_compressed_points(self, payload):
        points = []
        if len(payload) < POINT_COMP_UNIT_SIZE:
            return points
        elev_res, azm_res, dopp_res, range_res, snr_res = struct.unpack(
            POINT_COMP_UNIT_FMT, payload[:POINT_COMP_UNIT_SIZE])
        if not (0.0 < range_res < 10.0):
            return points
        off = POINT_COMP_UNIT_SIZE
        while off + POINT_COMP_SIZE <= len(payload):
            ei, ai, di, ri, si, ni = struct.unpack(POINT_COMP_FMT, payload[off:off + POINT_COMP_SIZE])
            off += POINT_COMP_SIZE
            rng = float(ri) * range_res
            azm = float(ai) * azm_res
            elev = float(ei) * elev_res
            doppler = float(di) * dopp_res
            snr_db = float(si) * snr_res
            noise_db = float(ni) * snr_res
            if rng < 0.05 or rng > 100.0:
                continue
            x = rng * math.cos(elev) * math.cos(azm)
            y = rng * math.cos(elev) * math.sin(azm)
            z = rng * math.sin(elev)
            if not all(math.isfinite(v) for v in (x, y, z, doppler)):
                continue
            points.append(RadarPoint(x=x, y=y, z=z, doppler=doppler,
                                     snr_db=snr_db, noise_db=noise_db))
        return points

    def _parse_side_info(self, payload):
        """Parse TLV 7 detected-point side info: uint16 snr, uint16 noise, 0.1 dB units."""
        side_info = []
        if len(payload) % 4 != 0:
            if self.debug:
                print(f'Bad side-info length: {len(payload)}')
            return side_info
        for off in range(0, len(payload), 4):
            snr_raw, noise_raw = struct.unpack('<HH', payload[off:off + 4])
            side_info.append({
                'snr_db': float(snr_raw) * 0.1,
                'noise_db': float(noise_raw) * 0.1,
                'snr_raw': int(snr_raw),
                'noise_raw': int(noise_raw),
            })
        return side_info

    def _parse_targets_with_fmt(self, payload, fmt, record_size, vel_idx):
        """
        Parse a target-list payload given an explicit struct format.

        vel_idx is a 3-tuple of (vx_index, vy_index, vz_index) into the
        unpacked value tuple.
        """
        targets = []
        if record_size == 0 or len(payload) % record_size != 0:
            if self.debug:
                print(f'Target payload length {len(payload)} not a multiple of '
                      f'record size {record_size} for fmt {fmt!r}')
            return targets
        vi, vj, vk = vel_idx
        for off in range(0, len(payload), record_size):
            values = struct.unpack(fmt, payload[off:off + record_size])
            vx, vy, vz = values[vi], values[vj], values[vk]
            targets.append(RadarTarget(
                id=int(values[0]),
                pos=(float(values[1]), float(values[2]), float(values[3])),
                vel=(float(vx), float(vy), float(vz)),
                speed=math.sqrt(vx * vx + vy * vy + vz * vz)))
        return targets

    def _read_frame(self):
        try:
            hdr = self._safe_read(FRAME_HEADER_SIZE)
        except TimeoutError:
            return None
        try:
            (version, total_len, platform, frame_num,
             time_cycles, num_detected, num_tlvs, subframe) = struct.unpack(FRAME_HEADER_FMT, hdr)
        except struct.error:
            return None

        remaining = int(total_len) - len(MAGIC_WORD) - FRAME_HEADER_SIZE
        if remaining < 0 or remaining > 262144:
            if self.debug:
                print(f'Bad frame length: total_len={total_len}, remaining={remaining}')
            return None

        points = []
        targets = []
        target_indices = []
        side_info = []
        bytes_read = 0

        for _ in range(num_tlvs):
            if bytes_read >= remaining:
                break
            try:
                th = self._safe_read(TLV_HEADER_SIZE)
                tlv_type, tlv_len = struct.unpack(TLV_HEADER_FMT, th)
            except (TimeoutError, struct.error):
                break
            bytes_read += TLV_HEADER_SIZE
            if tlv_len < 0 or tlv_len > 262144:
                if self.debug:
                    print(f'Bad TLV length: type={tlv_type}, len={tlv_len}')
                break
            try:
                payload = self._safe_read(tlv_len)
            except TimeoutError:
                break
            bytes_read += tlv_len
            if self.debug:
                print(f'frame={frame_num} num_obj={num_detected} tlv_type={tlv_type} tlv_len={tlv_len}')

            if tlv_type in TLV_SPHERICAL_POINTS:
                points = self._parse_spherical_points(payload)
            elif tlv_type in TLV_STANDARD_CARTESIAN_POINTS:
                points = self._parse_standard_cartesian_points(payload)
            elif tlv_type in TLV_COMPRESSED_POINTS:
                points = self._parse_compressed_points(payload)
            elif tlv_type in TLV_SIDE_INFO:
                side_info = self._parse_side_info(payload)
            elif tlv_type in TLV_TARGET_LIST_3D:
                # TLV type explicitly identifies the 3-D extended format.
                targets = self._parse_targets_with_fmt(
                    payload, TARGET_3D_FMT, TARGET_3D_SIZE, TARGET_3D_VEL_IDX)
            elif tlv_type in TLV_TARGET_LIST_SHORT:
                # TLV type explicitly identifies the compact short format.
                targets = self._parse_targets_with_fmt(
                    payload, TARGET_SHORT_FMT, TARGET_SHORT_SIZE, TARGET_SHORT_VEL_IDX)
            elif tlv_type in TLV_TARGET_INDEX:
                target_indices = list(payload)

        for i, pt in enumerate(points):
            if i < len(side_info):
                pt.snr_db = side_info[i]['snr_db']
                pt.noise_db = side_info[i]['noise_db']
            if i < len(target_indices):
                pt.target_id = int(target_indices[i])

        extra = remaining - bytes_read
        if extra > 0:
            try:
                self._safe_read(extra, timeout=0.2)
            except TimeoutError:
                pass

        return RadarFrame(frame_number=int(frame_num), timestamp=time.time(), points=points, targets=targets)

    def _run(self):
        while self._running:
            try:
                if not self._sync_magic():
                    break
                frame = self._read_frame()
                if frame is None:
                    continue
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
            except Exception as e:
                if self.debug:
                    print(f'Frame error: {e}')
                log.debug(f'Frame error: {e}')

    def start(self):
        log.info(f'Opening {self.cli_port_name} @ 115200 and {self.data_port_name} @ 921600')
        self._cli = self._open_port(self.cli_port_name, 115200)
        self._data = self._open_port(self.data_port_name, 921600)
        self._cli.reset_input_buffer()
        self._data.reset_input_buffer()
        self._send_config()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info('Radar driver running.')

    def stop(self):
        self._running = False
        try:
            if self._cli:
                self._cli.write(b'sensorStop\n')
                self._cli.flush()
                time.sleep(0.1)
        except Exception:
            pass
        try:
            if self._cli:
                self._cli.close()
            if self._data:
                self._data.close()
        except Exception:
            pass

    def get_frame(self, timeout=0.1):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
