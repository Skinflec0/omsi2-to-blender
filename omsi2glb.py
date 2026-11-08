#!/usr/bin/env python3
"""OMSI / OMSI 2 bus & object (.o3d) -> glTF binary (.glb) for Blender.

usage:
    python omsi2glb.py                       (opens a window: pick the bus folder)
    python omsi2glb.py <bus folder> [options]
    (you can also drag a folder onto the script)

What it does
    1. Finds the .bus file (or the model .cfg you name), reads its [model] cfg.
    2. Loads every [mesh] .o3d the cfg lists. O3D vertices are already in the
       vehicle's rest coordinates, so all parts line up with no extra transform.
    3. Finds textures (.tga .bmp .dds .png .jpg ...) anywhere in the folder,
       embeds them in the .glb, and converts OMSI's left-handed axes to glTF.
    If there is no usable cfg, every .o3d in the folder is exported instead.

Options
    -o FILE          output .glb path
    --bus NAME       which .bus file to use (default: ask / first)
    --cfg NAME       use this model .cfg instead of reading the .bus file
    --all-meshes     ignore the cfg and export every .o3d found
    --keep-dirt      also export dirt / rain overlays (skipped by default): meshes with
                     "dirt" in the name and material slots such as cldirt.tga / regen.tga.
                     OMSI fades these in and out with the weather, so a static export
                     would show them at full strength (e.g. opaque brown windows)
    --exclude TEXT   skip meshes whose file name contains TEXT (repeatable)
    --max-tex N      downscale textures so the longest side is <= N px
                     (default 4096, 0 = never downscale)
    --no-textures    export geometry only
    --alpha-mode M   how [matl_alpha] materials are exported: "blend" (default,
                     smooth transparency) or "mask" (hard cut-out, no sorting issues)

Supported: O3D versions 1-7. Versions 4+ may be scrambled; the loader undoes the
scramble when a key is present (key 0xFFFFFFFF means the geometry is stored plainly).
Needs: Python 3.8+, numpy, Pillow  (pip install numpy pillow)
Import the .glb in Blender with File > Import > glTF 2.0.
"""
import argparse
import io
import json
import os
import re
import struct
import sys

try:
    import numpy as np
    from PIL import Image
except ImportError as _exc:  # pragma: no cover
    sys.exit('missing dependency: %s\nInstall with:  pip install numpy pillow' % _exc.name)

__version__ = '0.3.0'

TEX_EXTS = ('.tga', '.dds', '.png', '.bmp', '.jpg', '.jpeg')

# glTF constants
GL_FLOAT, GL_UNSIGNED_INT = 5126, 5125
GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER = 34962, 34963

# Image.LANCZOS moved to Image.Resampling.LANCZOS in newer Pillow versions.
_LANCZOS = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')

# Dirt and rain layers are extra material slots that OMSI fades in and out with the weather.
# A static export would show them at full strength (opaque brown windows), so they are skipped.
OVERLAY_WORDS = ('dirt', 'dreck', 'schmutz', 'regen', 'wetness', 'rain_')

DEFAULT_MATERIAL = {'diffuse': (0.6, 0.6, 0.6, 1.0), 'specular': (0.0, 0.0, 0.0),
                    'emissive': (0.0, 0.0, 0.0), 'power': 0.0, 'tex': ''}


class O3DError(Exception):
    """A .o3d file could not be read."""


class ConvertError(Exception):
    """A user-facing error: the conversion cannot continue."""


# =============================================================================
#  Small helpers
# =============================================================================
def _f(x):
    """float(x), with NaN / inf replaced by 0 (they are not valid JSON)."""
    x = float(x)
    return x if x == x and abs(x) != float('inf') else 0.0


def _clamp01(x):
    return min(max(_f(x), 0.0), 1.0)


def _base(name):
    """File name part of a path that may use Windows or POSIX separators."""
    return os.path.basename(name.replace('\\', '/'))


def _norm(name):
    return re.sub(r'[^a-z0-9]', '', os.path.splitext(name.lower())[0])


def is_overlay(texname, alphascale=''):
    """True for dirt / rain layers (by texture name or by the variable that fades them)."""
    text = ('%s %s' % (texname, alphascale)).lower()
    return any(w in text for w in OVERLAY_WORDS)


def _note(lst, item):
    if item not in lst:
        lst.append(item)


def _pause(msg='Press Enter to close...'):
    try:
        input(msg)
    except EOFError:
        pass


# =============================================================================
#  O3D reader
# =============================================================================
def _uint(b, p, w):
    return int.from_bytes(b[p:p + w], 'little')


def _try_parse(b, start, cw, iw, mw, mcw, strict):
    """Parse an O3D body whose vertex marker (0x17) is at `start`.
    cw/iw/mw/mcw = byte widths of: counters, triangle indices, per-triangle
    material index, material count. Returns a dict or None if it doesn't fit."""
    n = len(b)
    try:
        if b[start] != 0x17:
            return None
        p = start + 1
        vc = _uint(b, p, cw)
        p += cw
        if vc == 0 or p + vc * 32 > n:
            return None
        verts = np.frombuffer(b, '<f4', vc * 8, p).reshape(vc, 8)
        p += vc * 32
        if b[p] != 0x49:
            return None
        p += 1
        tc = _uint(b, p, cw)
        p += cw
        rec = 3 * iw + mw
        if p + tc * rec > n:
            return None
        raw = np.frombuffer(b, np.uint8, tc * rec, p).reshape(tc, rec)
        p += tc * rec
        tri = np.ascontiguousarray(raw[:, :3 * iw]).view('<u%d' % iw).reshape(tc, 3).astype(np.int64)
        mat = np.ascontiguousarray(raw[:, 3 * iw:]).view('<u%d' % mw).reshape(tc).astype(np.int64)
        if tc and tri.max() >= vc:
            return None
        if b[p] != 0x26:
            return None
        p += 1
        mc = _uint(b, p, mcw)
        p += mcw
        if mc > 4096:
            return None
        mats = []
        for _ in range(mc):
            f = struct.unpack_from('<11f', b, p)
            p += 44
            ln = b[p]
            p += 1
            name = b[p:p + ln].decode('latin1')
            p += ln
            mats.append({'diffuse': f[0:4], 'specular': f[4:7], 'emissive': f[7:10],
                         'power': f[10], 'tex': name})
        while p < n and b[p] == 0x79:               # transform matrix: not needed
            p += 65
        if strict and p != n:
            return None
        return {'verts': verts, 'tris': tri, 'tmat': mat, 'mats': mats, 'leftover': n - p}
    except (struct.error, IndexError, ValueError):
        return None


def _unscramble_verts(verts, key, flags, version):
    """Undo OMSI's per-vertex scramble used on version >= 4 when key != 0xFFFFFFFF.

    Algorithm taken from the openOMSI loader (docs/FORMATS.md). Only the vertex
    block is scrambled; triangles, materials and the transform matrix are plain.
    """
    if key == 0xFFFFFFFF:
        return verts
    import math
    verts = np.array(verts, dtype=np.float32, copy=True)
    n = int(len(verts) % 0xFDE8)
    state = (key + version - 4 + (0x17D if (flags & 2) else 0)) % 0xFDE8
    b = 0
    for i in range(len(verts)):
        x, y, z = float(verts[i, 0]), float(verts[i, 1]), float(verts[i, 2])
        nx, ny, nz = float(verts[i, 3]), float(verts[i, 4]), float(verts[i, 5])
        u, v = float(verts[i, 6]), float(verts[i, 7])
        if key == 0:
            state = 0x130 if (flags & 2) else 0
        state = (state * n + n * b) % 8000
        fx = abs(x - math.floor(x))
        fy = abs(y - math.floor(y))
        fz = abs(z - math.floor(z))
        b = int(abs(fx * fy * fz) * 600) % 256
        if state < 1000:
            x, y = y, x
        elif state < 3000:
            x, z = z, x
        elif state > 7000:
            y, z = z, y
        if state % 4 == 0:
            nx = -nx
        if state % 6 == 0:
            ny = -ny
        if state % 7 == 0:
            nz = -nz
        if state < 600:
            ny, nz = nz, ny
        elif state > 4500:
            nx, ny = ny, nx
        elif state > 6500:
            nx, nz = nz, nx
        if state % 5 == 0:
            u -= ((state % 100) ** 2) / 10000.0
        if state % 3 == 0:
            v -= ((state % 50) ** 2) / 2500.0
        verts[i] = (x, y, z, nx, ny, nz, u, v)
    return verts


def read_o3d(path):
    """Read an .o3d file -> dict(verts Nx8, tris Mx3, tmat M, mats [...], version).

    Supports O3D versions 1-7. Versions 4+ may carry a scramble key; when the key
    is 0xFFFFFFFF the geometry is stored plainly (common for many add-on meshes).
    """
    try:
        with open(path, 'rb') as f:
            b = f.read()
    except OSError as e:
        raise O3DError('cannot read file: %s' % e)
    if len(b) < 8 or b[:2] != b'\x84\x19':
        raise O3DError('not an O3D file')
    ver = b[2]
    if ver > 7:
        raise O3DError('version %d is too new / not supported' % ver)

    if ver <= 2:
        r = _try_parse(b, 3, 2, 2, 2, 2, strict=False)
        if r:
            r['version'] = ver
            return r
        raise O3DError('could not read version %d layout' % ver)

    # version >= 3: optional extended header (flags + key from version 4)
    flags, key = 0, 0xFFFFFFFF
    if ver >= 4:
        if len(b) < 9:
            raise O3DError('truncated version %d header' % ver)
        flags = b[3]
        key = int.from_bytes(b[4:8], 'little')
        start = 8
    else:
        # version 3: try a few possible starts (some files have padding)
        start = None
        for s in range(3, min(len(b), 16)):
            if b[s] == 0x17:
                start = s
                break
        if start is None:
            raise O3DError('version 3 layout not recognised')

    # Prefer the layout that matches the flags bit (32-bit triangle indices)
    preferred_iw = 4 if (flags & 1) else 2
    candidates = [preferred_iw] + [w for w in (2, 4) if w != preferred_iw]

    r = None
    for iw in candidates:
        for mw in (2, 4):
            for mcw in (2, 4):
                cand = _try_parse(b, start, 4, iw, mw, mcw, strict=False)
                if cand is not None:
                    r = cand
                    break
            if r is not None:
                break
        if r is not None:
            break

    if r is None:
        raise O3DError('version %d layout not recognised' % ver)

    if key != 0xFFFFFFFF:
        r['verts'] = _unscramble_verts(r['verts'], key, flags, ver)

    r['version'] = ver
    return r


# =============================================================================
#  .bus / .cfg parsing
# =============================================================================
KEY = re.compile(r'^\[(\w+)\]\s*$')       # keywords must start at column 0 and stand alone


def _read_text(path):
    """Read an OMSI text file (usually Windows-1252) without ever failing on encoding."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return data.decode('utf-16')
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
    return data.decode('latin1')


def _params(lines, i):
    """Return (list of following non-empty lines up to the next keyword, new index)."""
    out = []
    while i < len(lines) and len(out) < 3:
        s = lines[i].strip()
        if KEY.match(lines[i].rstrip()):
            break
        i += 1
        if s:
            out.append(s)
    return out, i


def parse_bus_model(path):
    """-> relative path of the [model] cfg named in a .bus file, or None."""
    lines = _read_text(path).splitlines()
    for i, l in enumerate(lines):
        m = KEY.match(l.rstrip())
        if m and m.group(1).lower() == 'model':
            p, _ = _params(lines, i + 1)
            if p:
                return p[0]
    return None


def parse_cfg(path):
    """-> [{'mesh': 'Classic\\\\body.o3d', 'matl': [{'tex': str, 'alpha': int, 'alphascale': str}, ...]}, ...]

    [matl_alpha] sets a slot's transparency mode and [alphascale] names the variable that fades it
    (dirt and rain layers use this)."""
    lines = _read_text(path).splitlines()
    entries, cur, slot, i = [], None, None, 0
    while i < len(lines):
        m = KEY.match(lines[i].rstrip())
        if not m:
            i += 1
            continue
        key = m.group(1).lower()
        i += 1
        if key == 'mesh':
            p, i = _params(lines, i)
            if p:
                cur = {'mesh': p[0], 'matl': [], 'layers': []}
                entries.append(cur)
                slot = None
        elif key == 'matl' and cur is not None:
            p, i = _params(lines, i)
            idx = None
            if len(p) > 1 and re.fullmatch(r'-?\d+', p[0]):          # tolerate "index, texture"
                idx, p = int(p[0]), p[1:]
            # OMSI writes "texture name, number" here. The number is the material slot of the
            # .o3d that the texture goes on; repeated numbers are stacked layers on one slot.
            num = int(p[1]) if len(p) > 1 and re.fullmatch(r'-?\d+', p[1]) else None
            # An empty [matl] still occupies a material slot, so later slots keep their order.
            slot = {'tex': p[0] if p else '', 'alpha': 0, 'alphascale': '', 'number': num}
            cur['layers'].append((idx if idx is not None else num, slot))
        elif key == 'matl_alpha' and slot is not None:
            p, i = _params(lines, i)
            try:
                slot['alpha'] = int(float(p[0]))
            except (ValueError, IndexError):
                pass
        elif key == 'alphascale' and slot is not None:
            p, i = _params(lines, i)
            slot['alphascale'] = p[0] if p else ''
    return _resolve_layers(entries)


def _resolve_layers(entries):
    """Give every .o3d material slot one cfg override: the first layer that is not a
    dirt / rain overlay (a slot that only has overlay layers keeps one, so it is skipped later)."""
    for ent in entries:
        by_slot, nxt = {}, 0
        for target, slot in ent.pop('layers', []):
            if target is None or not 0 <= target < 256:
                target = nxt                     # no usable number: next slot in file order
            nxt = max(nxt, target + 1)
            by_slot.setdefault(target, []).append(slot)
        out = [None] * (max(by_slot) + 1 if by_slot else 0)
        for t, layers in by_slot.items():
            out[t] = next((s for s in layers if not is_overlay(s['tex'], s['alphascale'])), layers[0])
        ent['matl'] = out
    return entries


# =============================================================================
#  glTF binary writer
# =============================================================================
class GLB:
    """Minimal glTF 2.0 binary (.glb) writer."""

    def __init__(self):
        self.bin = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []
        self.textures = []
        self.images = []
        self._tex_index = {}

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        d = {'buffer': 0, 'byteOffset': len(self.bin), 'byteLength': len(data)}
        self.bin += data
        if target:
            d['target'] = target
        self.buffer_views.append(d)
        return len(self.buffer_views) - 1

    def accessor(self, arr, ctype, comp, target, minmax=False):
        arr = np.ascontiguousarray(arr)
        a = {'bufferView': self._view(arr.tobytes(), target), 'componentType': comp,
             'count': int(len(arr)), 'type': ctype}
        if minmax:
            a['min'] = arr.min(0).tolist()
            a['max'] = arr.max(0).tolist()
        self.accessors.append(a)
        return len(self.accessors) - 1

    def texture(self, png, key, name):
        """Embed a PNG once per `key` (a unique path) and return the texture index."""
        if key in self._tex_index:
            return self._tex_index[key]
        self.images.append({'bufferView': self._view(png), 'mimeType': 'image/png', 'name': name})
        self.textures.append({'source': len(self.images) - 1, 'sampler': 0})
        self._tex_index[key] = len(self.textures) - 1
        return self._tex_index[key]

    def save(self, path, roots):
        binary = bytes(self.bin)
        binary += b'\0' * (-len(binary) % 4)
        doc = {'asset': {'version': '2.0', 'generator': 'omsi2glb %s' % __version__},
               'scene': 0, 'scenes': [{'nodes': roots}],
               'buffers': [{'byteLength': len(binary)}]}
        for k, v in (('nodes', self.nodes), ('meshes', self.meshes), ('materials', self.materials),
                     ('accessors', self.accessors), ('bufferViews', self.buffer_views)):
            if v:
                doc[k] = v
        if self.textures:
            doc['textures'] = self.textures
            doc['images'] = self.images
            doc['samplers'] = [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]
        js = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
        js += b' ' * (-len(js) % 4)
        out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary))
        out += struct.pack('<II', len(js), 0x4E4F534A) + js
        out += struct.pack('<II', len(binary), 0x004E4942) + binary
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(out)
        os.replace(tmp, path)


# =============================================================================
#  Geometry
# =============================================================================
def smooth_normals(P, T):
    """Area-weighted vertex normals from triangles (used where a normal is missing)."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    fn = np.cross(b - a, c - a)
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0], dtype=P.dtype)
    return np.where(ln > 1e-12, N / np.maximum(ln, 1e-12), up).astype(np.float32)


def to_gltf_space(verts, tris):
    """OMSI is left-handed (+X right, +Z forward); glTF is right-handed.
    Mirror X and reverse the triangle winding. Returns P, N, UV (float32) and T."""
    P = np.array(verts[:, 0:3], dtype=np.float32)
    N = np.array(verts[:, 3:6], dtype=np.float32)
    UV = np.array(verts[:, 6:8], dtype=np.float32)
    P[:, 0] *= -1
    N[:, 0] *= -1
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    N = np.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)
    UV = np.nan_to_num(UV, nan=0.0, posinf=0.0, neginf=0.0)
    T = tris[:, [0, 2, 1]]
    ln = np.linalg.norm(N, axis=1)
    bad = ln < 1e-4
    if bad.any():
        N[bad] = smooth_normals(P, T)[bad]
        ln = np.linalg.norm(N, axis=1)
    N = (N / np.maximum(ln, 1e-12)[:, None]).astype(np.float32)
    return P, N, UV, T


# =============================================================================
#  File lookup
# =============================================================================
def index_files(folder):
    """{lower-case file name: [paths...]} for every file below `folder`."""
    idx = {}
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for f in sorted(files):
            idx.setdefault(f.lower(), []).append(os.path.join(root, f))
    return idx


def _pick(paths, hint=''):
    """Choose one of several same-named files, preferring the one whose path
    ends with the relative path (`hint`) the cfg gave."""
    if not paths:
        return None
    if len(paths) > 1 and hint:
        h = '/' + hint.replace('\\', '/').lower().lstrip('./')
        for p in paths:
            if ('/' + p.replace('\\', '/').lower()).endswith(h):
                return p
    return paths[0]


def _all_with_ext(files, ext):
    return sorted(p for n, ps in files.items() if n.endswith(ext) for p in ps)


def _resolve_mesh(ent, files, cfg_dir):
    if ent.get('path'):
        return ent['path']
    name = ent['mesh']
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    if cfg_dir:
        cand = os.path.normpath(os.path.join(cfg_dir, name.replace('\\', '/')))
        if os.path.isfile(cand):
            return cand
    return _pick(files.get(_base(name).lower(), []), name)


def ask_choice(title, options, label):
    print('\n' + title)
    for i, o in enumerate(options, 1):
        print('  %d) %s' % (i, label(o)))
    while True:
        try:
            s = input('Number [1]: ').strip() or '1'
        except EOFError:
            return options[0]
        if s.isdigit() and 1 <= int(s) <= len(options):
            return options[int(s) - 1]
        print('Please type a number from the list.')


def pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        try:
            path = filedialog.askdirectory(title='Select the OMSI bus folder (with Model, Texture, .bus files)')
        finally:
            root.destroy()
        return path or ''
    except Exception:
        try:
            return input('Paste the path of the bus folder: ').strip().strip('"')
        except EOFError:
            return ''


def select_entries(files, folder, bus=None, cfg=None, all_meshes=False, interactive=False):
    """Decide which meshes to load. Returns (entries, source name, cfg directory or None)."""
    entries, source, cfg_path = None, '', None
    if not all_meshes:
        if cfg:
            cfg_path = next((c for c in (cfg, os.path.join(folder, cfg)) if os.path.isfile(c)), None) \
                or _pick(files.get(_base(cfg).lower(), []), cfg)
            if not cfg_path:
                raise ConvertError('model cfg "%s" not found' % cfg)
            source = _base(cfg)
        else:
            buses = _all_with_ext(files, '.bus')
            if bus:
                wanted = _base(bus).lower()
                hit = [p for p in buses if os.path.basename(p).lower() in (wanted, wanted + '.bus')]
                if hit:
                    buses = hit
                elif buses:
                    print('note: no .bus file named "%s" found, falling back to the others' % bus)
            chosen = None
            if len(buses) > 1 and interactive:
                chosen = ask_choice('Several buses found, which one?', buses, os.path.basename)
            elif buses:
                chosen = buses[0]
                if len(buses) > 1:
                    print('note: several .bus files found, using %s' % os.path.basename(chosen))
            if chosen:
                source = os.path.basename(chosen)
                rel = parse_bus_model(chosen)
                if rel:
                    cand = os.path.normpath(os.path.join(os.path.dirname(chosen), rel.replace('\\', '/')))
                    cfg_path = cand if os.path.isfile(cand) else _pick(files.get(_base(rel).lower(), []), rel)
                    if not cfg_path:
                        print('note: %s names model file "%s" but it was not found' % (source, rel))
        if cfg_path:
            entries = parse_cfg(cfg_path)
            if entries:
                print('model: %s (%d mesh entries)' % (os.path.basename(cfg_path), len(entries)))
            else:
                print('note: no [mesh] entries found in %s' % os.path.basename(cfg_path))
                entries = None
    if entries is None:
        o3ds = _all_with_ext(files, '.o3d')
        entries = [{'mesh': p, 'matl': [], 'path': p} for p in o3ds]
        print('exporting every .o3d found (%d) - no usable model .cfg' % len(entries))
        source = source or os.path.basename(os.path.abspath(folder))
    return entries, source, (os.path.dirname(cfg_path) if cfg_path and entries else None)


# =============================================================================
#  Conversion
# =============================================================================
class Exporter:
    """Turns parsed O3D data into glTF meshes, materials and textures."""

    def __init__(self, files, textures=True, max_tex=4096, alpha_mode='blend', keep_dirt=False):
        self.files = files
        self.keep_dirt = keep_dirt
        self.overlays_skipped = 0
        self.textures = textures
        self.max_tex = max_tex
        self.alpha_mode = alpha_mode
        self.glb = GLB()
        self.missing, self.bad, self.fuzzy = set(), {}, {}
        self._png = {}
        self._mat = {}

    # ---- textures ---------------------------------------------------------
    def find_texture(self, name):
        """-> (path or None, matched_by_similar_name)"""
        base = _base(name).lower()
        if not base:
            return None, False
        if base in self.files:
            return _pick(self.files[base], name), False
        stem = os.path.splitext(base)[0]
        for e in TEX_EXTS:
            if stem + e in self.files:
                return _pick(self.files[stem + e], name), False
        # fuzzy: same start (e.g. WhiteCL.tga -> White_body.png)
        want = _norm(base)
        best, score = None, (0, 0)
        for fn, paths in self.files.items():
            if not fn.endswith(TEX_EXTS):
                continue
            have = _norm(fn)
            cp = len(os.path.commonprefix([want, have]))
            if cp >= max(4, int(0.6 * min(len(want), len(have)))) and (cp, -len(fn)) > score:
                best, score = paths[0], (cp, -len(fn))
        return (best, True) if best else (None, False)

    def _tex_png(self, path):
        if path in self._png:
            return self._png[path]
        res = None
        try:
            with Image.open(path) as im:
                im = im.convert('RGBA')
                if self.max_tex and self.max_tex > 0 and max(im.size) > self.max_tex:
                    im.thumbnail((self.max_tex, self.max_tex), _LANCZOS)
                if im.getchannel('A').getextrema()[0] == 255:      # fully opaque: drop alpha, smaller file
                    im = im.convert('RGB')
                bio = io.BytesIO()
                im.save(bio, 'PNG')
                res = bio.getvalue()
        except Exception as e:
            self.bad[os.path.basename(path)] = 'texture: %s' % e
        self._png[path] = res
        return res

    # ---- materials --------------------------------------------------------
    def material(self, m, override):
        override = override or {}
        texname = override.get('tex') or m['tex']
        alpha = override.get('alpha', 0) > 0
        em = tuple(round(_f(x), 3) for x in m['emissive'])
        diffuse = tuple(round(_f(x), 3) for x in m['diffuse'])
        # The .o3d material's own alpha (e.g. 0.3 on door glass) is what makes glass see-through.
        fa = diffuse[3] if 0.02 < diffuse[3] < 0.99 else 1.0
        if fa < 1.0:
            alpha = True

        path = png = None
        if self.textures and texname:
            path, fuzzy = self.find_texture(texname)
            if path is None and m['tex'] and m['tex'].lower() != texname.lower():
                path, fuzzy = self.find_texture(m['tex'])      # cfg name missing: try the o3d's own
            if path is None:
                self.missing.add(texname)
            else:
                if fuzzy:
                    self.fuzzy[texname] = os.path.basename(path)
                png = self._tex_png(path)

        # Untextured materials stay separate per texture name so they can be retextured in Blender.
        key = (os.path.normcase(path) if png else 'missing:' + texname.lower(),
               alpha, em, None if png else diffuse, fa)
        if key in self._mat:
            return self._mat[key]

        pbr = {'metallicFactor': 0.0, 'roughnessFactor': 0.8}
        mat = {'name': os.path.splitext(_base(texname))[0] or 'material', 'doubleSided': True,
               'pbrMetallicRoughness': pbr}
        if png:
            pbr['baseColorTexture'] = {'index': self.glb.texture(png, key[0], os.path.basename(path))}
            if fa < 1.0:
                pbr['baseColorFactor'] = [1.0, 1.0, 1.0, fa]
        else:
            d = [_clamp01(v) for v in diffuse]
            if not any(d[:3]):
                d = [0.6, 0.6, 0.6, 1.0]
            elif d[3] <= 0:
                d[3] = 1.0
            pbr['baseColorFactor'] = d
        if any(v > 0.01 for v in em):
            mat['emissiveFactor'] = [_clamp01(v) for v in em]
        if alpha:
            if self.alpha_mode == 'mask':
                mat['alphaMode'] = 'MASK'
                mat['alphaCutoff'] = 0.5
            else:
                mat['alphaMode'] = 'BLEND'
        self.glb.materials.append(mat)
        self._mat[key] = len(self.glb.materials) - 1
        return self._mat[key]

    # ---- meshes -----------------------------------------------------------
    def add_mesh(self, base, data, overrides):
        """Add one .o3d as a mesh + node. Returns the node index, or None if empty."""
        tmat = data['tmat']
        if len(data['tris']) == 0:
            return None
        mats = data['mats'] or [DEFAULT_MATERIAL]
        keep = []                                   # (material index, o3d material, cfg override)
        for mi in np.unique(tmat).tolist():
            m = mats[min(mi, len(mats) - 1)]
            ov = overrides[mi] if mi < len(overrides) else None
            hidden = is_overlay(m['tex']) and _f(m['diffuse'][3]) <= 0.02     # rain layer, fully see-through
            if hidden and not self.keep_dirt:
                self.overlays_skipped += 1
                continue
            if not self.keep_dirt and is_overlay((ov or {}).get('tex') or m['tex'],
                                                 (ov or {}).get('alphascale', '')):
                self.overlays_skipped += 1
                continue
            keep.append((mi, m, ov))
        if not keep:
            return None
        P, N, UV, T = to_gltf_space(data['verts'], data['tris'])
        g = self.glb
        att = {'POSITION': g.accessor(P, 'VEC3', GL_FLOAT, GL_ARRAY_BUFFER, True),
               'NORMAL': g.accessor(N, 'VEC3', GL_FLOAT, GL_ARRAY_BUFFER),
               'TEXCOORD_0': g.accessor(UV, 'VEC2', GL_FLOAT, GL_ARRAY_BUFFER)}
        prims = []
        for mi, m, ov in keep:
            sel = T[tmat == mi].astype(np.uint32).reshape(-1)
            idx = g.accessor(sel, 'SCALAR', GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER)
            prims.append({'attributes': att, 'indices': idx, 'material': self.material(m, ov)})
        name = os.path.splitext(base)[0]
        g.meshes.append({'name': name, 'primitives': prims})
        g.nodes.append({'name': name, 'mesh': len(g.meshes) - 1})
        return len(g.nodes) - 1


def convert(folder, out=None, bus=None, cfg=None, all_meshes=False, keep_dirt=False,
            exclude=(), max_tex=4096, textures=True, interactive=False, alpha_mode='blend'):
    """Convert the OMSI vehicle in `folder` to a .glb. Returns the output path."""
    if not os.path.isdir(folder):
        raise ConvertError('%s is not a folder' % folder)
    if alpha_mode not in ('blend', 'mask'):
        raise ConvertError('alpha mode must be "blend" or "mask"')
    files = index_files(folder)
    entries, source, cfg_dir = select_entries(files, folder, bus, cfg, all_meshes, interactive)

    skip_words = [w.lower() for w in exclude] + ([] if keep_dirt else ['dirt', 'rain'])
    ex = Exporter(files, textures, max_tex, alpha_mode, keep_dirt)
    skipped, seen, parts = [], set(), []

    for ent in entries:
        base = _base(ent['mesh'])
        low = base.lower()
        if not low.endswith('.o3d'):
            _note(skipped, base + ' (not an .o3d)')
            continue
        if any(w in low for w in skip_words):
            _note(skipped, base)
            continue
        path = _resolve_mesh(ent, files, cfg_dir)
        if not path:
            ex.missing.add(base)
            continue
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:                  # the cfg may list a mesh more than once
            continue
        seen.add(key)
        try:
            data = read_o3d(path)
        except O3DError as e:
            ex.bad[base] = str(e)
            continue
        node = ex.add_mesh(base, data, ent['matl'])
        if node is not None:
            parts.append(node)

    if not parts:
        problems = '; '.join('%s: %s' % kv for kv in ex.bad.items())
        raise ConvertError('nothing to export (no readable .o3d files found)'
                           + (' - ' + problems if problems else ''))

    root_name = os.path.splitext(source)[0] or 'omsi_model'
    ex.glb.nodes.append({'name': root_name, 'children': parts})
    out = out or os.path.join(folder, root_name + '.glb')
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    ex.glb.save(out, [len(ex.glb.nodes) - 1])

    print('wrote %s - %d parts, %d materials, %d textures'
          % (out, len(parts), len(ex.glb.materials), len(ex.glb.textures)))
    if skipped:
        print('skipped:', ', '.join(skipped))
    if ex.overlays_skipped:
        print('skipped %d dirt/rain overlay material slots (use --keep-dirt to include them)'
              % ex.overlays_skipped)
    if ex.fuzzy:
        print('textures matched by similar name (check these):',
              ', '.join('%s -> %s' % kv for kv in sorted(ex.fuzzy.items())))
    if ex.missing:
        print('files not found (skipped or untextured):', ', '.join(sorted(ex.missing)))
    for k, v in ex.bad.items():
        print('problem with %s: %s' % (k, v))
    return out


# =============================================================================
#  Command line
# =============================================================================
def build_parser():
    ap = argparse.ArgumentParser(
        description='OMSI bus/object (.o3d) -> glTF (.glb) for Blender')
    ap.add_argument('folder', nargs='?', help='bus folder (omit to pick one in a window)')
    ap.add_argument('-o', '--out', metavar='FILE', help='output .glb path')
    ap.add_argument('--bus', metavar='NAME', help='which .bus file to use')
    ap.add_argument('--cfg', metavar='NAME', help='use this model .cfg instead of the .bus file')
    ap.add_argument('--all-meshes', action='store_true', help='ignore the cfg, export every .o3d found')
    ap.add_argument('--keep-dirt', action='store_true', help='also export dirt / rain overlay meshes and material slots')
    ap.add_argument('--exclude', action='append', default=None, metavar='TEXT',
                    help='skip meshes whose file name contains TEXT (repeatable)')
    ap.add_argument('--max-tex', type=int, default=4096, metavar='N',
                    help='downscale textures to at most N px on the longest side (0 = never)')
    ap.add_argument('--no-textures', action='store_true', help='export geometry only')
    ap.add_argument('--alpha-mode', choices=('blend', 'mask'), default='blend',
                    help='export [matl_alpha] materials as smooth blend or hard cut-out')
    ap.add_argument('--version', action='version', version='omsi2glb ' + __version__)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):       # never crash on odd characters in file names
        try:
            stream.reconfigure(errors='replace')
        except Exception:
            pass

    interactive = a.folder is None
    code = 0
    try:
        folder = a.folder or pick_folder()
        if not folder:
            print('No folder selected.')
        else:
            convert(folder, a.out, a.bus, a.cfg, a.all_meshes, a.keep_dirt, a.exclude or [],
                    a.max_tex, not a.no_textures, interactive, a.alpha_mode)
            if interactive:
                print('\nDone. In Blender: File > Import > glTF 2.0.')
    except ConvertError as e:
        print('error: %s' % e, file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print('\ncancelled', file=sys.stderr)
        code = 130
    except Exception:
        if not interactive:
            raise
        import traceback
        traceback.print_exc()
        code = 1
    if interactive:
        _pause()
    return code


if __name__ == '__main__':
    sys.exit(main())
