```markdown
# omsi2glb

Convert **OMSI 2** buses and objects (`.o3d`) into a single **glTF binary (`.glb`)** that opens in
Blender (and any other glTF viewer) with the parts lined up, textured and facing the right way.

Two small tools, no install step beyond Python, NumPy and Pillow:

| File | What it is |
| --- | --- |
| `omsi2glb.py` | Command-line converter: `.bus` / `.cfg` / `.o3d` → `.glb` |
| `fix_shading.py` | Blender helper: re-smooths imported meshes so OMSI's hand-made normals stop looking pinched |

- Reads the `.bus` file, follows its `[model]` line to the model `.cfg`, and loads every `[mesh]` it lists
- Applies the cfg's `[matl]` / `[matl_alpha]` texture overrides
- Finds textures anywhere in the folder (`.tga .dds .png .bmp .jpg`), embeds them, optionally downscaled
- Converts OMSI's left-handed axes to glTF (X mirrored, triangle winding reversed)
- Skips dirt / rain overlays by default (they are weather layers, not geometry)
- The Blender helper re-smooths imported meshes by angle instead of keeping OMSI's baked normals

## Install

```bash
git clone https://github.com/<you>/omsi2glb.git
cd omsi2glb
pip install -r requirements.txt     # numpy + pillow, Python 3.8 or newer
```

`fix_shading.py` needs only Blender — no extra install.

## Use

```bash
python omsi2glb.py                           # opens a window: pick the bus folder
python omsi2glb.py "C:\path\to\My Bus"       # or pass the folder
python omsi2glb.py "C:\path\to\My Bus" -o out/mybus.glb --alpha-mode mask
```

On Windows you can also drag the bus folder onto `omsi2glb.py`.

Then in Blender: **File > Import > glTF 2.0 (.glb)**.

The output is written next to the bus files (named after the `.bus` file) unless you pass `-o`.

### Options

| Option | Meaning |
| --- | --- |
| `-o FILE` | Output `.glb` path (folders are created if needed) |
| `--bus NAME` | Which `.bus` file to use when the folder has several |
| `--cfg NAME` | Use this model `.cfg` directly instead of reading the `.bus` file |
| `--all-meshes` | Ignore the cfg and export every `.o3d` found |
| `--keep-dirt` | Also export meshes with "dirt" in the name (skipped by default, they are overlays) |
| `--exclude TEXT` | Skip meshes whose file name contains `TEXT` (repeatable) |
| `--max-tex N` | Downscale textures to at most `N` px on the longest side (default 4096, `0` = never) |
| `--no-textures` | Export geometry only |
| `--alpha-mode blend\|mask` | Export `[matl_alpha]` materials as smooth blending (default) or a hard cut-out |

If no usable cfg is found, every `.o3d` in the folder is exported instead.

## Fixing pinched / harsh shading in Blender

OMSI models often carry hand-made vertex normals that look fine inside OMSI (part of the shading is
baked into the textures) but show up as pinched gradients in Blender. `fix_shading.py` clears those
custom split normals on every visible mesh and re-smooths by angle instead.

1. Import the `.glb` and save your `.blend` first.
2. **Scripting** tab > **Open** > pick `fix_shading.py` > **Run Script**.

Change `ANGLE_DEGREES` at the top of the file if edges look too soft (lower it) or too faceted
(raise it); 35° is a good starting point. The script prints how many meshes it re-smoothed.

## What is supported

| O3D version | Status |
| --- | --- |
| 1, 2 | Supported |
| 3 | Attempted; accepted only if the layout parses exactly |
| 4 and newer | Not supported (scrambled / protected files) |

Not exported: animations (`[mesh_anim]`, doors, wheels turning), LODs, bump / environment / night maps,
and per-material specular settings. Every part is exported in its rest pose.

## Troubleshooting

- **Untextured (grey / white) parts**: the console lists every texture that was not found. Make sure the
  texture files are inside the folder you passed (subfolders are searched).
- **"textures matched by similar name"**: a texture was not found by its exact name, so the closest
  file name was used. Check these are right.
- **Transparent parts look wrong in Blender**: try `--alpha-mode mask`.
- **Shading looks pinched / has hard gradients**: run `fix_shading.py` in Blender (see above).
- **"version N is scrambled/protected"**: that model uses a protected O3D format this tool cannot read.
- **Huge file**: lower `--max-tex` (for example `--max-tex 1024`).

## Development

```bash
python -m unittest discover -s tests -v
```

Tests build tiny synthetic `.o3d` files, so no OMSI assets are needed. `fix_shading.py` is a
Blender-only script and is not unit-tested.

## Legal

This project is not affiliated with or endorsed by Aerosoft or the OMSI developers. OMSI vehicles and
their models, textures and cfg files are usually copyrighted by their authors. Only convert content you
made or have permission to use, and respect the licence of any bus you redistribute.

The tools themselves are released under the [MIT License](LICENSE).
```
