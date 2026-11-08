"""Fix pinched / harsh shading on an imported OMSI bus.

Run it in Blender: Scripting tab > Open this file > Run Script. Save your .blend first.

OMSI models often carry hand-made vertex normals that look fine in OMSI (part of the
shading is baked into the textures) but show up as pinched gradients in Blender.
This clears those normals on every visible mesh and re-smooths by angle instead.
Change ANGLE_DEGREES if edges look too soft (lower it) or too faceted (raise it).
"""
import math

import bpy

ANGLE_DEGREES = 35

if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

meshes = [o for o in bpy.context.view_layer.objects if o.type == 'MESH' and o.visible_get()]
bpy.ops.object.select_all(action='DESELECT')

done = 0
for obj in meshes:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    except RuntimeError:
        pass
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(ANGLE_DEGREES), keep_sharp_edges=False)
    except (AttributeError, RuntimeError, TypeError):
        bpy.ops.object.shade_smooth()                      # Blender before 4.1
        if hasattr(obj.data, 'use_auto_smooth'):
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = math.radians(ANGLE_DEGREES)
    obj.select_set(False)
    done += 1

print('Re-smoothed %d meshes at %d degrees' % (done, ANGLE_DEGREES))
