"""Write the K1 exactly as mjlab trains it (Booster 4 Hz PD gains, armature, feet-only collision, home keyframe)
to a standalone MJCF next to K1_22dof.xml, for the Isaac Sim MJCF -> USD importer.

usage: python scripts/export_mjcf.py [out.xml] [--arms]   (default: <K1 dir>/K1_22dof_mjlab[_arms].xml)
  --arms  also keep the arm/hand collision meshes (push/carry tasks touch objects with the hands)
  --all   every Booster collision mesh solid (Mjlab-PushIK: trunk and legs hit the box too)
"""
import re
import sys

from mjlab.entity import Entity

from mjlab_k1.k1_constants import K1_XML, get_k1_robot_cfg

arms = "--arms" in sys.argv
solid = "--all" in sys.argv
argv = [a for a in sys.argv[1:] if a not in ("--arms", "--all")]
name = "K1_22dof_mjlab_solid.xml" if solid else "K1_22dof_mjlab_arms.xml" if arms else "K1_22dof_mjlab.xml"
out = argv[0] if argv else str(K1_XML.with_name(name))
spec = Entity(get_k1_robot_cfg()).spec
spec.compiler.meshdir = "meshes/"
if arms:
  # group 1 = visual copies; the group-0 meshes are Booster's collision geoms
  for b in spec.bodies:
    if re.fullmatch(r"(Left|Right)_Arm_\d|(left|right)_hand_link", b.name):
      for g in b.geoms:
        if g.group != 1:
          g.contype, g.conaffinity = 1, 1
if solid:
  for b in spec.bodies:
    for g in b.geoms:
      if g.group != 1:
        g.contype, g.conaffinity = 1, 1
# USD prim names are unique per scope: a body named like its joint (Left_Hip_Pitch) makes the importer
# rename the joint to Left_Hip_Pitch_1. Rename the bodies instead so joint names stay canonical.
joint_names = {j.name for j in spec.joints}
for b in spec.bodies:
  if b.name in joint_names:
    b.name = f"{b.name}_link"
renamed = {m.name: f"{m.name}_mesh" for m in spec.meshes if m.name in joint_names}
for g in spec.geoms:
  if g.meshname in renamed:
    g.meshname = renamed[g.meshname]
for m in spec.meshes:
  if m.name in renamed:
    m.name = renamed[m.name]
# importer needs a ground-free robot file; keep the Booster solver options of the source file
xml = spec.to_xml()
open(out, "w").write(xml)
m = spec.compile()
print(f"MJCF {out} nq={m.nq} nu={m.nu} kp={[round(float(m.actuator_gainprm[i, 0]), 2) for i in range(m.nu)]}")
