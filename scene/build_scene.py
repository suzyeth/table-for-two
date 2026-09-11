"""Build the dual SO-101 dinner-table scene with contact-only manipulation.

Nothing in this scene attaches objects to the grippers: every object is held
only by finger contact and friction, so object mass, friction and shape
actually matter (the robustness perturbations the brief scores).

  * Robots: two copies of the MuJoCo Menagerie SO-101 (derived from the
    TheRobotStudio model) with primitive finger colliders and grasp-tuned
    contact parameters; prefixes ``left_`` / ``right_``.
  * Objects: a cabinet whose drawer has a bar handle on stand-offs; a spoon
    (handle + bowl) and a fork (handle + three tines) inside the drawer; a
    plate on a foot ring so its rim overhangs the table; an open-topped mug and
    bottle built as thin-walled shells; 24 beads inside the bottle standing in
    for water (0.27 g each, like 0.27 ml).
  * Layout from the SO-101 reach map: top-down grasps are reachable 13-28 cm
    from a base and up to ~8 cm above the table, so every top-down pick and
    place point sits there. A jaws-horizontal side grip near the table (how
    the bottle must be held so its mouth stays free for pouring) only reaches
    >= ~30 cm out, so the bottle stands at the far edge of the right arm's
    workspace. The left arm works the cabinet, fork and plate; the right arm
    the mug and bottle; the spoon is handed over between them.

World frame: +x away from the arm bases, +y toward the left arm, z up (m).

Run:  .venv\\Scripts\\python.exe scene\\build_scene.py
"""
import contextlib
import io
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SO101_DIR = ROOT / "third_party" / "mujoco_menagerie" / "robotstudio_so101"
ARM_XML = SO101_DIR / "so101.xml"
ASSETS_DIR = SO101_DIR / "assets"
OUT_XML = ROOT / "scene" / "bimanual_table.xml"

TABLE_TOP_Z = 0.40
ARM_BASES = {"left_": (-0.18, 0.14), "right_": (-0.18, -0.14)}

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
PLANE = mujoco.mjtGeom.mjGEOM_PLANE

WOOD = [0.55, 0.38, 0.24, 1.0]
CABINET_RGBA = [0.85, 0.85, 0.82, 1.0]
DRAWER_RGBA = [0.75, 0.6, 0.45, 1.0]
STEEL = [0.72, 0.73, 0.78, 1.0]
CERAMIC = [0.95, 0.95, 0.95, 1.0]
PROP_FRICTION = [1.0, 0.005, 0.0001]
# Gripper: sustained torque of a 7.4 V STS3215 (stall 1.62 N m; ~half of that without tripping
# its overload protection) and rubber finger pads (mu ~0.8 on ceramic/plastic). The pads'
# friction is what every finger-object contact uses (the pad geoms have contact priority), and
# sim/env.py scales it per seed.
GRIPPER_TORQUE = 0.8
PAD_FRICTION = [0.8, 0.005, 0.0005]

# Cabinet on the left-front, drawer sliding toward the arms (-x). The drawer handle is a
# bar on stand-offs, pinched top-down with one finger between bar and drawer front.
# Full-extension drawer: pulled 10 cm, the 9.6 cm utensils come out from under the
# cabinet top entirely, so they can be lifted straight up.
CABINET = {"pos": (0.14, 0.19), "half": (0.060, 0.065), "height": 0.050, "wall": 0.004}
DRAWER = {"travel": 0.105, "wall_h": 0.014, "damping": 0.5, "frictionloss": 0.3}
# Bar pull 2.8 cm off the drawer front (typical for real pulls); posts clear the 3 cm-wide jaws.
HANDLE = {"bar_half": (0.004, 0.030, 0.004), "standoff": 0.028}

# Shell containers: outer radius, height, wall thickness, mass, segments.
# 4.8 cm across (espresso-cup size): still inside the jaws' reach, and a wider target for the pour.
MUG = {"pos": (0.00, -0.15), "radius": 0.024, "height": 0.060, "wall": 0.0025, "mass": 0.12, "rgba": [0.2, 0.45, 0.8, 1]}
# 32 mm x 70 mm glass vial: ~40 g empty, plus 6.5 g of bead water.
BOTTLE = {"pos": (0.12, -0.22), "radius": 0.016, "height": 0.070, "wall": 0.002, "mass": 0.04, "rgba": [0.3, 0.75, 0.4, 0.85]}
SHELL_SEGMENTS = 16
# 8 mm container bases (like real cups and bottles). With 5 mm bases, beads landing on
# beads drove the lowest ones more than half-way into the disc, where the contact pushes
# them out through the bottom - water "leaking" through a solid base.
SHELL_BASE_HALF = 0.004
# Stiffer bead contacts (time constant 2x the 2 ms timestep, MuJoCo's recommended minimum).
WATER = {"count": 24, "radius": 0.004, "mass": 0.00027, "rgba": [0.35, 0.6, 1.0, 1], "solref": [0.004, 1.0]}

# Deep plate: a low wall around a base. A top-down pinch on the wall gives a vertical
# contact line that resists the plate pivoting (a flat rim only gives point contact).
# Starts in the open middle of the table: both hands carry it (sim/bimanual.py), and next to
# the cabinet the left hand's moving jaw hits the cabinet front.
PLATE = {"pos": (0.05, 0.00), "radius": 0.045, "height": 0.020, "wall": 0.003, "mass": 0.12, "rgba": CERAMIC}
# Utensils lie front-to-back in the drawer like in a cutlery drawer: long axis along x,
# head toward the back (+x), handle ends toward the arms so they come out first. They sit
# 4.8 cm apart so the jaws, closing sideways, keep the moving finger's back (~2.5 cm out)
# clear of the drawer side wall and the thin fixed finger clear of the other utensil.
# (x, y) are the handle-centre offsets from the cabinet centre.
UTENSILS = {"spoon": (-0.023, -0.022), "fork": (-0.023, 0.022)}
# 7 cm handle (room for two grippers during the hand-over), 10 mm square like a sturdy
# wooden or children's utensil. The SO-101 jaws end ~8 mm below the TCP, so a flat 5 mm
# handle leaves the pads only its top 1-2 mm once the jaws stop above the table; and on a
# 6 mm-wide handle the jaws sit almost fully shut (-0.156 of -0.17 rad), so the ~80 N
# squeeze slowly pushes through the soft contact until the handle pops out.
UTENSIL_HANDLE = (0.035, 0.005, 0.005)

# Place setting, seen from the arms: plate on the mat, fork left, spoon right (both
# laid along x), mug to the front right where the right arm can pour into it.
# Every grasp point at placing stays >= ~13 cm from its arm's base (the inner edge of
# top-down reach).
PLACE_TARGETS = {
    "plate": (-0.02, 0.03),
    "fork": (-0.02, 0.10),
    "spoon": (-0.03, -0.06),
    "mug": (0.06, -0.05),
}


def add_box(body, name, half_size, pos, rgba, **extra):
    return body.add_geom(name=name, type=BOX, size=list(half_size), pos=list(pos), rgba=rgba, **extra)


def add_environment(spec):
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    world = spec.worldbody
    world.add_geom(name="floor", type=PLANE, size=[3, 3, 0.05], rgba=[0.25, 0.3, 0.35, 1])
    world.add_light(name="key_light", pos=[0.0, 0.0, 2.0], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5])
    table = world.add_body(name="table")
    add_box(table, "table_top", (0.40, 0.38, 0.02), (0, 0, TABLE_TOP_Z - 0.02), WOOD, friction=PROP_FRICTION)
    leg_half_h = (TABLE_TOP_Z - 0.04) / 2
    for i, (lx, ly) in enumerate([(0.36, 0.34), (0.36, -0.34), (-0.36, 0.34), (-0.36, -0.34)]):
        add_box(table, f"table_leg_{i}", (0.02, 0.02, leg_half_h), (lx, ly, leg_half_h), WOOD)
    world.add_body(name="workspace_center", pos=[-0.02, 0.0, TABLE_TOP_Z + 0.03])


def add_cabinet(spec):
    """Low cabinet, open toward -x, with a sliding drawer and a bar handle on stand-offs."""
    cx, cy = CABINET["pos"]
    hx, hy = CABINET["half"]
    h, t = CABINET["height"], CABINET["wall"]
    cabinet = spec.worldbody.add_body(name="cabinet", pos=[cx, cy, TABLE_TOP_Z])
    add_box(cabinet, "cabinet_bottom", (hx, hy, t / 2), (0, 0, t / 2), CABINET_RGBA)
    add_box(cabinet, "cabinet_top", (hx, hy, t / 2), (0, 0, h - t / 2), CABINET_RGBA)
    for side, y in (("l", hy - t / 2), ("r", -(hy - t / 2))):
        add_box(cabinet, f"cabinet_side_{side}", (hx, t / 2, h / 2), (0, y, h / 2), CABINET_RGBA)
    add_box(cabinet, "cabinet_back", (t / 2, hy, h / 2), (hx - t / 2, 0, h / 2), CABINET_RGBA)

    dhx, dhy = hx - 0.006, hy - t - 0.002
    panel_x = -hx - 0.003
    drawer = cabinet.add_body(name="drawer", pos=[0, 0, t])
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0],
                     range=[0, DRAWER["travel"]], damping=DRAWER["damping"], frictionloss=DRAWER["frictionloss"])
    floor_front, floor_back = panel_x + 0.003, dhx
    mid, half_len = (floor_back + floor_front) / 2, (floor_back - floor_front) / 2
    wall_h = DRAWER["wall_h"]
    add_box(drawer, "drawer_bottom", (half_len, dhy, 0.002), (mid, 0, 0.002), DRAWER_RGBA, friction=PROP_FRICTION)
    add_box(drawer, "drawer_back", (0.002, dhy, wall_h / 2), (dhx - 0.002, 0, 0.004 + wall_h / 2), DRAWER_RGBA)
    for side, y in (("l", dhy - 0.002), ("r", -(dhy - 0.002))):
        add_box(drawer, f"drawer_side_{side}", (half_len, 0.002, wall_h / 2), (mid, y, 0.004 + wall_h / 2), DRAWER_RGBA)
    panel_half_h = (h - 2 * t) / 2 - 0.001
    add_box(drawer, "drawer_front", (0.003, hy, panel_half_h), (panel_x, 0, panel_half_h), DRAWER_RGBA)
    # Bar handle on two stand-offs, far enough out for fingers to close around it.
    bar = HANDLE["bar_half"]
    bar_x = panel_x - 0.003 - HANDLE["standoff"] - bar[0]
    handle_z = panel_half_h
    add_box(drawer, "drawer_handle", bar, (bar_x, 0, handle_z), [0.2, 0.2, 0.2, 1], friction=PROP_FRICTION)
    for side, y in (("l", bar[1] - 0.003), ("r", -(bar[1] - 0.003))):
        add_box(drawer, f"handle_post_{side}", (HANDLE["standoff"] / 2, 0.003, 0.003),
                (panel_x - 0.003 - HANDLE["standoff"] / 2, y, handle_z), [0.2, 0.2, 0.2, 1])


def add_shell(spec, name, params, extra_geoms=()):
    """Open-topped thin-walled cylinder built from wall segments plus a base disc."""
    x, y = params["pos"]
    body = spec.worldbody.add_body(name=name, pos=[x, y, TABLE_TOP_Z])
    body.add_freejoint(name=f"{name}_free")
    r, height, wall = params["radius"], params["height"], params["wall"]
    mid_r = r - wall / 2
    seg_half = mid_r * np.tan(np.pi / SHELL_SEGMENTS) + 0.0006
    wall_mass = params["mass"] * 0.8 / SHELL_SEGMENTS
    for i in range(SHELL_SEGMENTS):
        theta = 2 * np.pi * i / SHELL_SEGMENTS
        body.add_geom(name=f"{name}_wall_{i}", type=BOX, size=[wall / 2, seg_half, height / 2],
                      pos=[mid_r * np.cos(theta), mid_r * np.sin(theta), height / 2],
                      quat=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)], mass=wall_mass,
                      friction=PROP_FRICTION, condim=4, rgba=params["rgba"])
    body.add_geom(name=f"{name}_base", type=CYLINDER, size=[r, SHELL_BASE_HALF], pos=[0, 0, SHELL_BASE_HALF],
                  mass=params["mass"] * 0.2, friction=PROP_FRICTION, condim=4, rgba=params["rgba"])
    for geom in extra_geoms:
        body.add_geom(**geom)
    return body


def add_water(spec):
    """Beads stacked inside the bottle; they settle before the episode starts."""
    bx, by = BOTTLE["pos"]
    inner = BOTTLE["radius"] - BOTTLE["wall"] - WATER["radius"] - 0.0005
    per_layer = 6
    for i in range(WATER["count"]):
        layer, slot = divmod(i, per_layer)
        theta = 2 * np.pi * slot / per_layer + layer * 0.5
        radial = inner * (0.55 if slot % 2 else 0.95)
        pos = [bx + radial * np.cos(theta), by + radial * np.sin(theta),
               TABLE_TOP_Z + 2 * SHELL_BASE_HALF + 0.001 + WATER["radius"] + layer * 2.05 * WATER["radius"]]
        bead = spec.worldbody.add_body(name=f"water_{i}", pos=pos)
        bead.add_freejoint(name=f"water_{i}_free")
        bead.add_geom(name=f"water_{i}_geom", type=SPHERE, size=[WATER["radius"], 0, 0], mass=WATER["mass"],
                      friction=[0.3, 0.001, 0.0001], condim=3, solref=WATER["solref"], rgba=WATER["rgba"])


def add_plate(spec):
    """Deep plate built like the other thin-walled containers (low wall on a base disc)."""
    add_shell(spec, "plate", PLATE)


def add_utensils(spec):
    """Spoon (handle + bowl) and fork (handle + three tines), long axis along x, in the drawer."""
    cx, cy = CABINET["pos"]
    floor_z = TABLE_TOP_Z + CABINET["wall"] + 0.004
    hx, hy, hz = UTENSIL_HANDLE
    for name, (dx, dy) in UTENSILS.items():
        body = spec.worldbody.add_body(name=name, pos=[cx + dx, cy + dy, floor_z + hz + 0.0005])
        body.add_freejoint(name=f"{name}_free")
        body.add_geom(name=f"{name}_geom", type=BOX, size=[hx, hy, hz], mass=0.018, friction=PROP_FRICTION,
                      condim=4, rgba=STEEL)
        if name == "spoon":
            # A thin, light bowl on a sturdy handle keeps the balance point near the handle.
            body.add_geom(name="spoon_bowl", type=ELLIPSOID, size=[0.014, 0.010, 0.0035],
                          pos=[hx + 0.012, 0, 0.0035 - hz], mass=0.004, friction=PROP_FRICTION, condim=4, rgba=STEEL)
        else:
            for k, ty in enumerate((-0.004, 0.0, 0.004)):
                body.add_geom(name=f"fork_tine_{k}", type=BOX, size=[0.010, 0.0012, 0.0012],
                              pos=[hx + 0.010, ty, 0.0012 - hz], mass=0.002, friction=PROP_FRICTION, condim=4,
                              rgba=STEEL)


def add_containers(spec):
    # Handle 10 mm thick (like a real mug's): the other arm pinches it to steady the mug
    # while it is poured into; on a 6 mm bar the jaws would sit almost fully shut.
    add_shell(spec, "mug", MUG, extra_geoms=[
        {"name": "mug_handle", "type": BOX, "size": [0.005, 0.004, 0.018],
         "pos": [0, MUG["radius"] + 0.008, MUG["height"] / 2], "mass": 0.01, "rgba": MUG["rgba"]},
    ])
    add_shell(spec, "bottle", BOTTLE)
    add_water(spec)


def add_task_markers(spec):
    world = spec.worldbody
    px, py = PLACE_TARGETS["plate"]
    world.add_geom(name="placemat", type=BOX, size=[0.07, 0.14, 0.0008], pos=[px, py, TABLE_TOP_Z + 0.0008],
                   rgba=[0.62, 0.2, 0.18, 1], contype=0, conaffinity=0)
    for name, (x, y) in PLACE_TARGETS.items():
        world.add_site(name=f"{name}_target", pos=[x, y, TABLE_TOP_Z + 0.002], size=[0.01, 0.0, 0.0],
                       rgba=[1.0, 0.85, 0.1, 0.6], group=4)


def add_cameras(spec):
    track = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    for name, pos in {
        "overhead": (-0.02, 0.0, TABLE_TOP_Z + 0.75),
        "operator": (-0.55, 0.0, TABLE_TOP_Z + 0.40),
        "front_high": (0.45, 0.0, TABLE_TOP_Z + 0.45),
    }.items():
        spec.worldbody.add_camera(name=name, pos=list(pos), mode=track, targetbody="workspace_center", fovy=55)


def realistic_gripper(spec, prefix, pad_friction=None):
    """Cap the gripper servo and give the fingers stated pads, on an already attached arm.

    The Menagerie file models the optional 12 V gripper servo at stall (2.94 N m, ~37 N at the
    fingertip) and hard fingers with mu = 1. A stock SO-101 has the 7.4 V STS3215 (1.62 N m
    stall) whose overload protection cuts the current after a few seconds, so grasps run on
    what it can sustain, and the printed fingers get soft pads with a stated friction.
    """
    mu = PAD_FRICTION[0] if pad_friction is None else pad_friction
    for actuator in spec.actuators:
        if actuator.name == prefix + "gripper":
            actuator.forcerange = [-GRIPPER_TORQUE, GRIPPER_TORQUE]
    finger_bodies = {prefix + "gripper", prefix + "moving_jaw_so101_v1"}
    for geom in spec.geoms:  # pad boxes, tip spheres, finger hulls and the gripper housing
        if geom.parent and geom.parent.name in finger_bodies and (geom.contype or geom.conaffinity):
            geom.friction = [mu, *PAD_FRICTION[1:]]


def attach_arm(world_spec, prefix, xy):
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    arm.meshdir = str(ASSETS_DIR)
    frame = world_spec.worldbody.add_frame(pos=[xy[0], xy[1], TABLE_TOP_Z])
    with contextlib.redirect_stderr(io.StringIO()):
        world_spec.attach(arm, prefix=prefix, frame=frame)
    realistic_gripper(world_spec, prefix)


def build_spec():
    spec = mujoco.MjSpec()
    spec.modelname = "bimanual_so101_dinner_table_contact"
    spec.meshdir = str(ASSETS_DIR)
    spec.compiler.degree = False
    # Grasp-grade contact settings (as in the Menagerie SO-101 model).
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    add_environment(spec)
    add_cabinet(spec)
    add_plate(spec)
    add_utensils(spec)
    add_containers(spec)
    add_task_markers(spec)
    add_cameras(spec)
    for prefix, xy in ARM_BASES.items():
        attach_arm(spec, prefix, xy)
    return spec


def main():
    spec = build_spec()
    with contextlib.redirect_stderr(io.StringIO()):
        model = spec.compile()
    OUT_XML.write_text(spec.to_xml(), encoding="utf-8")
    print(f"wrote {OUT_XML}")
    print(f"bodies={model.nbody} geoms={model.ngeom} joints={model.njnt} actuators={model.nu} "
          f"cameras={[model.camera(i).name for i in range(model.ncam)]} equality constraints={model.neq}")


if __name__ == "__main__":
    main()
