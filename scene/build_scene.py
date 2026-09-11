"""Build the dual SO-101 dinner-table MuJoCo scene.

Composes two copies of TheRobotStudio SO-101 model (prefixed ``left_`` and
``right_``) with a table, a drawer cabinet and table-setting props, then writes
one self-contained MJCF file to ``scene/bimanual_table.xml``.

World frame: +x points away from the arm bases toward the cabinet, +y is the
left arm's side, z is up. All lengths in metres.

Run:  .venv\\Scripts\\python.exe scene\\build_scene.py
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SO101_DIR = ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101"
ARM_XML = SO101_DIR / "so101_new_calib.xml"
ASSETS_DIR = SO101_DIR / "assets"
OUT_XML = ROOT / "scene" / "bimanual_table.xml"

TABLE_TOP_Z = 0.40
# Arm base (x, y) on the table top and mount quaternion (w, x, y, z).
ARM_MOUNTS = {
    "left_": ((-0.20, 0.16), (1.0, 0.0, 0.0, 0.0)),
    "right_": ((-0.20, -0.16), (1.0, 0.0, 0.0, 0.0)),
}

# Cabinet: half-depth (x), half-width (y), inner height, wall thickness.
CABINET_POS = (0.15, 0.0, TABLE_TOP_Z)
CAB_HX, CAB_HY, CAB_H, WALL = 0.06, 0.075, 0.07, 0.005
DRAWER_TRAVEL = 0.10
# Sliding friction high enough that a passing arm does not push the drawer shut.
DRAWER_DAMPING, DRAWER_FRICTIONLOSS = 0.5, 0.6

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
PLANE = mujoco.mjtGeom.mjGEOM_PLANE

WOOD = [0.55, 0.38, 0.24, 1.0]
CABINET_RGBA = [0.85, 0.85, 0.82, 1.0]
DRAWER_RGBA = [0.75, 0.6, 0.45, 1.0]
PROP_FRICTION = [1.5, 0.05, 0.001]

# Free props on the table: name -> ((x, y), geom kwargs, half height, mass).
PROP_START = {
    "plate": ((-0.01, -0.14), dict(type=CYLINDER, size=[0.05, 0.004], rgba=[0.95, 0.95, 0.95, 1]),
              0.004, 0.15),
    "mug": ((0.03, 0.13), dict(type=CYLINDER, size=[0.022, 0.035], rgba=[0.2, 0.45, 0.8, 1]),
            0.035, 0.12),
    "bottle": ((0.0, -0.22), dict(type=CYLINDER, size=[0.018, 0.06], rgba=[0.3, 0.75, 0.4, 0.9]),
               0.06, 0.2),
}
# Utensils lie across the front of the drawer (long axis along y) so they
# clear the cabinet top once the drawer is pulled out. x is relative to the cabinet.
UTENSIL_START = {"spoon": (-0.044, [0.75, 0.75, 0.8, 1]), "fork": (-0.020, [0.6, 0.6, 0.68, 1])}
UTENSIL_HALF = [0.035, 0.006, 0.003]
UTENSIL_QUAT = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))

# Where each item belongs once the table is set, (x, y) on the table top.
PLACE_TARGETS = {
    "plate": (-0.08, 0.0),
    "spoon": (-0.08, 0.085),
    "fork": (-0.08, -0.085),
    "mug": (-0.03, 0.12),
}
# Bodies either gripper may hold through a weld (or connect) constraint.
GRASPABLE = ("plate", "mug", "bottle", "spoon", "fork", "drawer")
GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1")


def add_box(body, name, half_size, pos, rgba):
    return body.add_geom(name=name, type=BOX, size=list(half_size), pos=list(pos), rgba=rgba)


def add_environment(spec):
    """Floor, table, lighting and a target body the cameras look at."""
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    # Offscreen framebuffer large enough for 1080p stills and demo video.
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    world = spec.worldbody
    world.add_geom(name="floor", type=PLANE, size=[3, 3, 0.05], rgba=[0.25, 0.3, 0.35, 1])
    world.add_light(name="key_light", pos=[0.0, 0.0, 2.0], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5])

    table = world.add_body(name="table", pos=[0.0, 0.0, 0.0])
    add_box(table, "table_top", (0.42, 0.40, 0.02), (0, 0, TABLE_TOP_Z - 0.02), WOOD)
    leg_half_h = (TABLE_TOP_Z - 0.04) / 2
    for i, (lx, ly) in enumerate([(0.38, 0.36), (0.38, -0.36), (-0.38, 0.36), (-0.38, -0.36)]):
        add_box(table, f"table_leg_{i}", (0.02, 0.02, leg_half_h), (lx, ly, leg_half_h), WOOD)

    world.add_body(name="workspace_center", pos=[0.0, 0.0, TABLE_TOP_Z + 0.03])


def add_cabinet(spec):
    """Static cabinet open toward -x, with a drawer that slides out toward the arms."""
    cabinet = spec.worldbody.add_body(name="cabinet", pos=list(CABINET_POS))
    add_box(cabinet, "cabinet_bottom", (CAB_HX, CAB_HY, WALL), (0, 0, WALL), CABINET_RGBA)
    add_box(cabinet, "cabinet_top", (CAB_HX, CAB_HY, WALL), (0, 0, CAB_H + WALL), CABINET_RGBA)
    for side, y in (("l", CAB_HY - WALL), ("r", -(CAB_HY - WALL))):
        add_box(cabinet, f"cabinet_side_{side}", (CAB_HX, WALL, CAB_H / 2), (0, y, CAB_H / 2 + WALL),
                CABINET_RGBA)
    add_box(cabinet, "cabinet_back", (WALL, CAB_HY, CAB_H / 2), (CAB_HX - WALL, 0, CAB_H / 2 + WALL),
            CABINET_RGBA)

    # Drawer box sits on the cabinet bottom; its front panel stays outside the
    # cabinet so it never overlaps the top plate.
    dhx, dhy, wall_half_h = CAB_HX - 0.012, CAB_HY - 2 * WALL - 0.003, 0.015
    panel_half_h = (CAB_H - 2 * WALL) / 2 - 0.002
    panel_x = -CAB_HX - 0.005
    drawer = cabinet.add_body(name="drawer", pos=[0, 0, 2 * WALL])
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0],
                     range=[0, DRAWER_TRAVEL], damping=DRAWER_DAMPING, frictionloss=DRAWER_FRICTIONLOSS)
    # The floor runs all the way to the front panel so nothing slips through a gap.
    floor_front, floor_back = panel_x + 0.004, dhx
    add_box(drawer, "drawer_bottom", ((floor_back - floor_front) / 2, dhy, 0.003),
            ((floor_back + floor_front) / 2, 0, 0.003), DRAWER_RGBA)
    add_box(drawer, "drawer_back", (0.003, dhy, wall_half_h), (dhx - 0.003, 0, 0.006 + wall_half_h),
            DRAWER_RGBA)
    for side, y in (("l", dhy - 0.003), ("r", -(dhy - 0.003))):
        add_box(drawer, f"drawer_side_{side}", ((floor_back - floor_front) / 2, 0.003, wall_half_h),
                ((floor_back + floor_front) / 2, y, 0.006 + wall_half_h), DRAWER_RGBA)
    add_box(drawer, "drawer_front", (0.004, CAB_HY, panel_half_h), (panel_x, 0, panel_half_h),
            DRAWER_RGBA)
    add_box(drawer, "drawer_handle", (0.008, 0.025, 0.006), (-CAB_HX - 0.017, 0, panel_half_h),
            [0.2, 0.2, 0.2, 1])


def add_free_prop(spec, name, geom_kwargs, pos, mass, quat=(1.0, 0.0, 0.0, 0.0)):
    body = spec.worldbody.add_body(name=name, pos=list(pos), quat=list(quat))
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(name=f"{name}_geom", mass=mass, friction=PROP_FRICTION, condim=4, **geom_kwargs)
    return body


def add_props(spec):
    """Table-setting objects: plate, mug, bottle on the table; spoon and fork in the drawer."""
    for name, (xy, geom, half_h, mass) in PROP_START.items():
        add_free_prop(spec, name, geom, (xy[0], xy[1], TABLE_TOP_Z + half_h + 0.001), mass)

    cx, cy, cz = CABINET_POS
    drawer_floor = cz + 2 * WALL + 0.006
    for name, (dx, rgba) in UTENSIL_START.items():
        add_free_prop(spec, name, dict(type=BOX, size=UTENSIL_HALF, rgba=rgba),
                      (cx + dx, cy, drawer_floor + UTENSIL_HALF[2] + 0.001), 0.03, UTENSIL_QUAT)


def add_task_markers(spec):
    """Visual-only placemat plus one site per place target (hidden group 4)."""
    world = spec.worldbody
    plate_x, plate_y = PLACE_TARGETS["plate"]
    world.add_geom(name="placemat", type=BOX, size=[0.075, 0.15, 0.0008],
                   pos=[plate_x, plate_y, TABLE_TOP_Z + 0.0008], rgba=[0.62, 0.2, 0.18, 1],
                   contype=0, conaffinity=0)
    for name, (x, y) in PLACE_TARGETS.items():
        world.add_site(name=f"{name}_target", pos=[x, y, TABLE_TOP_Z + 0.002], size=[0.01, 0.0, 0.0],
                       rgba=[1.0, 0.85, 0.1, 0.6], group=4)


def add_cameras(spec):
    """Fixed scene cameras that track the workspace centre."""
    track = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    for name, pos in {
        "overhead": (0.0, 0.0, TABLE_TOP_Z + 0.85),
        "operator": (-0.60, 0.0, TABLE_TOP_Z + 0.45),
        "front_high": (0.55, 0.0, TABLE_TOP_Z + 0.55),
    }.items():
        spec.worldbody.add_camera(name=name, pos=list(pos), mode=track,
                                  targetbody="workspace_center", fovy=55)


def attach_arm(world_spec, prefix, xy, quat):
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    arm.meshdir = str(ASSETS_DIR)
    frame = world_spec.worldbody.add_frame(pos=[xy[0], xy[1], TABLE_TOP_Z], quat=list(quat))
    if hasattr(world_spec, "attach"):
        world_spec.attach(arm, prefix=prefix, frame=frame)
    else:  # MuJoCo < 3.3 API
        frame.attach_body(arm.worldbody.first_body(), prefix, "")


def add_grasp_constraints(spec):
    """One inactive constraint per (arm, graspable body); the env switches them on when a gripper closes.

    Gripper links never collide with graspable bodies so a held object does not
    fight the jaws; the constraint is what actually carries the object.
    """
    for prefix in ARM_MOUNTS:
        for obj in GRASPABLE:
            # The drawer is already constrained by its slide joint, so it gets a
            # position-only connect; a full weld would fight the slide.
            eq_type = mujoco.mjtEq.mjEQ_CONNECT if obj == "drawer" else mujoco.mjtEq.mjEQ_WELD
            spec.add_equality(name=f"{prefix}grasp_{obj}", type=eq_type,
                              objtype=mujoco.mjtObj.mjOBJ_BODY, name1=prefix + "gripper",
                              name2=obj, active=False, solref=[0.01, 1.0])
            for link in GRIPPER_BODIES:
                spec.add_exclude(bodyname1=prefix + link, bodyname2=obj)


def build_spec():
    spec = mujoco.MjSpec()
    spec.modelname = "bimanual_so101_dinner_table"
    spec.meshdir = str(ASSETS_DIR)
    spec.compiler.degree = False  # the SO-101 model stores joint ranges in radians
    spec.option.timestep = 0.002
    add_environment(spec)
    add_cabinet(spec)
    add_props(spec)
    add_task_markers(spec)
    add_cameras(spec)
    for prefix, (xy, quat) in ARM_MOUNTS.items():
        attach_arm(spec, prefix, xy, quat)
    add_grasp_constraints(spec)
    return spec


def main():
    spec = build_spec()
    model = spec.compile()
    OUT_XML.write_text(spec.to_xml(), encoding="utf-8")
    actuators = [model.actuator(i).name for i in range(model.nu)]
    cameras = [model.camera(i).name for i in range(model.ncam)]
    print(f"wrote {OUT_XML}")
    print(f"bodies={model.nbody} joints={model.njnt} actuators={model.nu} cameras={model.ncam}")
    print("actuators:", actuators)
    print("cameras:", cameras)


if __name__ == "__main__":
    main()
