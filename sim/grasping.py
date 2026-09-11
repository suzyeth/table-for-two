"""Gripper geometry and top-down grasp specs for contact-only grasping.

Menagerie SO-101 gripper frame: the fixed jaw's pad plane is at x = -0.0081 and
the moving jaw closes toward it along +x; fingertips sit near z = -0.10 and the
``gripperframe`` site at (0.012, -0.0002, -0.0981).

A grasp is a *pinch point* - a point in the gripper frame that must coincide
with the grasp centre on the object, set so the object sits just off the fixed
pad and the fingertips reach a chosen depth past the centre - plus the jaw
closing direction. The moving jaw then closes past contact and the object is
held only by the resulting normal force and friction. Every spec here was
checked one object at a time in ``tools/grasp_lab.py``.
"""
from dataclasses import dataclass

import numpy as np

from scene.build_scene import HANDLE, MUG, PLATE, UTENSIL_HANDLE
from sim.env import GRIPPER_OPEN

FIXED_PAD_X = -0.0081
# Jaw angle for the drawer handle, from the collision geometry: at 0 rad the pads are
# 14.7 mm apart (bar 8 mm + 3 mm clearance leaves ~4 mm) and the moving finger's back is
# 20-25 mm out, inside the 28 mm gap to the drawer front; wider and it lands on the front.
HANDLE_OPEN = 0.0
UTENSIL_OPEN = 0.05  # pads ~19 mm apart: ~7 mm spare beside the 10 mm handle, finger back ~27 mm out
UTENSIL_CLEARANCE = 0.002
# The jaws' collision bodies reach 8 mm below the TCP site (-106.2 mm moving, -104.8 mm
# fixed, site at -98.1 mm). With the pinch point 4 mm *below* the site, the handle centre
# (5 mm up) puts the lowest jaw point 1 mm above the surface and the pads over 5-7 mm of
# the 10 mm handle.
UTENSIL_OVERLAP = -0.004
PLATE_PINCH_BELOW_TOP = 0.004
PLATE_OVERLAP = -0.003
UTENSIL_NEIGHBOUR = 0.05  # closer than this, the other utensil decides which way the jaws face
UTENSIL_PAD_HALF = 0.004  # pads are 8 mm wide along the handle
SITE_LOCAL = np.array([0.012, -0.000218, -0.098127])
UP = np.array([0.0, 0.0, 1.0])
DOWN = -UP
PAD_CLEARANCE = 0.004  # gap between the fixed pad and the object before closing
PRE_HEIGHT = 0.03  # vertical descent onto every top-down grasp

UTENSILS = ("spoon", "fork")


@dataclass(frozen=True)
class Grasp:
    centre: np.ndarray  # world point that ends up between the pads
    closing: np.ndarray  # world direction fixed jaw -> moving jaw
    pinch: np.ndarray  # gripper-frame point placed at ``centre``
    symmetric: bool  # True if the jaws may equally close the other way round
    open_to: float = GRIPPER_OPEN  # jaw opening on the way in


def pinch_point(half_width, overlap, clearance=PAD_CLEARANCE):
    """Gripper-frame point for an object ``2*half_width`` wide, fingertips ``overlap`` past its centre."""
    return np.array([FIXED_PAD_X + clearance + half_width, 0.0, SITE_LOCAL[2] + overlap])


def horizontal(vector):
    flat = np.array(vector, dtype=float)
    flat[2] = 0.0
    return flat / np.linalg.norm(flat)


def utensil_axis(env, obj, toward=None, flat=True):
    """Long axis of a utensil (body x, handle -> head), flipped to point along ``toward``.

    ``flat`` drops the vertical part (for jaw directions); ``flat=False`` keeps the true 3-D
    axis, e.g. to find a point on a handle that sags in the other hand.
    """
    axis = env.object_frame(obj)[1][:, 0].copy()
    if flat:
        axis = horizontal(axis)
    if toward is not None and axis @ np.asarray(toward) < 0:
        axis = -axis
    return axis


def plate_wall_grasp(env, obj, direction):
    """Top-down pinch on the plate wall at horizontal world ``direction`` from its centre, for
    a two-handed carry: the thin fixed finger goes inside the plate and the moving jaw stays
    outside (moving jaws inside the plate meet in the middle when both hands hold it)."""
    origin, _ = env.object_frame(obj)
    out = horizontal(direction)
    wall_mid = PLATE["radius"] - PLATE["wall"] / 2
    centre = origin + UP * (PLATE["height"] - PLATE_PINCH_BELOW_TOP) + out * wall_mid
    return Grasp(centre, out, pinch_point(PLATE["wall"] / 2, PLATE_OVERLAP), False)


def balance_offset(env, obj):
    """Signed distance from a utensil's handle centre to the point over its centre of mass,
    along the head direction, kept on the handle so both pads land on it."""
    head = env.object_frame(obj)[1][:, 0]
    offset = float((env.centre_of_mass(obj) - env.object_frame(obj)[0]) @ head)
    limit = UTENSIL_HANDLE[0] - UTENSIL_PAD_HALF
    return float(np.clip(offset, -limit, limit))


def top_down_grasp(env, arm, obj, along=0.0, toward=None):
    """Top-down grasp on ``obj`` from the current (actual) object pose.

    ``along`` shifts a utensil grasp along its handle, measured in the direction
    ``toward`` (used by the hand-over so each hand takes its own end);
    ``along="balance"`` grips right over the utensil's centre of mass so it hangs level.
    """
    origin, rot = env.object_frame(obj)
    if obj in UTENSILS and along == "balance":
        along, toward = balance_offset(env, obj), rot[:, 0]
    if obj in UTENSILS:
        # The jaws stop ~1 mm above the surface beside the handle. They open only part way;
        # if the other utensil still lies close by, the thin fixed finger goes on its side
        # (the moving finger's back reaches ~2.5 cm out).
        axis = utensil_axis(env, obj, toward)
        closing = np.cross(UP, axis)
        pinch = pinch_point(UTENSIL_HANDLE[1], UTENSIL_OVERLAP, clearance=UTENSIL_CLEARANCE)
        other = env.object_frame(UTENSILS[1 - UTENSILS.index(obj)])[0]
        away = origin - other
        away[2] = 0.0
        centre = origin + utensil_axis(env, obj, toward, flat=False) * along
        if np.linalg.norm(away) < UTENSIL_NEIGHBOUR:
            closing = closing if closing @ away > 0 else -closing
            return Grasp(centre, closing, pinch, False, open_to=UTENSIL_OPEN)
        return Grasp(centre, closing, pinch, True, open_to=UTENSIL_OPEN)
    if obj == "mug":
        # Across the body just below the rim; the handle sticks out along the mug's +y,
        # beside the jaws rather than between them.
        centre = origin + UP * (MUG["height"] - 0.012)
        return Grasp(centre, horizontal(rot[:, 0]), pinch_point(MUG["radius"], 0.010), True)
    if obj == "plate":
        # Pinch the wall nearest the arm, fixed pad outside and moving jaw inside the plate.
        # The moving jaw reaches 8.1 mm + overlap below the pinch point, and the plate floor
        # is 8 mm thick: pinching 16 mm up the 20 mm wall with the pinch 3 mm below the TCP
        # keeps the jaw ~3 mm off the floor (a jaw that jams on the floor never closes) while
        # the pads still cover 6.5-7 mm of wall.
        toward_centre = horizontal(origin - env.arm_base(arm))
        wall_mid = PLATE["radius"] - PLATE["wall"] / 2
        centre = origin + UP * (PLATE["height"] - PLATE_PINCH_BELOW_TOP) - toward_centre * wall_mid
        return Grasp(centre, toward_centre, pinch_point(PLATE["wall"] / 2, PLATE_OVERLAP), False)
    if obj == "drawer":
        # Fixed finger on the arm side of the bar, moving jaw dropped into the gap between
        # bar and drawer front (opened only part way so it fits): pulling toward the arms
        # presses the moving jaw against the bar. Closing the other way round would need
        # more wrist roll than the SO-101 has.
        return Grasp(env.handle_point(), np.array([1.0, 0.0, 0.0]),
                     pinch_point(HANDLE["bar_half"][0], 0.006, clearance=0.003), False, open_to=HANDLE_OPEN)
    raise ValueError(f"no top-down grasp defined for '{obj}'")
