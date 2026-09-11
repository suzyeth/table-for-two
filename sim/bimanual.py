"""Two-handed carrying: both arms hold one object and move it as one rigid body.

A plate pinched by one wall hangs 6-8 deg tilted in the fingers (its centre of mass is
4.5 cm from the pinch, and no single pinch on a 3 mm wall can hold that torque), then
touches down on one edge. Held by opposite walls, the two torques cancel and it stays
level - which is how people carry a plate too.

Both arms follow the same Cartesian path for the object's origin: every waypoint is
solved for each arm and both interpolate over the same number of control steps, so the
hands stay the same distance apart throughout. The object is set down with the same
touch-sensing descent as a one-handed place.
"""
import numpy as np

from scene.build_scene import TABLE_TOP_Z
from sim.grasping import DOWN, UP, plate_wall_grasp
from sim.skills import PRESS_STEPS, REST_HEIGHT, TOUCH_OVERSHOOT, TOUCH_STEP

CARRY_LIFT = 0.04
CARRY_ABOVE = 0.03  # height above the rest pose where the shared descent starts
CARRY_STEP = 0.006  # waypoint spacing (m); with CARRY_TICKS control steps each, ~6 cm/s
CARRY_TICKS = 2
TOUCH_TICKS = 2


def lockstep(*generators):
    """Advance several arm generators together; an arm whose generator has finished holds still."""
    active = list(generators)
    while active:
        still = []
        for gen in active:
            try:
                next(gen)
                still.append(gen)
            except StopIteration:
                pass
        active = still
        if active:
            yield


def _shared_step(skills, orients, target, ticks):
    """Solve every arm for the same object-origin ``target`` and interpolate them together."""
    starts, goals = {}, {}
    for arm, sk in skills.items():
        q, _ = sk.solve(target, orients[arm])
        starts[arm] = sk.cmd.copy()
        goals[arm] = sk.cmd.copy()
        goals[arm][:5] = q
    for tick in range(1, ticks + 1):
        for arm, sk in skills.items():
            sk.cmd = starts[arm] + (goals[arm] - starts[arm]) * tick / ticks
        yield


def _move_together(skills, orients, start, end):
    count = max(1, int(np.ceil(np.linalg.norm(end - start) / CARRY_STEP)))
    for k in range(1, count + 1):
        yield from _shared_step(skills, orients, start + (end - start) * k / count, CARRY_TICKS)


def carry_together(env, skills, obj, target_xy, grip_dirs):
    """Pick ``obj`` with every arm in ``skills`` (arm -> ArmSkills), carry it level to ``target_xy``
    and set it down; ``grip_dirs`` maps each arm to the world direction of its grip from the centre."""
    grasp_fns = {arm: (lambda arm=arm: plate_wall_grasp(env, obj, grip_dirs[arm])) for arm in skills}
    yield from lockstep(*(sk.pick(obj, lift=0, grasp_fn=grasp_fns[arm]) for arm, sk in skills.items()))
    orients = {arm: {"approach": DOWN, "closing": sk.orient["closing"], "point": sk.held_point(obj)}
               for arm, sk in skills.items()}
    for arm, sk in skills.items():
        sk.orient = orients[arm]
    origin = env.object_frame(obj)[0]
    rest = np.array([target_xy[0], target_xy[1], TABLE_TOP_Z + REST_HEIGHT[obj]])
    lifted = origin + UP * CARRY_LIFT
    above = rest + UP * CARRY_ABOVE
    yield from _move_together(skills, orients, origin, lifted)
    yield from _move_together(skills, orients, lifted, above)
    # Touch-sensing descent, shared by both hands.
    floor = rest - UP * TOUCH_OVERSHOOT
    count = max(1, int(np.ceil(np.linalg.norm(above - floor) / TOUCH_STEP)))
    pressed = 0
    for k in range(1, count + 1):
        if env.supported(obj):
            if pressed >= PRESS_STEPS:
                break
            pressed += 1
        yield from _shared_step(skills, orients, above + (floor - above) * k / count, TOUCH_TICKS)
    if not env.supported(obj):
        for sk in skills.values():
            sk.warnings.append(f"{sk.arm} set {obj} down (two-handed) without it touching a support")
    yield from lockstep(*(sk.release_and_retreat() for sk in skills.values()))
