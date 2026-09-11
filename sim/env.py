"""Environment for the dual SO-101 dinner-table scene with contact-only manipulation.

Nothing attaches objects to the grippers. An object is "held" only while it
touches both the fixed and the moving jaw of the same arm, and it stays in the
hand only as long as friction holds it - so the per-seed mass and friction
randomisation genuinely changes how hard each grasp is.

Responsibilities:
  * domain randomisation per seed: object placement, mass and friction of
    every prop, lighting and table colour;
  * a 20 Hz control loop over 12 joint-position targets (6 per arm);
  * observations: joint state plus scene and wrist camera images;
  * task success from the physical state: items on their targets, upright and
    released, the drawer pulled out, and water beads inside the mug.
"""
from pathlib import Path

import mujoco
import numpy as np

from scene.build_scene import BOTTLE as SCENE_BOTTLE, MUG as SCENE_MUG
from sim.ik import ArmIK

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "scene" / "bimanual_table.xml"

ARMS = ("left_", "right_")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PROPS = ("plate", "mug", "bottle", "spoon", "fork")
UTENSILS = ("spoon", "fork")
PLACE_ITEMS = ("plate", "spoon", "fork", "mug")
FINGER_BODIES = ("gripper", "moving_jaw_so101_v1")
WATER_COUNT = 24

TABLE_TOP_Z = 0.40
GRIPPER_OPEN = 0.8
GRIPPER_SQUEEZE = -0.17
CONTROL_HZ = 20
HOME_OFFSET = np.array([0.10, 0.0, 0.14])  # home TCP relative to the arm base
SETTLE_STEPS = 750  # 1.5 s for the water beads and props to come to rest

# Success thresholds.
PLACE_TOL = 0.03
UPRIGHT_TOL_DEG = 12
DRAWER_OPEN_MIN = 0.045
POUR_FRACTION = 0.6
MAX_SPILLED = 2  # beads allowed outside both containers (a real pour may lose a drop)
BEAD_REST_SPEED = 0.05  # m/s; a bead faster than this is still falling or splashing
BOTTLE_INNER_RADIUS = SCENE_BOTTLE["radius"] - SCENE_BOTTLE["wall"]
BOTTLE_HEIGHT = SCENE_BOTTLE["height"]
LIFTED_MIN = 0.06  # the bottle must have been raised this far for "returned" to mean anything
MUG_INNER_RADIUS = SCENE_MUG["radius"] - SCENE_MUG["wall"]
MUG_HEIGHT = SCENE_MUG["height"]

# Randomisation ranges.
XY_JITTER = {"plate": 0.012, "mug": 0.012, "bottle": 0.012, "spoon": 0.004, "fork": 0.004}
UTENSIL_YAW = 0.08  # rad; the drawer leaves ~6 mm beside each utensil for the fingers
MASS_SCALE = (0.8, 1.2)
FRICTION_SCALE = (0.7, 1.3)
LIGHT_SCALE = (0.6, 1.2)


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class DinnerTableEnv:
    def __init__(self, obs_cameras=("overhead", "operator", "left_wrist_cam", "right_wrist_cam"),
                 obs_size=(128, 128)):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.substeps = int(round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))
        self.obs_cameras = obs_cameras
        self.obs_size = obs_size
        self._renderers = {}

        m = self.model
        self.act_idx = {a: np.array([m.actuator(a + j).id for j in JOINTS]) for a in ARMS}
        self.qpos_idx = {a: np.array([m.jnt_qposadr[m.joint(a + j).id] for j in JOINTS]) for a in ARMS}
        self.body_ids = {o: m.body(o).id for o in PROPS + ("drawer",)}
        self.tcp_site = {a: m.site(a + "gripperframe").id for a in ARMS}
        self.handle_geom = m.geom("drawer_handle").id
        self.table_body = m.body("table").id
        self.drawer_qadr = m.jnt_qposadr[m.joint("drawer_slide").id]
        self.free_qadr = {o: m.jnt_qposadr[m.joint(o + "_free").id] for o in PROPS}
        self.prop_geoms = {o: np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body_ids[o]])
                           for o in PROPS}
        self.finger_bodies = {a: {m.body(a + b).id for b in FINGER_BODIES} for a in ARMS}
        # Finger pad geoms: their friction (contact priority 1) is what every grasp uses.
        all_fingers = set().union(*self.finger_bodies.values())
        self.pad_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in all_fingers
                                   and (m.geom_contype[g] or m.geom_conaffinity[g])])
        self.fixed_jaw = {a: m.body(a + "gripper").id for a in ARMS}
        self.moving_jaw = {a: m.body(a + "moving_jaw_so101_v1").id for a in ARMS}
        self.water_ids = np.array([m.body(f"water_{i}").id for i in range(WATER_COUNT)])
        self.targets = {o: m.site(o + "_target").id for o in PLACE_ITEMS}
        self.ik = {a: ArmIK(m, a) for a in ARMS}

        self._qpos0 = m.qpos0.copy()
        self._mass0 = m.body_mass.copy()
        self._inertia0 = m.body_inertia.copy()
        self._friction0 = m.geom_friction.copy()
        self._light0 = m.light_diffuse.copy()
        self._headlight0 = m.vis.headlight.diffuse.copy()
        self._rgba0 = m.geom_rgba.copy()
        self.home = self._compute_home()
        self.reset(seed=0)

    # ------------------------------------------------------------------ setup
    def _compute_home(self):
        """Arm joint targets that park each TCP above its side of the table, gripper open."""
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        home = {}
        for arm in ARMS:
            base = data.xpos[self.model.body(arm + "base").id]
            q, _ = self.ik[arm].solve(data.qpos, base + HOME_OFFSET, approach=None)
            home[arm] = np.append(q, GRIPPER_OPEN)
        return home

    def reset(self, seed=0):
        m, d = self.model, self.data
        rng = np.random.default_rng(seed)
        m.body_mass[:] = self._mass0
        m.body_inertia[:] = self._inertia0
        m.geom_friction[:] = self._friction0
        m.light_diffuse[:] = self._light0
        m.vis.headlight.diffuse[:] = self._headlight0
        m.geom_rgba[:] = self._rgba0
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self._qpos0
        for arm in ARMS:
            d.qpos[self.qpos_idx[arm]] = self.home[arm]
            d.ctrl[self.act_idx[arm]] = self.home[arm]

        variation = {"seed": seed}
        if seed:  # seed 0 is the nominal scene
            for obj in PROPS:
                adr = self.free_qadr[obj]
                jitter = XY_JITTER[obj]
                offset = rng.uniform(-jitter, jitter, 2)
                d.qpos[adr:adr + 2] += offset
                if obj == "bottle":  # the water travels with the bottle
                    for bead in self.water_ids:
                        bead_adr = m.jnt_qposadr[m.body_jntadr[bead]]
                        d.qpos[bead_adr:bead_adr + 2] += offset
                if obj in UTENSILS:
                    turned = np.zeros(4)
                    mujoco.mju_mulQuat(turned, _yaw_quat(rng.uniform(-UTENSIL_YAW, UTENSIL_YAW)),
                                       self._qpos0[adr + 3:adr + 7].copy())
                    d.qpos[adr + 3:adr + 7] = turned
                mass_s, fric_s = rng.uniform(*MASS_SCALE), rng.uniform(*FRICTION_SCALE)
                body = self.body_ids[obj]
                m.body_mass[body] *= mass_s
                m.body_inertia[body] *= mass_s
                m.geom_friction[self.prop_geoms[obj], 0] *= fric_s
                variation[obj] = {"mass_scale": round(mass_s, 3), "friction_scale": round(fric_s, 3)}
            # Grip friction: the pad geoms win every finger-object contact (priority), so the
            # per-object scale above only changes object-on-table sliding; this one changes grasps.
            grip_s = rng.uniform(*FRICTION_SCALE)
            m.geom_friction[self.pad_geoms, 0] *= grip_s
            variation["grip_friction_scale"] = round(grip_s, 3)
            mujoco.mj_setConst(m, d)  # refresh mass-derived constants after the mass scaling
            light_s = rng.uniform(*LIGHT_SCALE)
            m.light_diffuse[:] *= light_s
            m.vis.headlight.diffuse[:] *= light_s
            table = m.geom("table_top").id
            m.geom_rgba[table, :3] = np.clip(self._rgba0[table, :3] * rng.uniform(0.75, 1.2, 3), 0, 1)
            variation["light_scale"] = round(light_s, 3)
        self.variation = variation
        self.drawer_max = 0.0
        self.bottle_lift_max = 0.0
        self.time_step = 0
        for i, arm in enumerate(ARMS):  # IK restarts draw from a per-episode stream: runs are repeatable
            self.ik[arm].rng = np.random.default_rng(2 * seed + i)
        mujoco.mj_forward(m, d)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(m, d)
        self.bottle_start = self.object_frame("bottle")[0].copy()
        return self.observe()

    # ---------------------------------------------------------------- control
    def step(self, action):
        """Apply 12 joint targets (left 6 then right 6) for one control period."""
        action = np.asarray(action, dtype=float)
        for i, arm in enumerate(ARMS):
            self.data.ctrl[self.act_idx[arm]] = action[6 * i:6 * i + 6]
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.drawer_max = max(self.drawer_max, float(self.data.qpos[self.drawer_qadr]))
        self.bottle_lift_max = max(self.bottle_lift_max, float(self.data.xpos[self.body_ids["bottle"]][2]))
        self.time_step += 1
        return self.observe()

    def hold(self, seconds=1.0):
        """Keep the current joint targets for ``seconds`` so everything comes to rest before scoring."""
        for _ in range(int(round(seconds * CONTROL_HZ))):
            self.step(self.data.ctrl[np.concatenate([self.act_idx[a] for a in ARMS])])

    def arm_qpos(self, arm):
        return self.data.qpos[self.qpos_idx[arm]].copy()

    def joint_state(self):
        return np.concatenate([self.arm_qpos(a) for a in ARMS])

    def tcp(self, arm):
        return self.data.site_xpos[self.tcp_site[arm]].copy()

    def object_frame(self, obj):
        """(origin, rotation) of a prop body; shells have their origin at the base centre."""
        body = self.body_ids[obj]
        return self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()

    def handle_point(self):
        return self.data.geom_xpos[self.handle_geom].copy()

    def arm_base(self, arm):
        return self.data.xpos[self.model.body(arm + "base").id].copy()

    # --------------------------------------------------------------- contact
    def _contact_bodies(self, obj):
        """Set of body ids touching ``obj``."""
        body = self.body_ids[obj]
        touching = set()
        for k in range(self.data.ncon):
            contact = self.data.contact[k]
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]
            if b1 == body:
                touching.add(b2)
            elif b2 == body:
                touching.add(b1)
        return touching

    def holder(self, obj):
        """Arm whose fixed and moving jaw both touch ``obj``, or None."""
        touching = self._contact_bodies(obj)
        for arm in ARMS:
            if self.fixed_jaw[arm] in touching and self.moving_jaw[arm] in touching:
                return arm
        return None

    def supported(self, obj):
        """True if ``obj`` touches anything other than the fingers (table, plate, drawer, ...)."""
        fingers = set().union(*self.finger_bodies.values())
        return bool(self._contact_bodies(obj) - fingers)

    def centre_of_mass(self, obj):
        return self.data.xipos[self.body_ids[obj]].copy()

    def touched_by_gripper(self, obj):
        touching = self._contact_bodies(obj)
        return any(touching & self.finger_bodies[arm] for arm in ARMS)

    # ----------------------------------------------------------- observation
    def render(self, camera, size=(480, 640)):
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.model, size[0], size[1])
        renderer = self._renderers[size]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def observe(self):
        obs = {"state": self.joint_state()}
        for cam in self.obs_cameras:
            obs[cam] = self.render(cam, self.obs_size)
        return obs

    # --------------------------------------------------------------- success
    def tilt_deg(self, obj):
        z_axis = self.data.xmat[self.body_ids[obj]].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0))))

    def _beads_inside(self, obj, radius, height):
        origin, rot = self.object_frame(obj)
        local = (self.data.xpos[self.water_ids] - origin) @ rot
        radial = np.linalg.norm(local[:, :2], axis=1)
        return (radial < radius) & (local[:, 2] > -0.002) & (local[:, 2] < height)

    def water_census(self):
        """Beads resting in the mug, still in the bottle, and spilled anywhere else."""
        speeds = np.array([np.linalg.norm(self.data.qvel[self.model.body_dofadr[b]:self.model.body_dofadr[b] + 3])
                           for b in self.water_ids])
        resting = speeds < BEAD_REST_SPEED
        in_mug = self._beads_inside("mug", MUG_INNER_RADIUS, MUG_HEIGHT) & resting
        in_bottle = self._beads_inside("bottle", BOTTLE_INNER_RADIUS, BOTTLE_HEIGHT)
        return {"mug": int(in_mug.sum()), "bottle": int((in_bottle & ~in_mug).sum()),
                "spilled": int((~in_mug & ~in_bottle).sum())}

    def water_in_mug(self):
        return self.water_census()["mug"]

    def _rests_on_table(self, obj):
        """The object touches the table (or placemat) and nothing else but, possibly, the fingers."""
        ignored = set().union(*self.finger_bodies.values()) | set(self.water_ids.tolist())
        touching = self._contact_bodies(obj) - ignored
        return bool(touching) and touching <= {self.table_body}

    def _placed(self, obj):
        origin, _ = self.object_frame(obj)
        target = self.data.site_xpos[self.targets[obj]]
        upright = obj in UTENSILS or self.tilt_deg(obj) < UPRIGHT_TOL_DEG
        return bool(np.linalg.norm(origin[:2] - target[:2]) < PLACE_TOL and self._rests_on_table(obj)
                    and upright and not self.touched_by_gripper(obj))

    def _poured(self):
        """Most of the water is in the mug and almost none on the table."""
        census = self.water_census()
        return bool(census["mug"] >= POUR_FRACTION * WATER_COUNT and census["spilled"] <= MAX_SPILLED)

    def _bottle_returned(self):
        """The bottle was lifted at some point and now stands upright, released, where it started."""
        origin = self.object_frame("bottle")[0]
        lifted = self.bottle_lift_max > TABLE_TOP_Z + LIFTED_MIN
        return bool(lifted and np.linalg.norm(origin[:2] - self.bottle_start[:2]) < PLACE_TOL
                    and self._rests_on_table("bottle") and self.tilt_deg("bottle") < UPRIGHT_TOL_DEG
                    and not self.touched_by_gripper("bottle"))

    def success(self):
        """Sub-goal checks from the current physical state (the drawer must still be open now)."""
        checks = {
            "drawer_open": bool(self.data.qpos[self.drawer_qadr] > DRAWER_OPEN_MIN),
            **{f"{o}_placed": self._placed(o) for o in PLACE_ITEMS},
            "poured": self._poured(),
            "bottle_returned": self._bottle_returned(),
        }
        checks["all"] = all(checks.values())
        return checks

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
