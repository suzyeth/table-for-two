"""Environment wrapper for the dual SO-101 dinner-table scene.

Responsibilities:
  * domain randomisation per seed (object placement, mass, friction, lighting,
    table colour) - the robustness criterion asks for 10 randomised seeds;
  * a 20 Hz control loop over 12 joint-position targets (6 per arm);
  * constraint-assisted grasping: when a gripper is commanded closed near a
    graspable body, the matching weld (or, for the drawer, connect) constraint
    is switched on at the current relative pose, and off when the gripper opens;
  * observations (joint state + camera images) and task-success checks.

Grasp selection: a caller (scripted skill or the executor running a learned
policy) may declare which object an arm intends to grasp; that object wins if
it is within INTENT_RADIUS, otherwise the nearest graspable within
GRASP_RADIUS is taken. This matters for the spoon and fork, which lie a couple
of centimetres apart in the drawer.
"""
from pathlib import Path

import mujoco
import numpy as np

from sim.ik import ArmIK

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "scene" / "bimanual_table.xml"

ARMS = ("left_", "right_")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
GRASPABLE = ("plate", "mug", "bottle", "spoon", "fork", "drawer")
PROPS = ("plate", "mug", "bottle", "spoon", "fork")
UTENSILS = ("spoon", "fork")
PLACE_ITEMS = ("plate", "spoon", "fork", "mug")

TABLE_TOP_Z = 0.40
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.05
GRASP_CLOSE_THRESHOLD = 0.3  # commanded gripper below this counts as "closing"
GRASP_RADIUS = 0.045
INTENT_RADIUS = 0.06
CONTROL_HZ = 20
HOME_OFFSET = np.array([0.12, 0.0, 0.16])  # home TCP relative to the arm base

# Success thresholds.
PLACE_TOL = 0.03
DRAWER_OPEN_MIN = 0.05
POUR_TILT_MIN = np.deg2rad(60)
POUR_XY_TOL = 0.04
POUR_SECONDS = 0.5

# Randomisation ranges.
XY_JITTER = 0.015
UTENSIL_JITTER = 0.004
UTENSIL_YAW = 0.15
MASS_SCALE = (0.8, 1.2)
FRICTION_SCALE = (0.7, 1.3)
LIGHT_SCALE = (0.6, 1.2)


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class DinnerTableEnv:
    def __init__(self, obs_cameras=("overhead", "operator"), obs_size=(128, 128)):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.substeps = int(round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))
        self.obs_cameras = obs_cameras
        self.obs_size = obs_size
        self._renderers = {}

        m = self.model
        self.act_idx = {a: np.array([m.actuator(a + j).id for j in JOINTS]) for a in ARMS}
        self.qpos_idx = {a: np.array([m.jnt_qposadr[m.joint(a + j).id] for j in JOINTS]) for a in ARMS}
        self.eq_ids = {(a, o): m.equality(f"{a}grasp_{o}").id for a in ARMS for o in GRASPABLE}
        self.body_ids = {o: m.body(o).id for o in GRASPABLE}
        self.gripper_body = {a: m.body(a + "gripper").id for a in ARMS}
        self.tcp_site = {a: m.site(a + "gripperframe").id for a in ARMS}
        self.handle_geom = m.geom("drawer_handle").id
        self.drawer_qadr = m.jnt_qposadr[m.joint("drawer_slide").id]
        self.free_qadr = {o: m.jnt_qposadr[m.joint(o + "_free").id] for o in PROPS}
        self.prop_geom = {o: m.geom(o + "_geom").id for o in PROPS}
        self.targets = {o: m.site(o + "_target").id for o in PLACE_ITEMS}
        self.ik = {a: ArmIK(m, a) for a in ARMS}

        # Nominal values restored before each randomisation.
        self._qpos0 = m.qpos0.copy()
        self._mass0 = m.body_mass.copy()
        self._friction0 = m.geom_friction.copy()
        self._light0 = m.light_diffuse.copy()
        self._headlight0 = m.vis.headlight.diffuse.copy()
        self._rgba0 = m.geom_rgba.copy()
        self.home = self._compute_home()
        self.reset(seed=0)

    # ------------------------------------------------------------------ setup
    def _compute_home(self):
        """Arm joint targets that park each TCP above the table, gripper open."""
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        home = {}
        for arm in ARMS:
            base = data.body(arm + "base").xpos
            q, _ = self.ik[arm].solve(data.qpos, base + HOME_OFFSET)
            home[arm] = np.append(q, GRIPPER_OPEN)
        return home

    def reset(self, seed=0):
        m, d = self.model, self.data
        rng = np.random.default_rng(seed)
        m.body_mass[:] = self._mass0
        m.geom_friction[:] = self._friction0
        m.light_diffuse[:] = self._light0
        m.vis.headlight.diffuse[:] = self._headlight0
        m.geom_rgba[:] = self._rgba0
        for eq in self.eq_ids.values():
            d.eq_active[eq] = 0
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self._qpos0

        for arm in ARMS:
            d.qpos[self.qpos_idx[arm]] = self.home[arm]
            d.ctrl[self.act_idx[arm]] = self.home[arm]

        variation = {"seed": seed}
        if seed:  # seed 0 is the nominal scene
            for obj in PROPS:
                adr = self.free_qadr[obj]
                jitter = UTENSIL_JITTER if obj in UTENSILS else XY_JITTER
                d.qpos[adr:adr + 2] += rng.uniform(-jitter, jitter, 2)
                if obj in UTENSILS:
                    base_quat = self._qpos0[adr + 3:adr + 7].copy()
                    turned = np.zeros(4)
                    mujoco.mju_mulQuat(turned, _yaw_quat(rng.uniform(-UTENSIL_YAW, UTENSIL_YAW)), base_quat)
                    d.qpos[adr + 3:adr + 7] = turned
                mass_s, fric_s = rng.uniform(*MASS_SCALE), rng.uniform(*FRICTION_SCALE)
                m.body_mass[self.body_ids[obj]] *= mass_s
                m.geom_friction[self.prop_geom[obj], 0] *= fric_s
                variation[obj] = {"mass_scale": round(mass_s, 3), "friction_scale": round(fric_s, 3)}
            light_s = rng.uniform(*LIGHT_SCALE)
            m.light_diffuse[:] *= light_s
            m.vis.headlight.diffuse[:] *= light_s
            table = m.geom("table_top").id
            m.geom_rgba[table, :3] = np.clip(self._rgba0[table, :3] * rng.uniform(0.75, 1.2, 3), 0, 1)
            variation["light_scale"] = round(light_s, 3)
        self.variation = variation

        self.held = {a: None for a in ARMS}
        self.grasp_intent = {a: None for a in ARMS}
        self.pour_time = 0.0
        self.drawer_max = 0.0
        self.time_step = 0
        mujoco.mj_forward(m, d)
        for _ in range(100):  # let props settle
            mujoco.mj_step(m, d)
        return self.observe()

    # ---------------------------------------------------------------- control
    def step(self, action):
        """Apply 12 joint targets (left 6 then right 6) for one control period."""
        action = np.asarray(action, dtype=float)
        for i, arm in enumerate(ARMS):
            self.data.ctrl[self.act_idx[arm]] = action[6 * i:6 * i + 6]
        self._update_grasps()
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self._update_progress()
        self.time_step += 1
        return self.observe()

    def arm_qpos(self, arm):
        return self.data.qpos[self.qpos_idx[arm]].copy()

    def joint_state(self):
        return np.concatenate([self.arm_qpos(a) for a in ARMS])

    def tcp(self, arm):
        return self.data.site_xpos[self.tcp_site[arm]].copy()

    def grasp_point(self, obj):
        if obj == "drawer":
            return self.data.geom_xpos[self.handle_geom].copy()
        return self.data.xpos[self.body_ids[obj]].copy()

    # --------------------------------------------------------------- grasping
    def set_grasp_intent(self, arm, obj):
        """Declare which object ``arm`` means to grasp next (None clears it)."""
        if obj is not None and obj not in GRASPABLE:
            raise ValueError(f"unknown graspable object '{obj}'")
        self.grasp_intent[arm] = obj

    def _update_grasps(self):
        for arm in ARMS:
            closing = self.data.ctrl[self.act_idx[arm][5]] < GRASP_CLOSE_THRESHOLD
            if self.held[arm] and not closing:
                self.data.eq_active[self.eq_ids[(arm, self.held[arm])]] = 0
                self.held[arm] = None
            elif self.held[arm] is None and closing:
                obj = self._select_graspable(arm)
                if obj:
                    self._activate(arm, obj)

    def _select_graspable(self, arm):
        tcp = self.tcp(arm)
        intent = self.grasp_intent[arm]
        if intent and np.linalg.norm(self.grasp_point(intent) - tcp) < INTENT_RADIUS:
            return intent
        best, best_dist = None, GRASP_RADIUS
        for obj in GRASPABLE:
            dist = np.linalg.norm(self.grasp_point(obj) - tcp)
            if dist < best_dist:
                best, best_dist = obj, dist
        return best

    def _activate(self, arm, obj):
        """Attach ``obj`` to the gripper at their current relative pose."""
        d, m = self.data, self.model
        eq = self.eq_ids[(arm, obj)]
        b1, b2 = self.gripper_body[arm], self.body_ids[obj]
        rot1 = d.xmat[b1].reshape(3, 3)
        if m.eq_type[eq] == mujoco.mjtEq.mjEQ_CONNECT:
            # Pin the current TCP point on both bodies; rotation stays free.
            tcp = self.tcp(arm)
            m.eq_data[eq, 0:3] = rot1.T @ (tcp - d.xpos[b1])
            m.eq_data[eq, 3:6] = d.xmat[b2].reshape(3, 3).T @ (tcp - d.xpos[b2])
        else:
            q1_inv, relquat = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(q1_inv, d.xquat[b1])
            mujoco.mju_mulQuat(relquat, q1_inv, d.xquat[b2])
            m.eq_data[eq, 0:3] = 0.0
            m.eq_data[eq, 3:6] = rot1.T @ (d.xpos[b2] - d.xpos[b1])
            m.eq_data[eq, 6:10] = relquat
            m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        self.held[arm] = obj

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
    def _bottle_tilt(self):
        z_axis = self.data.xmat[self.body_ids["bottle"]].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))

    def _update_progress(self):
        self.drawer_max = max(self.drawer_max, float(self.data.qpos[self.drawer_qadr]))
        bottle_rot = self.data.xmat[self.body_ids["bottle"]].reshape(3, 3)
        tip = self.data.xpos[self.body_ids["bottle"]] + bottle_rot @ np.array([0.0, 0.0, 0.06])
        mug = self.data.xpos[self.body_ids["mug"]]
        over_mug = np.linalg.norm(tip[:2] - mug[:2]) < POUR_XY_TOL and tip[2] > mug[2]
        if over_mug and self._bottle_tilt() > POUR_TILT_MIN:
            self.pour_time += 1.0 / CONTROL_HZ

    def _placed(self, obj):
        pos = self.data.xpos[self.body_ids[obj]]
        target = self.data.site_xpos[self.targets[obj]]
        on_table = pos[2] < TABLE_TOP_Z + 0.05
        return bool(np.linalg.norm(pos[:2] - target[:2]) < PLACE_TOL and on_table
                    and obj not in self.held.values())

    def success(self):
        checks = {
            "drawer_open": bool(self.drawer_max > DRAWER_OPEN_MIN),
            **{f"{o}_placed": self._placed(o) for o in PLACE_ITEMS},
            "poured": bool(self.pour_time >= POUR_SECONDS),
        }
        checks["all"] = all(checks.values())
        return checks

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
