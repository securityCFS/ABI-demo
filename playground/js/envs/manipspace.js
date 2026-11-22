// OGBench manipspace (UR5e + Robotiq 2F-85, one cube) rebuilt on the MuJoCo WebAssembly bindings.
// Mirrors ogbench/manipspace (cube_env, manipspace_env, controllers/diff_ik, lie): a 5-d relative
// end-effector action (dx, dy, dz, dyaw, d gripper) is turned into arm joint targets by the same differential IK
// (damped least squares on the attachment-site Jacobian, 20 iterations at most) and run through the arm's PD
// servos for 25 physics substeps of 2 ms. Observation, reset distribution, success test (cube within 4 cm of the
// target, reported one step late as OGBench's success_timing 'pre') and the 200-step limit follow the Python code.
import { Rng } from '../sim.js';

export { Rng };

// ---- rotations (lie.SO3 / lie.SE3 arithmetic, MuJoCo's mju_* quaternion helpers), wxyz quaternions -------------
const EPS64 = 1e-10;
function so3Mul(a, b) {           // lie.SO3.multiply, same expression order
  const [w0, x0, y0, z0] = a, [w1, x1, y1, z1] = b;
  return [-x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
          x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
          -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
          x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1];
}
const so3Inv = q => [q[0], -q[1], -q[2], -q[3]];
function so3FromZ(theta) {        // lie.SO3.exp([0, 0, theta])
  const tsq = 0 * 0 + 0 * 0 + theta * theta;
  if (tsq < EPS64) {
    const t4 = tsq * tsq;
    const real = 1.0 - tsq / 8.0 + t4 / 384.0, imag = 0.5 - tsq / 48.0 + t4 / 3840.0;
    return [real, imag * 0, imag * 0, imag * theta];
  }
  const safe = Math.sqrt(tsq), half = 0.5 * safe;
  const real = Math.cos(half), imag = Math.sin(half) / safe;
  return [real, imag * 0, imag * 0, imag * theta];
}
const so3Yaw = q => Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
function so3Apply(q, v) {         // (q [0, v] q^-1)[1:]
  const r = so3Mul(so3Mul(q, [0, v[0], v[1], v[2]]), so3Inv(q));
  return [r[1], r[2], r[3]];
}
function se3Mul(A, B) {           // {q, t}
  const t = so3Apply(A.q, B.t);
  return { q: so3Mul(A.q, B.q), t: [t[0] + A.t[0], t[1] + A.t[1], t[2] + A.t[2]] };
}
function normalize4(q) {          // mju_normalize4
  const n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n < 1e-15) { q[0] = 1; q[1] = 0; q[2] = 0; q[3] = 0; } else if (Math.abs(n - 1) > 1e-15) {
    const inv = 1 / n; q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
  }
  return q;
}
function mat2Quat(m, o = 0) {     // mju_mat2Quat on a row-major 3x3 at offset o
  const a = i => m[o + i], q = [0, 0, 0, 0];
  if (a(0) + a(4) + a(8) > 0) {
    q[0] = 0.5 * Math.sqrt(1 + a(0) + a(4) + a(8));
    q[1] = 0.25 * (a(7) - a(5)) / q[0]; q[2] = 0.25 * (a(2) - a(6)) / q[0]; q[3] = 0.25 * (a(3) - a(1)) / q[0];
  } else if (a(0) > a(4) && a(0) > a(8)) {
    q[1] = 0.5 * Math.sqrt(1 + a(0) - a(4) - a(8));
    q[0] = 0.25 * (a(7) - a(5)) / q[1]; q[2] = 0.25 * (a(1) + a(3)) / q[1]; q[3] = 0.25 * (a(2) + a(6)) / q[1];
  } else if (a(4) > a(8)) {
    q[2] = 0.5 * Math.sqrt(1 - a(0) + a(4) - a(8));
    q[0] = 0.25 * (a(2) - a(6)) / q[2]; q[1] = 0.25 * (a(1) + a(3)) / q[2]; q[3] = 0.25 * (a(5) + a(7)) / q[2];
  } else {
    q[3] = 0.5 * Math.sqrt(1 - a(0) - a(4) + a(8));
    q[0] = 0.25 * (a(3) - a(1)) / q[3]; q[1] = 0.25 * (a(2) + a(6)) / q[3]; q[2] = 0.25 * (a(5) + a(7)) / q[3];
  }
  return normalize4(q);
}
function mulQuat(a, b) {          // mju_mulQuat
  return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
          a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
          a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
}
function quat2Vel(q, dt) {        // mju_quat2Vel
  const ax = [q[1], q[2], q[3]];
  const n = Math.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);   // mju_normalize3
  if (n < 1e-15) { ax[0] = 1; ax[1] = 0; ax[2] = 0; } else if (Math.abs(n - 1) > 1e-15) {
    const inv = 1 / n; ax[0] *= inv; ax[1] *= inv; ax[2] *= inv;
  }
  let speed = 2 * Math.atan2(n, q[0]);
  if (speed > Math.PI) speed -= 2 * Math.PI;
  speed /= dt;
  return [ax[0] * speed, ax[1] * speed, ax[2] * speed];
}
// Solve the dense n x n system A x = b (LU with partial pivoting, as LAPACK gesv); A is overwritten.
function solve(A, b, n) {
  const x = b.slice(), piv = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    let p = k, mx = Math.abs(A[k * n + k]);
    for (let i = k + 1; i < n; i++) { const v = Math.abs(A[i * n + k]); if (v > mx) { mx = v; p = i; } }
    piv[k] = p;
    if (p !== k) {
      for (let j = 0; j < n; j++) { const t = A[k * n + j]; A[k * n + j] = A[p * n + j]; A[p * n + j] = t; }
      const t = x[k]; x[k] = x[p]; x[p] = t;
    }
    const d = A[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = A[i * n + k] / d; A[i * n + k] = f;
      for (let j = k + 1; j < n; j++) A[i * n + j] -= f * A[k * n + j];
      x[i] -= f * x[k];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= A[i * n + j] * x[j];
    x[i] = s / A[i * n + i];
  }
  return x;
}

export const _math = { so3Mul, so3Inv, so3FromZ, so3Yaw, so3Apply, se3Mul, mat2Quat, mulQuat, quat2Vel, solve };

export class Env {
  // xml: scene.xml text; assets: {file name: Uint8Array} of every file in envMeta.assets (fetched from
  // envMeta.asset_dir). The files and the XML are written once into the WebAssembly module's in-memory file system
  // (mj.FS) and compiled with MjModel.from_xml_path; MjVFS.addBuffer works too but copies 9 MB byte by byte (6-12 s).
  // share: another Env of the same scene; its compiled MjModel is reused (e.g. for the ghost run), assets may be null.
  constructor(mj, xml, envMeta, assets, share) {
    this.mj = mj; this.meta = envMeta;
    if (share) this.model = share.model;
    else {
      if (!assets) throw new Error('manipspace: pass the mesh assets {name: Uint8Array} as the 4th constructor argument');
      const dir = `/manipspace_${Env._dirs = (Env._dirs || 0) + 1}`;
      mj.FS.mkdir(dir);
      for (const name of envMeta.assets) {
        if (!assets[name]) throw new Error(`manipspace: missing asset ${name}`);
        mj.FS.writeFile(`${dir}/${name}`, assets[name]);
      }
      mj.FS.writeFile(`${dir}/scene.xml`, xml);
      const vfs = new mj.MjVFS();
      this.model = mj.MjModel.from_xml_path(`${dir}/scene.xml`, vfs);
      vfs.delete();
      for (const name of envMeta.assets) mj.FS.unlink(`${dir}/${name}`);   // compiled into the model; free the copies
    }
    this.data = new mj.MjData(this.model);
    this.ik = new mj.MjData(this.model);           // scratch data for the IK (kinematics of the arm only)
    const m = this.model, I = envMeta.indices, C = envMeta.controller;
    this.nq = m.nq; this.nv = m.nv; this.nu = m.nu; this.ngeom = m.ngeom;
    if (m.nq !== I.nq || m.nv !== I.nv || m.nu !== I.nu) throw new Error('manipspace: model sizes differ from env_meta');
    this.I = I; this.armQ = I.arm_qpos; this.armV = I.arm_dof;
    this.lo = envMeta.action.low; this.hi = envMeta.action.high;
    this.down = C.effector_down_wxyz; this.downInv = so3Inv(this.down);
    this.Tpa = { q: C.T_pa_wxyz_xyz.slice(0, 4), t: C.T_pa_wxyz_xyz.slice(4, 7) };
    this.ws = C.workspace_bounds; this.ikp = C.ik;
    this.jacp = new mj.DoubleBuffer(3 * m.nv); this.jacr = new mj.DoubleBuffer(3 * m.nv);
    this.substeps = envMeta.substeps; this.limit = envMeta.max_episode_steps; this.thr = envMeta.success.threshold;
    this.O = envMeta.observation;
    // rendered geoms: MuJoCo's default visible groups 0-2 (visual meshes, cube, target); collision geoms are group 3
    this.dynamicGeoms = [];
    for (let g = 0; g < this.ngeom; g++) if (m.geom_bodyid[g] !== 0 && m.geom_group[g] <= 2) this.dynamicGeoms.push(g);
    this.task = 0; this.init = envMeta.tasks[0].init_xyz.slice(); this.goal = envMeta.tasks[0].goal_xyz.slice();
    this.t = 0; this.ret = 0; this.lastReward = 0; this.success = false; this.done = false; this._succ = false;
    this.mj.mj_resetData(m, this.data);
    this._setTarget();
    this.mj.mj_forward(m, this.data);
  }

  // k: task index; or {goal: [x, y(, z)], init: [x, y(, z)]} for a clicked target (z defaults to the table, 0.02).
  setTask(k) {
    if (typeof k === 'number') {
      const t = this.meta.tasks[k]; this.task = k; this.init = t.init_xyz.slice(); this.goal = t.goal_xyz.slice();
    } else {
      this.task = null;
      if (k.goal) this.goal = [k.goal[0], k.goal[1], k.goal.length > 2 ? k.goal[2] : this.meta.click_targets.z];
      if (k.init) this.init = [k.init[0], k.init[1], k.init.length > 2 ? k.init[2] : this.meta.click_targets.z];
    }
    this._setTarget();
    this._succ = false;                              // a new target starts like a reset (success flag cleared)
  }
  setGoal(xy) { this.setTask({ goal: xy }); }
  _setTarget() {
    const d = this.data, k = this.I.target_mocap;
    d.mocap_pos[3 * k] = this.goal[0]; d.mocap_pos[3 * k + 1] = this.goal[1]; d.mocap_pos[3 * k + 2] = this.goal[2];
    d.mocap_quat[4 * k] = 1; d.mocap_quat[4 * k + 1] = 0; d.mocap_quat[4 * k + 2] = 0; d.mocap_quat[4 * k + 3] = 0;
  }

  // Differential IK of controllers/diff_ik on the attachment site: returns the arm joint targets.
  ikSolve(pos, quat, currQ) {
    const m = this.model, d = this.ik, p = this.ikp, sid = this.I.attach_site, nv = m.nv, A = this.armV;
    for (let i = 0; i < 6; i++) d.qpos[this.armQ[i]] = currQ[i];
    for (let it = 0; it < p.max_iters; it++) {
      this.mj.mj_kinematics(m, d); this.mj.mj_comPos(m, d);
      const sp = d.site_xpos, err = new Array(6);
      for (let i = 0; i < 3; i++) err[i] = pos[i] - sp[3 * sid + i];
      const sq = mat2Quat(d.site_xmat, 9 * sid);
      const ev = quat2Vel(mulQuat(quat, [sq[0], -sq[1], -sq[2], -sq[3]]), 1.0);
      err[3] = ev[0]; err[4] = ev[1]; err[5] = ev[2];
      const np = Math.sqrt(err[0] * err[0] + err[1] * err[1] + err[2] * err[2]);
      const no = Math.sqrt(err[3] * err[3] + err[4] * err[4] + err[5] * err[5]);
      if (np <= p.pos_thresh && no <= p.ori_thresh) break;
      this.mj.mj_jacSite(m, d, this.jacp, this.jacr, sid);
      const jp = this.jacp.GetView(), jr = this.jacr.GetView(), J = new Float64Array(36);
      for (let c = 0; c < 6; c++) for (let r = 0; r < 3; r++) { J[r * 6 + c] = jp[r * nv + A[c]]; J[(r + 3) * 6 + c] = jr[r * nv + A[c]]; }
      const H = new Float64Array(36);
      for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
        let s = 0; for (let c = 0; c < 6; c++) s += J[i * 6 + c] * J[j * 6 + c];
        H[i * 6 + j] = s + (i === j ? p.damping : 0);
      }
      const y = solve(H, err, 6), upd = new Array(6);
      let mx = 0;
      for (let c = 0; c < 6; c++) { let s = 0; for (let r = 0; r < 6; r++) s += J[r * 6 + c] * y[r]; upd[c] = s; mx = Math.max(mx, Math.abs(s)); }
      if (mx > p.max_angle_change) for (let c = 0; c < 6; c++) upd[c] *= p.max_angle_change / mx;
      for (let c = 0; c < 6; c++) d.qpos[this.armQ[c]] += upd[c] * 1.0;      // mj_integratePos on hinge joints
    }
    return this.armQ.map(i => d.qpos[i]);
  }

  // manipspace_env.set_control: action in [-1, 1]^5 -> ctrl (6 arm joint targets, gripper 0..255).
  // Reads the pinch-site pose of the last physics forward, as the Python code does.
  computeCtrl(action, armQ, gripQ, pinchPos, pinchMat) {
    const a = new Array(5);
    // unnormalize_action on a float32 action: numpy evaluates 0.5 * (a + 1) in float32, the rest in float64
    for (let i = 0; i < 5; i++) a[i] = 0.5 * Math.fround(Math.fround(action[i]) + 1) * (this.hi[i] - this.lo[i]) + this.lo[i];
    const yaw0 = so3Yaw(mat2Quat(pinchMat, 0));
    const open = Math.min(1, Math.max(0, gripQ / this.meta.controller.gripper_opening_scale));
    const tp = [pinchPos[0] + a[0], pinchPos[1] + a[1], pinchPos[2] + a[2]];
    let rot = so3Mul(so3Mul(so3FromZ(a[3]), so3FromZ(yaw0)), this.downInv);
    let grip = open + a[4];
    for (let i = 0; i < 3; i++) tp[i] = Math.min(this.ws[1][i], Math.max(this.ws[0][i], tp[i]));
    const yaw = Math.min(Math.PI, Math.max(-Math.PI, so3Yaw(rot)));
    rot = so3Mul(so3FromZ(yaw), this.down);
    grip = Math.min(1, Math.max(0, grip));
    const Twa = se3Mul({ q: rot, t: tp }, this.Tpa);
    const q = this.ikSolve(Twa.t, Twa.q, armQ);
    return q.concat([this.meta.controller.gripper_ctrl_scale * grip]);
  }

  _pinch() {
    const d = this.data, s = this.I.pinch_site;
    return [Array.from(d.site_xpos.subarray(3 * s, 3 * s + 3)), Array.from(d.site_xmat.subarray(9 * s, 9 * s + 9))];
  }

  setControl(action) {
    const d = this.data, [pp, pm] = this._pinch();
    const ctrl = this.computeCtrl(action, this.armQ.map(i => d.qpos[i]), d.qpos[this.I.gripper_qpos], pp, pm);
    const A = this.I.arm_actuators;
    for (let i = 0; i < 6; i++) d.ctrl[A[i]] = ctrl[i];
    d.ctrl[this.I.gripper_actuator] = ctrl[6];
    return ctrl;
  }

  cubeDist() {
    const q = this.data.qpos, c = this.I.cube_qpos, g = this.goal;
    const dx = q[c] - g[0], dy = q[c + 1] - g[1], dz = q[c + 2] - g[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  atGoal() { return this.cubeDist() <= this.thr; }       // cube_env._compute_successes on the current state
  distToGoal() { return this.cubeDist(); }

  // Reward labels of the critic's reference states for a cube target (the paper's relabelling of the offline data):
  // w_k = 1 if ||ref_xyz_k - goal|| < 0.04 (strict, 3-D, goal z = table height 0.02), else 0. refXYZ: Float32Array [3R].
  // Reproduces the paper's labels of all 5 tasks exactly (w_task{k}.bin).
  static goalLabels(refXYZ, goal, threshold = 0.04) {
    const R = refXYZ.length / 3, w = new Float32Array(R), g = [goal[0], goal[1], goal.length > 2 ? goal[2] : 0.02];
    for (let k = 0; k < R; k++) {
      const dx = refXYZ[3 * k] - g[0], dy = refXYZ[3 * k + 1] - g[1], dz = refXYZ[3 * k + 2] - g[2];
      w[k] = Math.sqrt(dx * dx + dy * dy + dz * dz) < threshold ? 1 : 0;
    }
    return w;
  }

  // A new episode: arm from IK on a random pinch pose, cube at the task's start with +-1 cm and a random yaw.
  // opts.draws = {eff_pos: [x, y, z], yaw, jitter: [dx, dy], cube_yaw} replaces the random draws (parity tests).
  reset(rng, opts = {}) {
    const m = this.model, d = this.data, R = this.meta.reset, mj = this.mj;
    mj.mj_resetData(m, d);
    for (let i = 0; i < 6; i++) d.qpos[this.armQ[i]] = R.home_qpos[i];
    const B = R.arm_sampling_bounds;
    const dr = opts.draws || {
      eff_pos: [0, 1, 2].map(i => rng.uniform(B[0][i], B[1][i])), yaw: rng.uniform(-Math.PI, Math.PI),
      jitter: [rng.uniform(-R.cube_xy_jitter, R.cube_xy_jitter), rng.uniform(-R.cube_xy_jitter, R.cube_xy_jitter)],
      cube_yaw: rng.uniform(0, 2 * Math.PI),
    };
    if (opts.jitter === false) dr.jitter = [0, 0];
    const ori = so3Mul(so3FromZ(dr.yaw), this.down);
    const Twa = se3Mul({ q: ori, t: dr.eff_pos }, this.Tpa);
    const q = this.ikSolve(Twa.t, Twa.q, R.home_qpos);
    for (let i = 0; i < 6; i++) d.qpos[this.armQ[i]] = q[i];
    mj.mj_forward(m, d);
    const c = this.I.cube_qpos, cq = so3FromZ(dr.cube_yaw);
    d.qpos[c] = this.init[0] + dr.jitter[0]; d.qpos[c + 1] = this.init[1] + dr.jitter[1]; d.qpos[c + 2] = this.init[2];
    for (let i = 0; i < 4; i++) d.qpos[c + 3 + i] = cq[i];
    this._setTarget();
    mj.mj_forward(m, d);
    for (let i = 0; i < this.nu; i++) d.ctrl[i] = 0;
    for (let i = 0; i < 6; i++) d.ctrl[this.I.arm_actuators[i]] = q[i];   // hold pose until the first action
    this.t = 0; this.ret = 0; this.lastReward = 0; this.success = false; this.done = false; this._succ = false;
    this.draws = dr;
  }

  copyStateFrom(other) {
    const d = this.data, o = other.data;
    for (const f of ['qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'mocap_pos', 'mocap_quat']) d[f].set(o[f]);
    d.time = o.time;
    this.goal = other.goal.slice(); this.init = other.init.slice(); this.task = other.task;
    this.mj.mj_forward(this.model, d);
    // keep the other run's last-substep site poses and contact forces (what the controller and obs read)
    for (const f of ['site_xpos', 'site_xmat', 'cfrc_ext']) d[f].set(o[f]);
    this._succ = other._succ;
    this.t = 0; this.ret = 0; this.lastReward = 0; this.success = false; this.done = false;
  }

  // Observation from explicit inputs (parity tests) or from the current data.
  static obsFrom(O, armQ, armV, pinchPos, pinchMat, gripQ, cfrc6, cubeQ7) {
    const o = new Float32Array(28), c = O.xyz_center, s = O.xyz_scaler;
    for (let i = 0; i < 6; i++) { o[i] = armQ[i]; o[6 + i] = armV[i]; }
    for (let i = 0; i < 3; i++) o[12 + i] = (pinchPos[i] - c[i]) * s;
    const y = so3Yaw(mat2Quat(pinchMat, 0));
    o[15] = Math.cos(y); o[16] = Math.sin(y);
    o[17] = Math.min(1, Math.max(0, gripQ / 0.8)) * O.gripper_scaler;
    let f = 0; for (let i = 0; i < 6; i++) f += cfrc6[i] * cfrc6[i];
    o[18] = Math.min(1, Math.max(0, Math.sqrt(f) / O.contact_scale));
    for (let i = 0; i < 3; i++) o[19 + i] = (cubeQ7[i] - c[i]) * s;
    for (let i = 0; i < 4; i++) o[22 + i] = cubeQ7[3 + i];
    const cy = so3Yaw([cubeQ7[3], cubeQ7[4], cubeQ7[5], cubeQ7[6]]);
    o[26] = Math.cos(cy); o[27] = Math.sin(cy);
    return o;
  }
  obs() {
    const d = this.data, [pp, pm] = this._pinch(), b = this.I.right_pad_body, c = this.I.cube_qpos;
    return Env.obsFrom(this.O, this.armQ.map(i => d.qpos[i]), this.armV.map(i => d.qvel[i]), pp, pm,
                       d.qpos[this.I.gripper_qpos], d.cfrc_ext.subarray(6 * b, 6 * b + 6), d.qpos.subarray(c, c + 7));
  }

  // One control step: clip, IK + servo targets, 25 substeps (onSubstep(k) after each), contact forces.
  step(action, onSubstep) {
    const m = this.model, d = this.data;
    const a = Array.from(action, x => Math.max(-1, Math.min(1, x)));
    const pre = this._succ;                                   // success_timing 'pre'
    const reward = this.atGoal() ? 0 : -1;                    // singletask reward on the state before the step
    this.setControl(a);
    for (let k = 0; k < this.substeps; k++) { this.mj.mj_step(m, d); if (onSubstep) onSubstep(k); }
    this.mj.mj_rnePostConstraint(m, d);
    this._succ = this.atGoal();                               // post_step
    this.t += 1; this.lastReward = reward; this.ret += reward;
    this.success = pre; this.done = pre || this.t >= this.limit;
    return this.success;
  }

  hud() {
    const d = this.data, g = Math.min(1, Math.max(0, d.qpos[this.I.gripper_qpos] / 0.8));
    return [['time', `${(this.t * this.meta.control_timestep).toFixed(2)} s`], ['step', `${this.t} / ${this.limit}`],
            ['cube to target', `${(100 * this.cubeDist()).toFixed(1)} cm`], ['gripper closed', `${Math.round(100 * g)}%`],
            ['return', this.ret.toFixed(0)]];
  }

  // Poses of the rendered moving geoms, flattened [x y z, 9 rotation entries] per geom.
  poses(out) {
    const xp = this.data.geom_xpos, xm = this.data.geom_xmat;
    let o = 0;
    for (const g of this.dynamicGeoms) {
      out[o++] = xp[3 * g]; out[o++] = xp[3 * g + 1]; out[o++] = xp[3 * g + 2];
      for (let k = 0; k < 9; k++) out[o++] = xm[9 * g + k];
    }
    return out;
  }

  // Static description for the renderer: every geom (with group/visibility), and the vertices (float32, geom frame)
  // and triangles of every mesh used by a rendered geom (geom.mesh = index into meshes, -1 otherwise).
  describe() {
    const m = this.model, d = this.data, geoms = [], meshes = [], meshIndex = new Map();
    for (let g = 0; g < this.ngeom; g++) {
      const mat = m.geom_matid[g];
      const rgba = mat >= 0 ? Array.from(m.mat_rgba.slice(4 * mat, 4 * mat + 4)) : Array.from(m.geom_rgba.slice(4 * g, 4 * g + 4));
      const visible = m.geom_group[g] <= 2;
      let mesh = -1;
      if (m.geom_type[g] === 7 && visible) {
        const id = m.geom_dataid[g];
        if (!meshIndex.has(id)) {
          const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
          meshes.push({ id, name: m.mesh(id).name, vert: Float32Array.from(m.mesh_vert.subarray(3 * va, 3 * (va + vn))),
                        face: Int32Array.from(m.mesh_face.subarray(3 * fa, 3 * (fa + fn))) });
          meshIndex.set(id, meshes.length - 1);
        }
        mesh = meshIndex.get(id);
      }
      geoms.push({
        id: g, name: m.geom(g).name, type: m.geom_type[g], body: m.geom_bodyid[g], group: m.geom_group[g], visible, mesh,
        size: Array.from(m.geom_size.slice(3 * g, 3 * g + 3)), rgba,
        pos: Array.from(d.geom_xpos.slice(3 * g, 3 * g + 3)), mat: Array.from(d.geom_xmat.slice(9 * g, 9 * g + 9)),
      });
    }
    return { geoms, dynamic: this.dynamicGeoms, meshes, cubeGeom: this.I.cube_geom, targetGeom: this.I.target_geom };
  }
}
