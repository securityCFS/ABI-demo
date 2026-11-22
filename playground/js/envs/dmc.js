// DeepMind Control Suite domains (quadruped, cheetah, walker) rebuilt on the MuJoCo WebAssembly bindings, as the paper
// runs them (dm_control suite tasks plus the URLB-style variants, flat float32 observations, actions in [-1, 1]).
// Everything domain-specific comes from env_meta.json: observation layout, reward features, reward families and
// knobs, reset rule. The reward of a task is a dm_control tolerance rule on a few features of the state; the same rule
// gives the critic's labels on its reference states (labels) and the reward of every step (Env.reward), so a task can be
// any value of a knob (e.g. a target speed), with the paper's tasks as presets.
import { Rng } from '../sim.js';

// ---------- dm_control.utils.rewards.tolerance (scalar; 'gaussian' and 'linear' sigmoids), float64 ----------
export function tolerance(x, lo, hi, margin = 0, sigmoid = 'gaussian', valueAtMargin = 0.1) {
  const inBounds = lo <= x && x <= hi;
  if (margin === 0) return inBounds ? 1 : 0;
  if (inBounds) return 1;
  const d = (x < lo ? lo - x : x - hi) / margin;
  if (sigmoid === 'linear') {
    const s = d * (1 - valueAtMargin);
    return Math.abs(s) < 1 ? 1 - s : 0;
  }
  if (sigmoid === 'gaussian') {
    const scale = Math.sqrt(-2 * Math.log(valueAtMargin));
    return Math.exp(-0.5 * (d * scale) ** 2);
  }
  throw new Error(`unsupported sigmoid ${sigmoid}`);
}

// ---------- reward families: f = features of one state (env_meta.reward_features order), p = knob values ----------
const STAND_HEIGHT = 1.2;
const RULES = {
  quadruped(f, family, p) {
    // _upright_reward(deviation_angle = 0): tolerance(torso_upright, [cos 0, inf), margin 1 + cos 0, linear, 0)
    const up = tolerance(f[0], 1, Infinity, 2, 'linear', 0);
    if (family === 'stand') return up;
    if (family === 'move') return up * tolerance(f[1], p.speed, Infinity, p.speed, 'linear', 0.5);
    if (family === 'jump') return up * tolerance(f[2], p.height, Infinity, p.height, 'linear', 0.5);
    throw new Error(`quadruped: unknown family ${family}`);
  },
  cheetah(f, family, p) {
    // speed v: run / walk (v > 0) and run_backward / walk_backward (v < 0); tolerance(fw * vx, [|v|, inf), |v|, linear, 0)
    if (family !== 'speed') throw new Error(`cheetah: unknown family ${family}`);
    const fw = p.speed < 0 ? -1 : 1, s = Math.abs(p.speed);
    return tolerance(fw * f[0], s, Infinity, s, 'linear', 0);
  },
  walker(f, family, p) {
    if (family === 'spin') return tolerance(f[3], p.angmom, Infinity, p.angmom, 'linear', 0);
    if (family !== 'speed') throw new Error(`walker: unknown family ${family}`);
    const standing = tolerance(f[0], STAND_HEIGHT, Infinity, STAND_HEIGHT / 2);
    const upright = (1 + f[1]) / 2;
    const stand = (3 * standing + upright) / 4;
    if (p.speed === 0) return stand;
    const move = tolerance(f[2], p.speed, Infinity, p.speed / 2, 'linear', 0.5);
    return stand * (5 * move + 1) / 6;
  },
};

// A task: a preset index into envMeta.tasks, a preset id, a preset object, or {family, params} (a knob value).
export function resolveTask(envMeta, k) {
  if (typeof k === 'number') return envMeta.tasks[k];
  if (typeof k === 'string') return envMeta.tasks.find(t => t.id === k);
  return k;
}

// The paper preset closest to a task (same family, nearest knob value): its QGF tau and action map serve knob values
// between presets (a demo choice; the paper only has the presets).
export function nearestPreset(envMeta, k) {
  const task = resolveTask(envMeta, k);
  const same = envMeta.tasks.filter(t => t.family === task.family);
  if (!same.length) throw new Error(`no preset in family ${task.family}`);
  const dist = t => Object.keys(task.params || {}).reduce((s, q) => s + Math.abs((t.params[q] ?? 0) - task.params[q]), 0);
  return same.reduce((b, t) => (dist(t) < dist(b) ? t : b), same[0]);
}

// Reward of one state with features f (float64 array-like) under a task.
export function taskReward(envMeta, task, f) {
  return RULES[envMeta.domain](f, task.family, task.params || {});
}

// The critic's labels for a task: the task's reward at each reference state (features: Float64Array R * F, row-major,
// in the critic's reference order). Rounded once to float32, as the paper's label cache.
export function labels(envMeta, features, k) {
  const task = resolveTask(envMeta, k), F = envMeta.reward_features.length, R = features.length / F;
  const w = new Float32Array(R), f = new Float64Array(F);
  for (let r = 0; r < R; r++) {
    for (let j = 0; j < F; j++) f[j] = features[r * F + j];
    w[r] = taskReward(envMeta, task, f);
  }
  return w;
}

export class Env {
  constructor(mj, xml, meta) {
    this.mj = mj; this.meta = meta;
    this.model = mj.MjModel.from_xml_string(xml);
    this.data = new mj.MjData(this.model);
    const m = this.model;
    this.nq = m.nq; this.nv = m.nv; this.nu = m.nu; this.na = m.na; this.ngeom = m.ngeom;
    if (m.nq !== meta.nq || m.nv !== meta.nv || m.nu !== meta.nu || m.na !== meta.na) throw new Error('scene does not match env_meta');
    this.sd = meta.obs.dim;
    this.dsblActuation = (mj.mjtDisableBit && mj.mjtDisableBit.mjDSBL_ACTUATION) ? mj.mjtDisableBit.mjDSBL_ACTUATION.value : meta.dsbl_actuation;
    // action -> ctrl as the paper's wrapper stack computes it (dm_control action_scale.Wrapper under NumPy 1.x promotion):
    // float64 map: lo + (hi - lo) / 2 * f32(a + 1); float32 map (dm_control suite tasks, whose action spec the paper's
    // wrapper casts to float32): f32(lo32 + f32(f32(f32(hi32 - lo32) / 2) * f32(a + 1))). They differ by ~1e-8 in ctrl.
    const cr = meta.action.ctrlrange, f = Math.fround;
    this.ctrlLo = Float64Array.from(cr.map(r => r[0])); this.ctrlHalf = Float64Array.from(cr.map(r => (r[1] - r[0]) / 2));
    this.ctrlLo32 = Float64Array.from(cr.map(r => f(r[0]))); this.ctrlHalf32 = Float64Array.from(cr.map(r => f(f(f(r[1]) - f(r[0])) / 2)));
    this.feat = new Float64Array(meta.reward_features.length);
    this.dynamicGeoms = [];
    for (let g = 0; g < this.ngeom; g++) if (m.geom_bodyid[g] !== 0) this.dynamicGeoms.push(g);
    this.setTask(0);
    this.t = 0; this.ret = 0; this.lastReward = 0; this.success = null; this.done = false;
  }

  setTask(k) {
    this.task = resolveTask(this.meta, k);
    this.actionMap = this.task.action_map || nearestPreset(this.meta, this.task).action_map;
    return this.task;
  }

  // mj_forward with actuation disabled (dm_control Physics.reset / after_reset).
  _forwardNoActuation() {
    const o = this.model.opt, f = o.disableflags;
    o.disableflags = f | this.dsblActuation;
    this.mj.mj_forward(this.model, this.data);
    o.disableflags = f;
  }

  _resetCounters() { this.t = 0; this.ret = 0; this.lastReward = 0; this.success = null; this.done = false; }

  // A new episode from the paper's initial-state distribution (all randomness from rng).
  // opts.draw replays a given draw (tests): quadruped {quat}, walker/cheetah {qpos: values of the randomized joints}.
  reset(rng, opts = {}) {
    const m = this.model, d = this.data, mj = this.mj, kind = this.meta.reset.kind, draw = opts.draw || null;
    mj.mj_resetData(m, d); this._forwardNoActuation();
    if (kind === 'quadruped') {
      let q = draw ? draw.quat.slice() : [rng.normal(), rng.normal(), rng.normal(), rng.normal()];
      const n = Math.hypot(q[0], q[1], q[2], q[3]); q = q.map(v => v / n);
      let z = 0, ncon = 1, tries = 0;
      while (ncon > 0) {                       // _find_non_contacting_height
        mj.mj_resetData(m, d); this._forwardNoActuation();
        d.qpos[0] = 0; d.qpos[1] = 0; d.qpos[2] = z;
        for (let i = 0; i < 4; i++) d.qpos[3 + i] = q[i];
        this._forwardNoActuation();
        ncon = d.ncon; z += this.meta.reset.z_step; tries += 1;
        if (tries > this.meta.reset.max_attempts) throw new Error('no non-contacting configuration');
      }
    } else if (kind === 'joints') {            // randomize_limited_and_rotational_joints / limited joints only (cheetah)
      const J = this.meta.reset.joints;
      J.forEach((j, i) => { d.qpos[j.qposadr] = draw ? draw.qpos[i] : rng.uniform(j.lo, j.hi); });
      if (this.meta.reset.settle_steps) {      // cheetah: physics.step() x 200 from the un-forwarded randomized pose
        mj.mj_step2(m, d);
        for (let k = 1; k < this.meta.reset.settle_steps; k++) mj.mj_step(m, d);
        mj.mj_step1(m, d);
        d.time = 0;
      }
    } else throw new Error(`unknown reset kind ${kind}`);
    this._forwardNoActuation();                // after_reset of the episode's reset_context
    this._resetCounters();
  }

  // Set a recorded physics state (tests and replays): qpos, qvel, act (and optionally qacc_warmstart, ctrl).
  setState(s, { forward = 'noact' } = {}) {
    const d = this.data;
    d.qpos.set(s.qpos); d.qvel.set(s.qvel); if (this.na) d.act.set(s.act);
    if (s.warm) d.qacc_warmstart.set(s.warm);
    if (s.ctrl) d.ctrl.set(s.ctrl);
    if (forward === 'noact') this._forwardNoActuation();
    else if (forward === 'full') this.mj.mj_forward(this.model, d);
  }

  copyStateFrom(other) {
    const d = this.data, o = other.data;
    d.qpos.set(o.qpos); d.qvel.set(o.qvel); if (this.na) d.act.set(o.act);
    d.ctrl.set(o.ctrl); d.qacc_warmstart.set(o.qacc_warmstart); d.time = o.time;
    this.mj.mj_forward(this.model, d);
    d.sensordata.set(o.sensordata);            // acceleration-stage sensors as read after the other env's last step
    this._resetCounters();
  }

  _read(src, i) {
    return this.data[src][i];
  }

  obs() {
    const o = new Float32Array(this.sd), d = this.data;
    let k = 0;
    for (const s of this.meta.obs.segments) {
      const a = d[s.src];
      if (s.fn === 'asinh') for (const i of s.idx) o[k++] = Math.asinh(a[i]);
      else for (const i of s.idx) o[k++] = a[i];
    }
    return o;
  }

  features() {
    const f = this.feat, d = this.data;
    this.meta.reward_features.forEach((r, j) => { f[j] = d[r.src][r.idx]; });
    return f;
  }

  reward() { return taskReward(this.meta, this.task, this.features()); }

  // One control step (dm_control legacy step). onSubstep(k) after each physics substep.
  step(action, onSubstep) {
    const m = this.model, d = this.data, mj = this.mj, n = this.meta.n_sub_steps;
    const f = Math.fround;
    if (this.actionMap === 'float32') {
      for (let i = 0; i < this.nu; i++) d.ctrl[i] = f(this.ctrlLo32[i] + f(this.ctrlHalf32[i] * f(f(Math.max(-1, Math.min(1, action[i]))) + 1)));
    } else {
      for (let i = 0; i < this.nu; i++) d.ctrl[i] = this.ctrlLo[i] + this.ctrlHalf[i] * f(f(Math.max(-1, Math.min(1, action[i]))) + 1);
    }
    mj.mj_step2(m, d);
    if (onSubstep && n > 1) onSubstep(0);
    for (let k = 1; k < n; k++) {
      mj.mj_step(m, d);
      if (onSubstep && k < n - 1) onSubstep(k);
    }
    mj.mj_step1(m, d);
    if (onSubstep) onSubstep(n - 1);
    this.t += 1;
    this.lastReward = this.reward();
    this.ret += this.lastReward;
    this.success = null;
    this.done = this.t >= this.meta.episode_length;
    return this.lastReward;
  }

  hud() {
    const f = this.features(), dom = this.meta.domain, out = [['return', this.ret.toFixed(1)], ['step', `${this.t} / ${this.meta.episode_length}`]];
    if (dom === 'quadruped') out.push(['speed', `${f[1].toFixed(2)} m/s`], ['height', `${f[2].toFixed(2)} m`], ['upright', f[0].toFixed(2)]);
    else if (dom === 'cheetah') out.push(['speed', `${f[0].toFixed(2)} m/s`]);
    else if (dom === 'walker') out.push(['speed', `${f[2].toFixed(2)} m/s`], ['height', `${f[0].toFixed(2)} m`], ['spin', f[3].toFixed(1)]);
    out.push(['reward', this.lastReward.toFixed(3)]);
    return out;
  }

  // Poses of the moving geoms, flattened [x y z, 9 rotation entries] per geom.
  poses(out) {
    const xp = this.data.geom_xpos, xm = this.data.geom_xmat;
    let o = 0;
    for (const g of this.dynamicGeoms) {
      out[o++] = xp[3 * g]; out[o++] = xp[3 * g + 1]; out[o++] = xp[3 * g + 2];
      for (let k = 0; k < 9; k++) out[o++] = xm[9 * g + k];
    }
    return out;
  }

  // Static description of every geom for the renderer (meshes: vertices and faces as well).
  describe() {
    const m = this.model, d = this.data, geoms = [];
    for (let g = 0; g < this.ngeom; g++) {
      const mat = m.geom_matid[g];
      const rgba = mat >= 0 ? Array.from(m.mat_rgba.slice(4 * mat, 4 * mat + 4)) : Array.from(m.geom_rgba.slice(4 * g, 4 * g + 4));
      const e = {
        id: g, name: m.geom(g).name, type: m.geom_type[g], body: m.geom_bodyid[g],
        size: Array.from(m.geom_size.slice(3 * g, 3 * g + 3)), rgba,
        pos: Array.from(d.geom_xpos.slice(3 * g, 3 * g + 3)), mat: Array.from(d.geom_xmat.slice(9 * g, 9 * g + 9)),
      };
      const mesh = m.geom_dataid[g];
      if (m.geom_type[g] === 7 && mesh >= 0) {
        const va = m.mesh_vertadr[mesh], vn = m.mesh_vertnum[mesh], fa = m.mesh_faceadr[mesh], fn = m.mesh_facenum[mesh];
        e.mesh = { vert: Array.from(m.mesh_vert.slice(3 * va, 3 * (va + vn))), face: Array.from(m.mesh_face.slice(3 * fa, 3 * (fa + fn))) };
      }
      geoms.push(e);
    }
    return { geoms, dynamic: this.dynamicGeoms, follow: this.meta.camera.follow };
  }
}

export { Rng };
