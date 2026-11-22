// OGBench locomaze (ant) environment rebuilt on the MuJoCo WebAssembly bindings.
// Mirrors ogbench/locomaze: frame_skip 5 (dt 0.1 s), observation = [qpos, qvel],
// reset noise U(-0.1, 0.1) on qpos and 0.1 N(0, 1) on qvel, start xy drawn within +-1 of
// a cell centre, success when the torso xy is within 0.5 of the goal (episode ends there).

export class Rng {
  constructor(seed) { this.s = (seed >>> 0) || 1; this.spare = null; }
  next() {               // mulberry32, uniform in [0, 1)
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(lo, hi) { return lo + (hi - lo) * this.next(); }
  normal() {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return v; }
    let u = 0, v = 0;
    while (u <= 1e-12) u = this.next();
    v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
  fillNormal(arr) { for (let i = 0; i < arr.length; i++) arr[i] = this.normal(); return arr; }
}

export class LocoMaze {
  constructor(mj, xml, meta) {
    this.mj = mj; this.meta = meta;
    this.model = mj.MjModel.from_xml_string(xml);
    this.data = new mj.MjData(this.model);
    this.nq = this.model.nq; this.nv = this.model.nv; this.nu = this.model.nu;
    this.ngeom = this.model.ngeom;
    this.targetGeom = this.model.geom('target').id;
    this.goal = [0, 0];
    this.t = 0; this.success = false; this.done = false;
    this.dynamicGeoms = [];
    for (let g = 0; g < this.ngeom; g++) if (this.model.geom_bodyid[g] !== 0) this.dynamicGeoms.push(g);
  }

  // Cell (i, j) of maze_map to world xy, as ogbench ij_to_xy.
  ijToXy(i, j) { return [j * this.meta.maze_unit - this.meta.offset_x, i * this.meta.maze_unit - this.meta.offset_y]; }
  xyToIj(x, y) {
    const u = this.meta.maze_unit;
    return [Math.floor((y + this.meta.offset_y + 0.5 * u) / u), Math.floor((x + this.meta.offset_x + 0.5 * u) / u)];
  }
  isFree(x, y) {
    const [i, j] = this.xyToIj(x, y), mm = this.meta.maze_map;
    return i >= 0 && i < mm.length && j >= 0 && j < mm[0].length && mm[i][j] === 0;
  }

  setGoal(xy) {
    this.goal = [xy[0], xy[1]];
    const p = this.model.geom_pos;           // visual marker only (contype 0)
    p[3 * this.targetGeom] = xy[0]; p[3 * this.targetGeom + 1] = xy[1];
  }

  reset(startXY, rng, xyNoise = true) {
    const m = this.model, d = this.data, s = this.meta.reset_noise_scale;
    this.mj.mj_resetData(m, d);
    const qpos0 = m.qpos0;
    for (let i = 0; i < this.nq; i++) d.qpos[i] = qpos0[i] + rng.uniform(-s, s);
    for (let i = 0; i < this.nv; i++) d.qvel[i] = s * rng.normal();
    const n = xyNoise ? this.meta.xy_noise : 0;
    d.qpos[0] = startXY[0] + rng.uniform(-n, n);
    d.qpos[1] = startXY[1] + rng.uniform(-n, n);
    this.mj.mj_forward(m, d);
    this.t = 0; this.success = false; this.done = false;
  }

  copyStateFrom(other) {
    this.data.qpos.set(other.data.qpos); this.data.qvel.set(other.data.qvel);
    this.mj.mj_forward(this.model, this.data);
    this.t = 0; this.success = false; this.done = false;
  }

  obs() {
    const o = new Float32Array(this.nq + this.nv);
    o.set(this.data.qpos, 0); o.set(this.data.qvel, this.nq);
    return o;
  }

  xy() { return [this.data.qpos[0], this.data.qpos[1]]; }
  distToGoal() { const [x, y] = this.xy(); return Math.hypot(x - this.goal[0], y - this.goal[1]); }

  // One environment step (frame_skip substeps). onSubstep(k) is called after each substep.
  step(action, onSubstep) {
    const d = this.data;
    for (let i = 0; i < this.nu; i++) d.ctrl[i] = Math.max(-1, Math.min(1, action[i]));
    for (let k = 0; k < this.meta.frame_skip; k++) {
      this.mj.mj_step(this.model, d);
      if (onSubstep) onSubstep(k);
    }
    this.t += 1;
    this.success = this.distToGoal() <= this.meta.goal_tol;
    this.done = this.success || this.t >= this.meta.max_episode_steps;
    return this.success;
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

  // Static description of every geom for the renderer.
  describe() {
    const m = this.model, d = this.data, geoms = [];
    for (let g = 0; g < this.ngeom; g++) {
      const mat = m.geom_matid[g];
      const rgba = mat >= 0 ? Array.from(m.mat_rgba.slice(4 * mat, 4 * mat + 4)) : Array.from(m.geom_rgba.slice(4 * g, 4 * g + 4));
      geoms.push({
        id: g, name: m.geom(g).name, type: m.geom_type[g], body: m.geom_bodyid[g],
        size: Array.from(m.geom_size.slice(3 * g, 3 * g + 3)), rgba,
        pos: Array.from(d.geom_xpos.slice(3 * g, 3 * g + 3)), mat: Array.from(d.geom_xmat.slice(9 * g, 9 * g + 9)),
      });
    }
    return { geoms, dynamic: this.dynamicGeoms };
  }
}
