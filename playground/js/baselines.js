// Zero-shot task-conditioned actors for the comparison ("ghost") run: the reproduced TD-JEPA actor and the one-step FB
// actor. Each reads a task the way its own zero-shot evaluation does: the goal labels its own inference samples (the
// same goal rule as ABI), the method's task inference turns the labels into a latent z, and the actor then acts with
// a = pi(s, z) at every environment step (H_exec = 1). Nothing here is trained or tuned in the browser.
import * as ort from '../vendor/ort/ort.wasm.min.mjs';
import { Rng } from './sim.js';

const IN_NODE = typeof process !== 'undefined' && !!(process.versions && process.versions.node);
if (!IN_NODE) ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;

const T = (data, dims) => new ort.Tensor('float32', data, dims);
const f32 = bytes => new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();

// Cholesky factor (lower, row-major, float64) of a symmetric positive definite n x n matrix.
function cholesky(A, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (!(s > 0)) throw new Error('inference Gram matrix is not positive definite');
        L[i * n + i] = Math.sqrt(s);
      } else L[i * n + j] = s / L[j * n + j];
    }
  }
  return L;
}

function cholSolve(L, b, n) {
  const y = new Float64Array(n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}

export class ZActor {
  // spec: one entry of baselines.json "policies"; fetchBytes(path) resolves paths relative to models/<env>/ (as for Guide).
  static async load(fetchBytes, spec, progress) {
    const a = new ZActor(); a.spec = spec;
    const inf = spec.inference;
    progress && progress(`Loading ${spec.short || spec.label}…`);
    a.session = await ort.InferenceSession.create(await fetchBytes(spec.onnx), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    a.sd = spec.sd; a.ad = spec.ad; a.d = spec.z_dim; a.M = inf.M;
    a.B = f32(await fetchBytes(inf.B));
    if (a.B.length !== a.M * a.d) throw new Error(`${inf.B}: ${a.B.length} values, expected ${a.M} x ${a.d}`);
    const lp = inf.label_points;
    a.q = f32(await fetchBytes(lp.file)); a.qd = lp.dim;
    if (a.q.length !== a.M * a.qd) throw new Error(`${lp.file}: ${a.q.length} values, expected ${a.M} x ${a.qd}`);
    a.r = new Float64Array(a.M);
    a.z = new Float32Array(a.d); a.valid = false; a.count = 0;
    a.rng = new Rng(spec.noise ? (spec.noise.default_seed || 1) : 1);
    if (inf.z_rule.kind === 'lstsq_project') {
      // Normal equations of the least-squares fit, factored once: G = B^T B (float64).
      const d = a.d, M = a.M, B = a.B, G = new Float64Array(d * d);
      for (let k = 0; k < M; k++) {
        const o = k * d;
        for (let i = 0; i < d; i++) { const bi = B[o + i]; if (bi === 0) continue; for (let j = 0; j <= i; j++) G[i * d + j] += bi * B[o + j]; }
      }
      for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) G[j * d + i] = G[i * d + j];
      a.L = cholesky(G, d);
    }
    return a;
  }

  info() {
    const s = this.spec;
    return { id: s.id, label: s.label, short: s.short, note: s.note, deterministic: s.deterministic, hexec: s.H_exec, M: this.M };
  }

  // Noise source of a stochastic actor (TD-JEPA's noise-conditioned actor): any object with fillNormal(Float32Array).
  setRng(rng) { this.rng = rng; }

  // Labels from a goal with the method's goal rule (the environment's success test at each inference sample's arrival
  // state). goal: [x, y] (antmaze) or [x, y(, z)] (cube; a missing z takes the rule's goal_fill, the table height).
  setGoal(goal) {
    const rule = this.spec.inference.label_rule, qd = this.qd, q = this.q;
    const g = Array.from({ length: qd }, (_, j) => (goal[j] ?? (rule.goal_fill ? rule.goal_fill[j] : undefined)));
    if (g.some(v => typeof v !== 'number' || !isFinite(v))) throw new Error(`goal needs ${qd} coordinates`);
    const r2 = rule.radius;
    for (let k = 0; k < this.M; k++) {
      let s = 0;
      for (let j = 0; j < qd; j++) { const dlt = q[k * qd + j] - g[j]; s += dlt * dlt; }
      const dist = Math.sqrt(s);
      this.r[k] = (rule.inclusive ? dist <= r2 : dist < r2) ? rule.hit : rule.miss;
    }
    return this.inferZ();
  }

  // Labels from an arbitrary rule: labelFn(point: Float32Array(qd), k) -> number.
  setGoalLabels(labelFn) {
    for (let k = 0; k < this.M; k++) this.r[k] = +labelFn(this.q.subarray(k * this.qd, (k + 1) * this.qd), k);
    return this.inferZ();
  }

  // Labels given directly (length M).
  setLabels(r) {
    if (r.length !== this.M) throw new Error(`labels have ${r.length} entries, expected ${this.M}`);
    for (let k = 0; k < this.M; k++) this.r[k] = r[k];
    return this.inferZ();
  }

  // Task inference of the method, in float64, from this.r.
  inferZ() {
    const zr = this.spec.inference.z_rule, d = this.d, M = this.M, B = this.B, r = this.r;
    let count = 0; for (let k = 0; k < M; k++) count += r[k] !== 0;
    const x = new Float64Array(d);
    if (zr.kind === 'lstsq_project') {
      // TD-JEPA reward_inference + project_z: x = argmin ||B x - r||, z = sqrt(d) x / max(||x||, 1e-12).
      const b = new Float64Array(d);
      for (let k = 0; k < M; k++) { const rk = r[k]; if (rk === 0) continue; for (let i = 0; i < d; i++) b[i] += rk * B[k * d + i]; }
      x.set(cholSolve(this.L, b, d));
    } else if (zr.kind === 'softmax_reward_mean') {
      // one-step FB infer_latent: w = softmax(T r) over the samples, m = mean_k (w_k r_k B_k), z = sqrt(d) m / ||m||.
      let mx = -Infinity; for (let k = 0; k < M; k++) mx = Math.max(mx, zr.temperature * r[k]);
      let Z = 0; const w = new Float64Array(M);
      for (let k = 0; k < M; k++) { w[k] = Math.exp(zr.temperature * r[k] - mx); Z += w[k]; }
      for (let k = 0; k < M; k++) { const c = w[k] / Z * r[k] / M; if (c === 0) continue; for (let i = 0; i < d; i++) x[i] += c * B[k * d + i]; }
    } else throw new Error('unknown z rule ' + zr.kind);
    let n = 0; for (let i = 0; i < d; i++) n += x[i] * x[i]; n = Math.sqrt(n);
    // No positive sample: TD-JEPA's projection gives z = 0, one-step FB's normalisation is undefined (0/0). Both are
    // reported as invalid and leave z = 0.
    this.valid = count > 0 && n > 0;
    const scale = this.valid ? Math.sqrt(d) / Math.max(n, zr.kind === 'lstsq_project' ? 1e-12 : 0) : 0;
    for (let i = 0; i < d; i++) this.z[i] = x[i] * scale;
    this.count = count;
    return { count, total: M, valid: this.valid, z: this.z };
  }

  // One action for the current observation (Float32Array(sd)); the actor is queried at every environment step.
  async act(obs) {
    const feeds = { obs: T(obs instanceof Float32Array ? obs : Float32Array.from(obs), [1, this.sd]), z: T(this.z, [1, this.d]) };
    if (this.spec.noise) feeds[this.spec.noise.input] = T(this.rng.fillNormal(new Float32Array(this.ad)), [1, this.ad]);
    const out = await this.session.run(feeds);
    return Float32Array.from(out.action.data);
  }

  // As act, with the noise given (Float32Array(ad)); for checks against recorded actions.
  async actWithNoise(obs, noise) {
    const feeds = { obs: T(Float32Array.from(obs), [1, this.sd]), z: T(this.z, [1, this.d]) };
    if (this.spec.noise) feeds[this.spec.noise.input] = T(Float32Array.from(noise), [1, this.ad]);
    return Float32Array.from((await this.session.run(feeds)).action.data);
  }

  // The actor's own action, as it acts unguided: TD-JEPA's noise-conditioned actor draws its noise from `rng` (one draw
  // per step), one-step FB takes its mode, clipped to [-1, 1] (paper, App. "Guiding external trained actors").
  async ownAction(obs, rng) {
    if (this.spec.noise) return this.actWithNoise(obs, rng.fillNormal(new Float32Array(this.ad)));
    const a = await this.act(obs);
    for (let i = 0; i < a.length; i++) a[i] = Math.max(-1, Math.min(1, a[i]));
    return a;
  }

  // The N candidate actions that ABI ranks (same appendix): slot 0 is the own action, slots 1..N-1 are further samples,
  // from TD-JEPA's own sampler, or for one-step FB (log std fixed and untrained) its mode plus Gaussian noise of scale
  // OSFB_SIGMA = 0.2 (the actor noise of TD-JEPA), clipped to [-1, 1].
  async candidates(obs, own, rng, N = 16) {
    const c = [own];
    for (let i = 1; i < N; i++) {
      const e = rng.fillNormal(new Float32Array(this.ad));
      if (this.spec.noise) c.push(await this.actWithNoise(obs, e));
      else c.push(own.map((v, d) => Math.max(-1, Math.min(1, v + OSFB_SIGMA * e[d]))));
    }
    return c;
  }
}
const OSFB_SIGMA = 0.2;
