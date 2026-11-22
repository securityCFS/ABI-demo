// Base policies (Flow BC samplers) and ABI's chunk critic as ONNX graphs, and the guidance rules
// of the paper: best-of-N ranking, QGF (critic-gradient steps inside the flow sampler), and QGF
// followed by best-of-4. The critic and the base policies are frozen; a goal only changes the
// reward weights w on the critic's reference states.
import * as ort from '../vendor/ort/ort.wasm.min.mjs';

const IN_NODE = typeof process !== 'undefined' && !!(process.versions && process.versions.node);
if (!IN_NODE) ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;

const T = (data, dims) => new ort.Tensor('float32', data, dims);

// Orthogonal projector onto the first `keep` DCT-II bases over the chunk's time axis.
function dctLowpass(H, keep) {
  const B = [];
  for (let k = 0; k < keep; k++) {
    const row = new Float64Array(H);
    for (let n = 0; n < H; n++) row[n] = Math.cos(Math.PI * (n + 0.5) * k / H) * Math.sqrt(2 / H) / (k === 0 ? Math.SQRT2 : 1);
    B.push(row);
  }
  const P = new Float64Array(H * H);
  for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) { let s = 0; for (let k = 0; k < keep; k++) s += B[k][i] * B[k][j]; P[i * H + j] = s; }
  return P;
}

export class Guide {
  static async load(fetchBytes, meta, progress) {
    const g = new Guide(); g.meta = meta;
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const open = async (f, label) => { progress && progress(`Loading ${label}…`); return ort.InferenceSession.create(await fetchBytes(f), opts); };
    g.actors = [];
    for (const a of meta.actors) {
      g.actors.push({ ...a, propose: await open(a.propose, a.label), velocity: a.velocity ? await open(a.velocity, a.label + ' (flow field)') : null });
    }
    g.score = await open(meta.critic.score, "ABI's critic");
    // The gradient graph is only needed by QGF: it is fetched on first use, which halves the initial download.
    g.grad = null; g.openGrad = meta.critic.grad ? () => open(meta.critic.grad, "ABI's critic gradient") : null;
    // Reference states: positions for goal rules (mazes), or none when each task ships its labels (DMC).
    if (meta.critic.ref_xy) {
      const ref = await fetchBytes(meta.critic.ref_xy);
      g.refXY = new Float32Array(ref.buffer, ref.byteOffset, ref.byteLength / 4).slice();
      g.R = g.refXY.length / 2;
    } else { g.refXY = null; g.R = meta.critic.R; }
    g.window = meta.window; g.sd = meta.sd; g.ad = meta.ad; g.H = meta.chunk;
    g.flowSteps = meta.flow_steps; g.P = dctLowpass(g.H, meta.qgf ? meta.qgf.keep_dct : 3);
    g.defaultActor = meta.default_actor || g.actors[0].id;
    g.w = new Float32Array(g.R);
    return g;
  }

  info() {
    return {
      actors: this.actors.map(a => ({ id: a.id, label: a.label, short: a.short, note: a.note })),
      qgf: !!(this.openGrad && this.actors.every(a => a.velocity)), footer: this.meta.footer || '', R: this.R,
      reward: this.meta.reward, taskAlpha: this.meta.task_alpha || null,
      guideActors: this.meta.guide_actors || [],      // RL actors offered as guided policies (others: comparison only)
    };
  }

  // Reward label of each reference state for a goal: the environment's success test at that state.
  setGoal(goal) {
    const r = this.meta.reward, tol = r.radius, mask = new Uint8Array(this.R);
    let count = 0;
    for (let k = 0; k < this.R; k++) {
      const hit = Math.hypot(this.refXY[2 * k] - goal[0], this.refXY[2 * k + 1] - goal[1]) <= tol;
      mask[k] = hit ? 1 : 0; count += mask[k];
      this.w[k] = hit ? r.hit : r.miss;
    }
    return { mask, count, total: this.R };
  }

  // Reward labels of the reference states given directly (a task's own labels, Float32Array of length R).
  setW(w) {
    if (w.length !== this.R) throw new Error(`labels have ${w.length} entries, the critic has ${this.R} reference states`);
    this.w.set(w); let count = 0; for (let k = 0; k < this.R; k++) count += w[k] !== 0;
    return { count, total: this.R };
  }

  buildWindow(history) {
    const L = this.window, sd = this.sd, h = new Float32Array(L * sd), v = new Float32Array(L);
    const n = Math.min(L, history.length), off = L - n;
    for (let k = 0; k < n; k++) { h.set(history[history.length - n + k], (off + k) * sd); v[off + k] = 1; }
    return [T(h, [L, sd]), T(v, [L])];
  }

  actor(id) { return this.actors.find(a => a.id === id) || this.actors[0]; }

  async propose(actor, state, noise, N) {
    const out = await actor.propose.run({ state: T(state, [this.sd]), noise: T(noise, [N, this.H, this.ad]) });
    return out.actions.data;
  }

  async scores(win, cands, N) {
    const out = await this.score.run({ history: win[0], history_valid: win[1], candidates: T(cands, [N, this.H, this.ad]), w: T(this.w, [this.R]) });
    return out.scores.data;
  }

  async gradient(win, u, N) {
    if (!this.grad) this.grad = await (this.gradPending = this.gradPending || this.openGrad());
    const out = await this.grad.run({ history: win[0], history_valid: win[1], candidates: T(u, [N, this.H, this.ad]), w: T(this.w, [this.R]) });
    return out.grad.data;
  }

  // QGF (state_qgf.GuidedFlow): at flow time t the Euler endpoint u1 = clip(x + (1 - t) v) is scored,
  // its gradient is projected onto the DCT low-pass and applied as a step of per-coordinate RMS alpha.
  // maskDims: action dims whose guidance push is zeroed before the norm (the paper's gripper "mask" rule); the step
  // factor keeps sqrt(D) with the masked dims counted, and the flow velocity still moves them.
  async qgf(actor, state, win, noise, N, alpha, maskDims = []) {
    const H = this.H, ad = this.ad, D = H * ad, x = Float32Array.from(noise), K = this.flowSteps;
    const u1 = new Float32Array(N * D), pg = new Float64Array(D), mask = new Float64Array(ad).fill(1);
    for (const d of maskDims) mask[d] = 0;
    for (let i = 0; i < K; i++) {
      const t = i / K;
      const v = (await actor.velocity.run({ state: T(state, [this.sd]), x_t: T(x, [N, H, ad]), t: T(new Float32Array([t]), [1]) })).velocity.data;
      if (alpha === 0) { for (let k = 0; k < x.length; k++) x[k] += v[k] / K; continue; }
      for (let k = 0; k < x.length; k++) u1[k] = Math.max(-1, Math.min(1, x[k] + (1 - t) * v[k]));
      const g = await this.gradient(win, u1, N);
      for (let n = 0; n < N; n++) {
        let norm = 0;
        for (let a = 0; a < H; a++) for (let d = 0; d < ad; d++) {
          let s = 0; for (let b = 0; b < H; b++) s += this.P[a * H + b] * g[n * D + b * ad + d];
          s *= mask[d]; pg[a * ad + d] = s; norm += s * s;
        }
        const scale = alpha * Math.sqrt(D) / Math.max(Math.sqrt(norm), 1e-12);
        for (let k = 0; k < D; k++) x[n * D + k] += v[n * D + k] / K + scale * pg[k];
      }
    }
    for (let k = 0; k < x.length; k++) x[k] = Math.max(-1, Math.min(1, x[k]));
    return x;
  }

  // ABI guiding an external single-step actor (paper, App. "Guiding external trained actors"). cands: the actor's N
  // candidate actions (slot 0 = its own action). Each candidate a_i is completed into K 8-step chunks [a_i, tail_k] by the
  // frozen base policy (replacement inpainting along the rectified-flow path x_t = (1-t) eps + t a, 10 Euler steps), with
  // the K tail noises shared by all candidates; a candidate's score is the mean over its K chunks, and the best action is
  // executed (one action, replanning at every step).
  async decideActor(cands, state, history, rng, K = 4) {
    const t0 = performance.now(), H = this.H, ad = this.ad, D = H * ad, N = cands.length, M = K * N, S = this.flowSteps;
    const base = this.actor(this.defaultActor);
    const noise = rng.fillNormal(new Float32Array(K * D));
    const x = new Float32Array(M * D), e0 = new Float32Array(M * ad), a0 = new Float32Array(M * ad);
    for (let k = 0; k < K; k++) for (let i = 0; i < N; i++) {          // row k * N + i: tail k of candidate i
      const m = k * N + i;
      x.set(noise.subarray(k * D, (k + 1) * D), m * D);
      e0.set(noise.subarray(k * D, k * D + ad), m * ad);
      for (let d = 0; d < ad; d++) a0[m * ad + d] = Math.max(-1, Math.min(1, cands[i][d]));
    }
    const st = T(state, [this.sd]);
    for (let j = 0; j < S; j++) {
      const t = j / S;
      for (let m = 0; m < M; m++) for (let d = 0; d < ad; d++) x[m * D + d] = (1 - t) * e0[m * ad + d] + t * a0[m * ad + d];
      const v = (await base.velocity.run({ state: st, x_t: T(x, [M, H, ad]), t: T(new Float32Array([t]), [1]) })).velocity.data;
      for (let q = 0; q < x.length; q++) x[q] += v[q] / S;
    }
    for (let m = 0; m < M; m++) for (let d = 0; d < ad; d++) x[m * D + d] = a0[m * ad + d];
    for (let q = 0; q < x.length; q++) x[q] = Math.max(-1, Math.min(1, x[q]));
    const raw = await this.scores(this.buildWindow(history), x, M);
    const scores = new Array(N).fill(0);
    for (let k = 0; k < K; k++) for (let i = 0; i < N; i++) scores[i] += raw[k * N + i] / K;
    let chosen = 0;
    for (let i = 1; i < N; i++) if (scores[i] > scores[chosen]) chosen = i;
    return { actions: [cands[chosen]], scores, chosen, N, mode: 'actor_bfn', ms: performance.now() - t0 };
  }

  // One decision: returns the 8-step chunk to execute (as a list of actions) and the candidate scores.
  async decide(cfg, state, history, rng) {
    const t0 = performance.now(), actor = this.actor(cfg.actor), D = this.H * this.ad;
    // qgf_bfn is the paper's QGF+BFN4u consumer: 4 QGF draws and 1 unguided draw, ranked by the critic.
    const mode = cfg.mode, N = mode === 'bfn' ? cfg.N : mode === 'qgf_bfn' ? 5 : 1;
    const win = this.buildWindow(history);
    let cands;
    const md = cfg.maskDims || [];
    if (mode === 'qgf') cands = await this.qgf(actor, state, win, rng.fillNormal(new Float32Array(D)), 1, cfg.alpha, md);
    else if (mode === 'qgf_bfn') {
      cands = new Float32Array(5 * D);
      cands.set(await this.qgf(actor, state, win, rng.fillNormal(new Float32Array(4 * D)), 4, cfg.alpha, md), 0);
      cands.set(await this.propose(actor, state, rng.fillNormal(new Float32Array(D)), 1), 4 * D);
    } else cands = await this.propose(actor, state, rng.fillNormal(new Float32Array(N * D)), N);
    let scores = null, chosen = 0;
    if (mode !== 'unguided' || cfg.scoreUnguided) {
      scores = await this.scores(win, cands, N);
      for (let k = 1; k < N; k++) if (scores[k] > scores[chosen]) chosen = k;
    }
    const actions = [];
    for (let s = 0; s < this.H; s++) actions.push(cands.slice(chosen * D + s * this.ad, chosen * D + (s + 1) * this.ad));
    return { actions, scores: scores ? Array.from(scores) : null, chosen, N, mode, ms: performance.now() - t0 };
  }
}
