// Simulation and inference run here, off the UI thread. The main thread renders the poses it receives.
// The environment module is chosen by env_meta.kind (js/envs/<kind>.js), see ENV_INTERFACE.md.
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { Rng } from './sim.js';
import { Guide } from './guide.js';

let mj, env, ghost, guide, meta, envMeta, modelDir;
// policy: the policy ABI guides, a Flow BC id of demo.json (cfg.actor = that id) or a zero-shot RL actor of
// baselines.json (cfg.actor = the default Flow BC, whose flow completes the actor's candidates). ghostPolicy: the gray
// run, 'same' (the same policy without ABI), 'base' (Flow BC without ABI) or an actor id.
let cfg = { actor: null, policy: null, mode: 'bfn', N: 16, alpha: 0.1, hexec: 1, ghost: true, ghostPolicy: 'same' };
let hold = false;     // video capture: the page asks the simulation to wait while its frame queue is long
let baselineSpecs = [], baselineMods = new Map(), ZActor = null;
let speed = 1, playing = false, busy = false, stepOnce = false, epoch = 0, leg = 0;   // leg: bumped by every goal or task change
let episodeSeed = 1, rngMain, rngGhost, rngReset, ownMain, ownGhost, rngActor;
let plan = [], ghostPlan = [], history = [], ghostHistory = [];
let wallStart = 0, simStart = 0, recentMs = [], DT = 0.1, frameEvery = 1, EnvClass = null, settingTask = 0, envMod = null;
let refFeatures = null;
const labelCache = new Map();    // label file -> promise of its Float32Array (prefetched after start-up)
const labelFile = f => { if (!labelCache.has(f)) labelCache.set(f, fetchBytes(new URL(f, modelDir), true).then(f32)); return labelCache.get(f); };
// The simulation runs ahead of the display by a lead that covers the slowest recent decision, so the
// display plays at an even pace while a decision is computed.
const leadMs = () => Math.min(1500, Math.max(250, 1.3 * Math.max(0, ...recentMs) + 80));

const post = (m, t) => self.postMessage(m, t || []);
const sleep = ms => new Promise(r => setTimeout(r, ms));
// Let queued messages (pause, task, goal, settings) run. Awaiting ONNX inference only resolves microtasks, so a
// simulation that runs behind the wall clock would otherwise never hand the thread back until its episode ends.
const yieldTask = () => new Promise(r => { const c = new MessageChannel(); c.port1.onmessage = () => { c.port1.close(); r(); }; c.port2.postMessage(0); });
const fetchText = async p => { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.text(); };
const fetchJSON = async p => JSON.parse(await fetchText(p));
// Downloads. A page needs about 40 MB before it can start (ABI's critic alone is 24 MB on the wire). init() requests
// every file of that set at once instead of one after another, the loading message counts the megabytes and the time
// left, and when no download receives a byte for STALL_TICKS free seconds of this thread, the open requests are restarted:
// on a slow link the sequential loader showed one stage message for minutes, and a stalled request kept the page
// loading forever. The stall test is over all downloads together: over HTTP/2 the browser serves the requests of one
// page roughly one after another, so a request queued behind the 24 MB critic receives nothing for a long time while
// the connection is busy, and a per-request test restarted it (seen on the live site, 2026-11-22).
// sizes.json (tests/make_sizes.py) gives [bytes, bytes on the wire] of every file under env/, models/ and vendor/: GitHub
// Pages sends most files gzip-compressed, and the browser hands over the decompressed bytes. The wasm runtimes and the
// meshes shrink about 4x while the weights do not, so the progress counts wire bytes (each file's bytes times its ratio)
// and the time left is not underestimated while the runtimes arrive.
const STALL_TICKS = 20, TRIES = 4;
const ROOT = new URL('../', import.meta.url);
const ahead = new Map();          // URL -> promise of its bytes, requested ahead of use and handed out once
const bar = { got: 0, total: 0, open: 0, shown: 0, label: '', stage: '', note: '', hist: [] };   // downloads the user waits for
let sizes = {};
const running = new Set();       // AbortControllers of the open downloads
let idle = 0, watch = 0;         // free seconds of this thread without a byte on any download
function watchdog(on) {
  if (on && !watch) { idle = 0; watch = setInterval(() => { if (++idle >= STALL_TICKS) { idle = 0; for (const ac of running) ac.abort(); } }, 1000); }
  if (!on && watch && running.size === 0) { clearInterval(watch); watch = 0; }
}
const sizeOf = u => { const v = sizes[String(u).slice(ROOT.href.length)]; return Array.isArray(v) ? v : [v || 0, v || 0]; };
function progress(msg) { bar.stage = msg; report(true); }
function report(force) {
  const now = performance.now();
  if (!force && now - bar.shown < 250) return;
  bar.shown = now;
  let msg = bar.stage;
  if (bar.open > 0) {
    const mb = x => (x / 1e6).toFixed(1), total = Math.max(bar.total, bar.got);
    msg = `${bar.label}: ${mb(bar.got)} of ${mb(total)} MB`;
    // time left at the rate of the last 10 s: the mean rate since the start overstates it once the small files are
    // done and only the critic is still arriving
    bar.hist.push([now, bar.got]);
    while (bar.hist.length > 2 && now - bar.hist[1][0] > 10000) bar.hist.shift();
    const [t1, g1] = bar.hist[0], rate = (bar.got - g1) / ((now - t1) / 1000);
    if (now - t1 > 3000 && rate > 0) {
      const left = (total - bar.got) / rate;
      if (left > 90) msg += `, about ${Math.ceil(left / 60)} min left`;
      else if (left > 8) msg += `, about ${Math.ceil(left / 10) * 10} s left`;
    }
    if (bar.note) msg += ` (${bar.note})`;
  }
  post({ type: 'progress', msg });
}
// quiet: a background download (label files), not counted in the loading message.
async function download(url, quiet) {
  if (!quiet) {
    if (bar.open === 0) Object.assign(bar, { got: 0, total: 0, note: '', hist: [], label: bar.stage.replace(/…$/, '') });
    bar.open++;
  }
  const [raw, wire] = sizeOf(url), ratio = raw > 0 ? wire / raw : 1;
  if (!quiet) bar.total += wire;
  try {
    for (let k = 1; ; k++) {
      const ac = new AbortController();
      let got = 0;
      // A tick only runs while this thread is free, so time spent compiling a model does not count as a stall.
      running.add(ac); watchdog(true);
      try {
        const r = await fetch(url, { signal: ac.signal });
        if (!r.ok) throw Object.assign(new Error(`${url}: ${r.status}`), { final: true });
        const reader = r.body.getReader(), parts = [];
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          idle = 0; parts.push(value); got += value.length;
          if (!quiet) { bar.got += value.length * ratio; report(false); }
        }
        // one whole chunk is returned as is; callers view the bytes as Float32Array, so a view at an offset is copied
        if (parts.length === 1 && parts[0].byteOffset === 0 && parts[0].byteLength === parts[0].buffer.byteLength) return parts[0];
        const out = new Uint8Array(got);
        let o = 0;
        for (const q of parts) { out.set(q, o); o += q.length; }
        return out;
      } catch (e) {
        if (!quiet) bar.got -= got * ratio;
        if (e.final || k >= TRIES) throw e;
        if (!quiet) { bar.note = 'the connection stalled, the download was restarted'; report(true); }
      } finally {
        running.delete(ac); watchdog(false);
      }
    }
  } finally {
    if (!quiet) { bar.open--; report(true); }
  }
}
function prefetch(u) {
  const k = String(u);
  if (!ahead.has(k)) { const pr = download(k, false); pr.catch(() => {}); ahead.set(k, pr); }
}
const fetchBytes = (p, quiet) => {
  const k = String(p), pr = ahead.get(k);
  if (pr) { ahead.delete(k); return pr; }
  return download(k, quiet);
};

async function init(name, autoplay) {
  const envDir = new URL(`../env/${name}/`, import.meta.url);
  modelDir = new URL(`../models/${name}/`, import.meta.url);
  const [sz, em, dm] = await Promise.all([fetchJSON(new URL('sizes.json', ROOT)).catch(() => ({})),
    fetchJSON(new URL('env_meta.json', envDir)), fetchJSON(new URL('demo.json', modelDir))]);
  sizes = sz; envMeta = em;
  // the start-up set, all requested now: the two WebAssembly runtimes, the meshes, the base policies and the critic
  // asset_suffix: the meshes are stored as <name>.bin on GitHub Pages, which compresses .bin but not .stl (9.0 -> 2.3 MB)
  const assetDir = new URL(em.asset_dir || 'assets/', envDir), assetURL = n => new URL(n + (em.asset_suffix || ''), assetDir), MJ_WASM = new URL('../vendor/mujoco/mujoco.wasm', import.meta.url);
  const ORT_WASM = new URL('../vendor/ort/ort-wasm-simd-threaded.wasm', import.meta.url);
  progress('Downloading the simulator and models…');
  [MJ_WASM, ORT_WASM, ...(em.assets || []).map(assetURL),
   ...dm.actors.flatMap(a => [a.propose, a.velocity]).filter(Boolean).map(f => new URL(f, modelDir)),
   new URL(dm.critic.score, modelDir), ...(dm.critic.ref_xy ? [new URL(dm.critic.ref_xy, modelDir)] : [])].forEach(prefetch);
  bar.stage = 'Preparing the simulation…';      // shown when the downloads are done
  mj = await loadMujoco({ wasmBinary: await fetchBytes(MJ_WASM) });
  progress('Preparing the simulation…');
  envMod = await import(`./envs/${envMeta.kind || 'locomaze'}.js`);
  const { Env } = envMod;
  const xml = await fetchText(new URL('scene.xml', envDir));
  if (Env.prepare) await Env.prepare(mj, envDir, fetchBytes);      // e.g. mesh assets into the WASM file system
  let assets = null;                                                // mesh files the scene references by name
  if (envMeta.assets) {
    const files = await Promise.all(envMeta.assets.map(n => fetchBytes(assetURL(n))));
    assets = Object.fromEntries(envMeta.assets.map((n, k) => [n, files[k]]));
  }
  EnvClass = Env;
  env = adapt(new Env(mj, xml, envMeta, assets)); ghost = adapt(new Env(mj, xml, envMeta, null, env));   // the ghost shares the compiled model
  DT = envMeta.control_dt || envMeta.control_timestep || 0.1;
  // frames for the display: every substep for short control steps, about 5 per control step otherwise
  const sub = envMeta.substeps || envMeta.n_sub_steps || envMeta.frame_skip || 1;
  frameEvery = Math.max(1, Math.floor(sub / Math.max(1, Math.round(DT * 120))));   // at most about 120 frames per simulated second
  meta = dm;
  Guide.useWasm(await fetchBytes(ORT_WASM));
  guide = await Guide.load(p => fetchBytes(new URL(p, modelDir)), meta, progress);
  cfg.actor = guide.defaultActor; cfg.policy = cfg.actor;
  // Comparison policies for the ghost run (task-conditioned zero-shot actors), loaded on first selection.
  try {
    const spec = await fetchJSON(new URL('baselines/baselines.json', modelDir));
    baselineSpecs = spec.policies || spec;
    ({ ZActor } = await import('./baselines.js'));
  } catch (e) { baselineSpecs = []; }
  if (meta.default_mode) cfg.mode = meta.default_mode;
  rngReset = new Rng(1); env.reset(rngReset);
  post({ type: 'ready', name, desc: env.describe(), envMeta, features: env.ui, tasks: env.tasks(), models: guide.info(),
         refXY: guide.refXY, cfg, marker: env.marker(),
         baselines: baselineSpecs.map(b => ({ id: b.id, label: b.label, short: b.short, note: b.note })) });
  await applyTask(0, true);
  if (autoplay) { playing = true; wallStart = performance.now(); simStart = 0; loop(); }
  // fetch the other tasks' label files in the background, so that a task switch answers at once
  for (const f of Object.values(meta.critic.labels || {})) labelFile(f).catch(() => labelCache.delete(f));
  if (EnvClass.goalLabels && meta.critic.ref_xyz) refPointsFile().catch(() => {});      // first clicked goal
}

// Defaults for the optional parts of the environment interface (ENV_INTERFACE.md lists the required ones).
function adapt(e) {
  const meta = e.meta || envMeta;
  // UI flags (`ui`); an environment may name them `features` (the ant), DMC uses `features()` for reward features
  e.ui = e.features && typeof e.features === 'object' ? e.features : meta.kind === 'manipspace'
    ? { clickGoal: 'table', clickStart: false, taskSwitch: 'restart', successTask: true }
    : { taskSwitch: 'continue', successTask: meta.success_task === true,
        // task sliders between the paper's presets are off: a closed-loop sweep showed no graded behaviour between
        // presets (tests/results/knob_sweep), so the page offers the presets only
        knobs: meta.show_knobs ? meta.knobs : null };
  if (!e.tasks) e.tasks = () => meta.tasks.map((t, k) => ({ label: t.label || `Task ${k + 1}`,
    title: t.title || (t.goal_xyz ? `OGBench evaluation task ${k + 1} (${t.task_name || ''}): move the cube from (${t.init_xyz.slice(0, 2)}) to (${t.goal_xyz.slice(0, 2)})` : '') }));
  if (!e.marker) e.marker = () => ({ goal: e.goal ? Array.from(e.goal) : null });
  if (!e.newLeg) e.newLeg = () => { e.t = 0; e.ret = 0; e.success = e.success === null ? null : false; e.done = false; };
  if (!e.setStart) e.setStart = () => {};
  if (!e.setGoal) e.setGoal = () => {};
  if (e.taskIdx === undefined) { const set = e.setTask.bind(e); e.taskIdx = 0; e.setTask = k => { e.taskIdx = k; set(k); }; }
  if (e.ret === undefined) e.ret = 0;
  return e;
}

async function baseline(id) {
  if (!baselineMods.has(id)) {
    const spec = baselineSpecs.find(b => b.id === id);
    progress(`Loading ${spec.label}…`);
    const za = await ZActor.load(p => fetchBytes(new URL(p, modelDir)), spec);   // spec paths are relative to models/<env>/
    za.setRng(new Rng(episodeSeed * 15485863 + 7));
    baselineMods.set(id, za); await taskToBaseline(za);
  }
  return baselineMods.get(id);
}
// The comparison actors infer their task from the same goal, through their own zero-shot inference.
async function taskToBaseline(za) {
  if (!env.goal) return;
  const r = za.setGoal(env.goal);
  post({ type: 'ghost_info', id: za.spec.id, count: r.count, total: r.total, valid: r.valid });
}
async function taskToBaselines() { for (const za of baselineMods.values()) await taskToBaseline(za); }

const f32 = b => new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
let refPoints = null, refPointsP = null;
const refPointsFile = () => (refPointsP ||= fetchBytes(new URL(meta.critic.ref_xyz, modelDir), true).then(f32).catch(e => { refPointsP = null; throw e; }));
// Reward labels of the reference states for the current task or goal.
async function labels() {
  const L = meta.critic.labels, k = env.taskIdx, t = k !== null ? envMeta.tasks[k] : null;
  const f = L && t ? (L[t.id] || L[String(k + 1)] || L[k]) : null;
  settingTask = k;
  if (f) {
    const w = await labelFile(f);
    return { ...guide.setW(w), mask: null, settingTask, meanLabel: mean(w) };
  }
  // A knob value (DMC): the task's reward at every reference state, from the stored reward features; the QGF tau
  // is that of the nearest preset of the same family (meta.knob_alpha_rule).
  if (meta.critic.features && env.knobTask) {
    if (!refFeatures) { const b = await fetchBytes(new URL(meta.critic.features, modelDir), true); refFeatures = new Float64Array(b.buffer, b.byteOffset, b.byteLength / 8).slice(); }
    const w = envMod.labels(envMeta, refFeatures, env.knobTask);
    settingTask = envMeta.tasks.indexOf(envMod.nearestPreset(envMeta, env.knobTask));
    return { ...guide.setW(w), mask: null, settingTask, meanLabel: mean(w) };
  }
  // A clicked goal. Objects (cube): the environment's 3-D goal rule on the reference object positions; the QGF
  // settings are those of the nearest task (meta.click_alpha_rule). The ant: the guide's 2-D rule on the torso positions.
  if (EnvClass.goalLabels && meta.critic.ref_xyz) {
    if (!refPoints) refPoints = await refPointsFile();
    const w = Float32Array.from(EnvClass.goalLabels(refPoints, env.goal));
    if (meta.click_alpha_rule) {
      let best = Infinity;
      envMeta.tasks.forEach((tk, j) => { const g = tk.goal_xyz || [], d = Math.hypot(g[0] - env.goal[0], g[1] - env.goal[1]); if (d < best) { best = d; settingTask = j; } });
    }
    return { ...guide.setW(w), mask: null, settingTask };
  }
  return { ...guide.setGoal(env.goal), settingTask: null };
}
// QGF gripper rule of the paper (per task): mask the critic's push on the gripper dims, or guide them too.
function qgfMask() {
  const tg = meta.task_gripper, key = cfg.mode === 'qgf' ? 'qgf' : 'qgf_bfn';
  if (!tg || !cfg.mode.startsWith('qgf') || settingTask === null || !tg[key]) return [];
  return tg[key][settingTask] === 'mask' ? (meta.qgf.gripper_dims || []) : [];
}

function mean(w) { let s = 0; for (let i = 0; i < w.length; i++) s += w[i]; return s / w.length; }
// A knob value that equals a preset is that preset (the paper's label file and tau).
function presetOf(task) {
  return envMeta.tasks.findIndex(t => t.family === task.family &&
    Object.keys(task.params).every(q => Math.abs((t.params[q] ?? NaN) - task.params[q]) < 1e-9));
}
async function applyTask(k, restart) {
  if (typeof k === 'object' && k !== null) {
    const p = presetOf(k);
    if (p >= 0) k = p;
    else {
      env.setTask(k); ghost.setTask(k); env.taskIdx = null; ghost.taskIdx = null; env.knobTask = k; ghost.knobTask = k;
      post({ type: 'goal_info', task: null, knob: k, marker: env.marker(), ...(await labels()) });
      if (restart) { episodeSeed += 1; newEpisode(); sendPoses(); } else newLeg();
      return;
    }
  }
  env.knobTask = null; ghost.knobTask = null;
  env.setTask(k); ghost.setTask(k);
  post({ type: 'goal_info', task: k, marker: env.marker(), ...(await labels()) });
  await taskToBaselines();
  if (restart) { episodeSeed += 1; newEpisode(); sendPoses(); } else newLeg();
}

function newLeg() {           // continue from the current state under the new task or goal, with a fresh budget
  leg += 1; env.newLeg(); ghost.newLeg(); plan = []; ghostPlan = [];
  post({ type: 'leg_start', marker: env.marker(), task: env.taskIdx, knob: env.knobTask || null }); status();
}

function resetHistory(o) { return [o]; }
function pushHistory(h, o) { h.push(o); if (h.length > guide.window) h.shift(); }

function newEpisode() {
  epoch += 1;
  rngReset = new Rng(episodeSeed * 7919 + 13);
  rngMain = new Rng(episodeSeed * 104729 + 1);
  rngGhost = new Rng(episodeSeed * 104729 + 1);     // same noise stream as the main run's first draw
  // an RL actor's own action draws its noise from ownMain / ownGhost (the same stream, so the unguided gray run is the
  // green run without ABI), its further candidates and the flow tails from rngActor
  ownMain = new Rng(episodeSeed * 15485863 + 7); ownGhost = new Rng(episodeSeed * 15485863 + 7);
  rngActor = new Rng(episodeSeed * 15485863 + 11);
  env.reset(rngReset); ghost.copyStateFrom(env);
  for (const za of baselineMods.values()) za.setRng(new Rng(episodeSeed * 15485863 + 7));
  plan = []; ghostPlan = [];
  history = resetHistory(env.obs()); ghostHistory = resetHistory(ghost.obs());
  wallStart = performance.now(); simStart = 0;
  post({ type: 'episode_start', marker: env.marker(), task: env.taskIdx, seed: episodeSeed });
  status();
}

function runInfo(e) { return { t: e.t, ret: e.ret, success: e.success, done: e.done, hud: e.hud() }; }
function status(decision) {
  post({ type: 'status', ...runInfo(env), playing, ghost: cfg.ghost ? runInfo(ghost) : null, decision });
}

// One frame of the current poses (both runs), sent after a reset so the view shows the new start at once.
function sendPoses() {
  const L = 12 * env.dynamicGeoms.length;
  sendFrame([env.poses(new Float64Array(L))], cfg.ghost ? [ghost.poses(new Float64Array(L))] : null);
}
function sendFrame(mainBatch, ghostBatch) {
  const n = mainBatch.length, L = mainBatch[0].length;
  const m = new Float64Array(n * L); mainBatch.forEach((b, k) => m.set(b, k * L));
  let g = null;
  if (ghostBatch) { g = new Float64Array(n * L); ghostBatch.forEach((b, k) => g.set(b, k * L)); }
  post({ type: 'frames', n, dt: DT / Math.max(1, n), main: m, ghost: g, t: env.t }, g ? [m.buffer, g.buffer] : [m.buffer]);
}

const isActor = id => baselineSpecs.some(b => b.id === id);
// The gray run's policy: an actor id, or 'bc' (the Flow BC of cfg.actor without ABI).
function ghostOf(c) {
  const g = c.ghostPolicy || 'same';
  if (g === 'same') return isActor(c.policy) ? c.policy : 'bc';
  return g === 'base' ? 'bc' : g;
}

async function tick() {
  const myEpoch = epoch, myLeg = leg, t0 = performance.now();
  let decision = null;
  if (!env.done && plan.length === 0) {
    let d;
    if (isActor(cfg.policy)) {
      // ABI guiding an RL actor (paper, App. "Guiding external trained actors"): one action, replanning every step
      const za = await baseline(cfg.policy);
      if (myEpoch !== epoch) return;
      const obs = env.obs(), own = await za.ownAction(obs, ownMain);
      d = cfg.mode === 'unguided' ? { actions: [own], scores: null, chosen: 0, N: 1, mode: 'unguided', ms: performance.now() - t0 }
        : await guide.decideActor(await za.candidates(obs, own, rngActor, 16), obs, history, rngActor);
    } else d = await guide.decide({ ...cfg, maskDims: qgfMask() }, env.obs(), history, rngMain);
    if (myEpoch !== epoch || myLeg !== leg) { await yieldTask(); return; }   // the goal or task changed meanwhile: decide again
    plan = d.actions.slice(0, cfg.hexec); decision = d;
  }
  if (cfg.ghost && !ghost.done && ghostPlan.length === 0) {
    const gp = ghostOf(cfg);
    if (gp !== 'bc') {
      const za = await baseline(gp);
      if (myEpoch !== epoch) return;
      ghostPlan = [await za.ownAction(ghost.obs(), ownGhost)];
    } else {
      const d = await guide.decide({ ...cfg, mode: 'unguided', N: 1 }, ghost.obs(), ghostHistory, rngGhost);
      if (myEpoch !== epoch || myLeg !== leg) { await yieldTask(); return; }
      ghostPlan = d.actions.slice(0, cfg.hexec);
    }
  }
  if (decision) { recentMs.push(performance.now() - t0); if (recentMs.length > 16) recentMs.shift(); }
  const L = 12 * env.dynamicGeoms.length, mainBatch = [], ghostBatch = cfg.ghost ? [] : null;
  const a = env.done ? null : plan.shift(), ga = cfg.ghost && !ghost.done ? ghostPlan.shift() : null;
  if (a) env.step(a, k => { if (k === undefined || (k + 1) % frameEvery === 0) mainBatch.push(env.poses(new Float64Array(L))); });
  if (ga) ghost.step(ga, k => { if (k === undefined || (k + 1) % frameEvery === 0) ghostBatch.push(ghost.poses(new Float64Array(L))); });
  const n = Math.max(mainBatch.length, ghostBatch ? ghostBatch.length : 0, 1);
  while (mainBatch.length < n) mainBatch.push(env.poses(new Float64Array(L)));
  while (ghostBatch && ghostBatch.length < n) ghostBatch.push(ghost.poses(new Float64Array(L)));
  if (a) pushHistory(history, env.obs());
  if (ga) pushHistory(ghostHistory, ghost.obs());
  sendFrame(mainBatch, ghostBatch);
  status(decision ? { scores: decision.scores, chosen: decision.chosen, N: decision.N, ms: decision.ms, mode: decision.mode } : undefined);
  const allDone = env.done && (!cfg.ghost || ghost.done);
  if (allDone) {
    post({ type: 'episode_end', main: runInfo(env), ghost: cfg.ghost ? runInfo(ghost) : null, task: env.taskIdx, knob: env.knobTask || null, cfg: { ...cfg } });
    playing = false; status();
  }
  while (hold && playing && myEpoch === epoch) await sleep(15);
  // pace to wall clock (speed 0 = as fast as possible, but yield so that messages such as pause get through)
  if (speed === 0) await yieldTask();
  if (speed > 0) {
    simStart += DT;
    const target = wallStart + 1000 * simStart / speed - leadMs(), now = performance.now();
    if (target > now) await sleep(target - now);
    else { if (now - target > 2000) { wallStart = now; simStart = 0; } await yieldTask(); }   // behind the clock: still yield
  }
}

async function loop() {
  if (busy) return; busy = true;
  try {
    while (playing || stepOnce) {
      stepOnce = false;
      await tick();
    }
  } catch (e) { playing = false; post({ type: 'error', msg: String(e && e.stack || e) }); }
  busy = false;
}

self.onmessage = async ev => {
  const m = ev.data;
  try {
    switch (m.type) {
      case 'init': await init(m.env, m.autoplay); break;
      case 'hold': hold = !!m.on; break;
      case 'config': {
        const prevG = cfg.ghost, prevGp = ghostOf(cfg), prevBc = cfg.actor, prevPol = cfg.policy; Object.assign(cfg, m.cfg);
        const gp = ghostOf(cfg);
        // a newly chosen gray policy starts from the green run's current state
        if (cfg.ghost && (!prevG || gp !== prevGp || (gp === 'bc' && cfg.actor !== prevBc))) { ghost.copyStateFrom(env); ghost.t = env.t; ghostPlan = []; ghostHistory = history.slice(); }
        plan = cfg.policy !== prevPol ? [] : plan.slice(0, cfg.hexec);
        break;
      }
      case 'task': await applyTask(m.k, m.restart !== undefined ? m.restart : env.ui.taskSwitch === 'restart'); break;
      case 'goal':          // a clicked goal: labels from the environment's goal rule
        env.setGoal(m.p); ghost.setGoal(m.p); env.taskIdx = null; ghost.taskIdx = null;
        post({ type: 'goal_info', task: null, marker: env.marker(), ...(await labels()) });
        await taskToBaselines();
        if (m.restart) { episodeSeed += 1; newEpisode(); sendPoses(); } else newLeg();
        break;
      case 'start': env.setStart(m.p); ghost.setStart(m.p); episodeSeed += 1; newEpisode(); sendPoses(); break;
      case 'reset': if (m.seed !== undefined) episodeSeed = m.seed; else if (m.newSeed !== false) episodeSeed += 1; newEpisode();
        sendPoses();
        if (m.play) { playing = true; loop(); }
        break;
      case 'play': if (env.done) { episodeSeed += 1; newEpisode(); } playing = true; wallStart = performance.now(); simStart = 0; loop(); break;
      case 'pause': playing = false; status(); break;
      case 'step': if (!env.done) { stepOnce = true; loop(); } break;
      case 'speed': speed = m.speed; wallStart = performance.now(); simStart = 0; break;
    }
  } catch (e) { post({ type: 'error', msg: String(e && e.stack || e) }); }
};
