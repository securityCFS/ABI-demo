import { SceneView } from './render.js';
import { MiniMap } from './minimap.js';

const $ = id => document.getElementById(id);
const NV = [2, 4, 8, 16, 32], AV = [0, 0.03, 0.1, 0.3, 1, 3];
const MODE_NOTE = {
  unguided: () => 'The policy alone: one sampled chunk per decision, no critic.',
  bfn: c => `The policy proposes ${c.N} chunks of 8 actions. ABI's critic scores each chunk for the current task and the best one is executed.`,
  qgf: c => `One chunk is sampled while the critic's action gradient pushes each flow step toward higher value (step size α = ${c.alpha}).`,
  qgf_bfn: c => `Four chunks are sampled with QGF and one without guidance, and the critic picks the best of the five (α = ${c.alpha}).`,
};
const MODE_LABEL = { unguided: 'unguided', bfn: 'best-of-N', qgf: 'QGF', qgf_bfn: 'QGF, best of 4+1', actor_bfn: 'best-of-16 actions' };
// ABI guiding a zero-shot RL actor (paper, App. "Guiding external trained actors").
const ACTOR_NOTE = n => `The ${n} proposes 16 actions: its own action and 15 further samples. The frozen Flow BC completes each into 8-step chunks (4 completions shared by all actions), ABI's critic scores them, and the best action is executed. The policy replans at every step.`;

const params = new URLSearchParams(location.search);
const ENV_NAME = params.get('env') || 'cube-single-play';
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let view, map = null, envMeta, features, tasks, models, cfg, speed = 1, playing = false, done = false;
const stats = { starved: 0, ticks: 0 };
let queue = [], cursor = null, lastNow = 0, pendingAuto = null, frameLen = 0;
const tally = new Map();
let values = [], taskIdx = 0, knobTask = null, alphaTask = 0, baselines = [], ghostWarn = {}, guidedMode = 'bfn', defaultBc = null;
// Video capture (tests/record_video.mjs): the page renders a fixed amount of simulated time per frame on request.
const CAPTURE = params.has('capture'), CAP_FPS = 30, CAP_SPEED = +(params.get('cspeed') || 1);
// capture: poses the simulation may run ahead of the recorded frame. A short queue (?cq=6) makes a click or a task
// switch act at the recorded frame instead of after the queued poses.
const CAP_Q = Math.max(4, +(params.get('cq') || 150)), CAP_Q_LOW = Math.min(60, Math.max(2, Math.round(CAP_Q * 0.4)));
let capFrame = 0, capBanner = null, held = false;

worker.onmessage = ev => {
  const m = ev.data;
  if (m.type === 'progress') { $('loading-msg').textContent = m.msg; if (view && !CAPTURE) banner(m.msg, '', 4000); }
  else if (m.type === 'ready') onReady(m);
  else if (m.type === 'frames') onFrames(m);
  else if (m.type === 'status') onStatus(m);
  else if (m.type === 'episode_start') onEpisodeStart(m);
  else if (m.type === 'leg_start') onLegStart(m);
  else if (m.type === 'episode_end') later(() => onEpisodeEnd(m));
  else if (m.type === 'goal_info') onGoalInfo(m);
  else if (m.type === 'ghost_info') onGhostInfo(m);
  else if (m.type === 'error') { console.error(m.msg); banner('Error: ' + m.msg.split('\n')[0], 'fail'); $('loading-msg').textContent = m.msg; }
};
worker.postMessage({ type: 'init', env: ENV_NAME, autoplay: !CAPTURE && !params.has('paused') });
loadRegistry();

// Environment menu: switching reloads the page for the new environment.
async function loadRegistry() {
  try {
    const reg = await (await fetch('envs.json')).json(), sel = $('env');
    const groups = {};
    for (const e of reg) {
      if (!groups[e.group]) { groups[e.group] = document.createElement('optgroup'); groups[e.group].label = e.group; sel.appendChild(groups[e.group]); }
      const o = document.createElement('option'); o.value = e.name; o.textContent = e.label; groups[e.group].appendChild(o);
    }
    sel.value = ENV_NAME;
    const cur = reg.find(e => e.name === ENV_NAME);
    if (cur) { $('title-env').textContent = cur.title || cur.label; document.title = `ABI playground · ${cur.label}`; }
    sel.onchange = () => { const p = new URLSearchParams(location.search); p.set('env', sel.value); location.search = p.toString(); };
  } catch (e) { $('env').hidden = true; }
}

function onReady(m) {
  envMeta = m.envMeta; features = m.features || {}; tasks = m.tasks; models = m.models; cfg = m.cfg;
  cfg.ghost = true;                                     // the comparison run (gray) is always on
  if (cfg.mode !== 'unguided') guidedMode = cfg.mode;
  view = new SceneView($('scene'), m.desc, envMeta);
  view.setGhostVisible(true);
  frameLen = 12 * m.desc.dynamic.length;
  if (features.map) map = new MiniMap($('map'), envMeta, m.refXY, xy => setGoal(xy), xy => setStart(xy));
  else { $('map').hidden = true; $('refs-row').hidden = true; }
  $('restart-row').hidden = !features.clickGoal && features.taskSwitch !== 'continue';
  $('restart-label').textContent = features.clickGoal ? 'restart from the start when the goal moves' : 'restart the episode when the task changes';
  $('goal-title').textContent = features.clickGoal ? 'Goal' : 'Task';
  const hints = {
    floor: ['Or click the floor or the map to place your own goal. Shift+click (right-click on the map) sets the start.',
            'Pick a task on the right, or click the floor to place a goal.'],
    table: ['Or click inside the marked area on the table to place your own target for the cube.', 'Pick a task on the right, or click inside the marked area on the table to place a target for the cube.'],
  }[features.clickGoal] || ['Switch the task at any time, also while the agent moves. ABI only reads the new reward labels, nothing is retrained.',
                            'Pick a task on the right. You can switch it while the agent moves.'];
  $('task-hint').textContent = hints[0]; $('intro-1').textContent = hints[1];
  $('hint').textContent = features.clickGoal === 'table' ? 'Click the marked area to set the target. Drag to orbit.' : 'Drag to orbit.';
  window.__demo = { map, view, worker, stats };   // for automated checks
  showMarker(m.marker);
  tasks.forEach((t, k) => {
    const b = document.createElement('button'); b.textContent = t.label; if (t.title) b.title = t.title;
    b.onclick = () => {
      clearTimeout(pendingAuto); taskIdx = k; alphaTask = k; applyTaskAlpha();
      const restart = features.taskSwitch === 'restart' || $('restart-on-goal').checked;
      worker.postMessage({ type: 'task', k, restart });
    };
    $('tasks').appendChild(b);
  });
  buildKnobs();
  baselines = m.baselines || []; defaultBc = cfg.actor; cfg.policy = cfg.policy || cfg.actor; cfg.ghostPolicy = 'same';
  // green: the policy ABI guides (behavior cloning, or a zero-shot RL actor where one is exported)
  const group = (label, items) => {
    const g = document.createElement('optgroup'); g.label = label;
    for (const [v, t] of items) { const o = document.createElement('option'); o.value = v; o.textContent = t; g.appendChild(o); }
    $('policy').appendChild(g);
  };
  group('Behavior cloning', models.actors.map(a => [a.id, a.label]));
  const guided = baselines.filter(b => (models.guideActors || []).includes(b.id));
  if (guided.length) group('Zero-shot RL actor', guided.map(b => [b.id, b.label]));
  $('policy').value = cfg.policy;
  // gray (advanced): the same policy without ABI by default, or another policy
  const bc = models.actors.find(a => a.id === defaultBc);
  for (const [v, t] of [['base', `${bc ? bc.short : 'Flow BC'}, without ABI`], ...baselines.map(b => [b.id, b.short || b.label])]) {
    const o = document.createElement('option'); o.value = v; o.textContent = t; $('ghost-policy').appendChild(o);
  }
  $('ghost-block').hidden = !baselines.length;          // without RL actors the gray run is always the same policy
  if (!models.qgf) for (const b of document.querySelectorAll('#mode button')) if (b.dataset.mode.startsWith('qgf')) { b.disabled = true; b.title = 'QGF needs the gradient graph, which is not part of this build.'; }
  $('n').max = NV.length - 1; $('n').min = 0; $('n').value = NV.indexOf(cfg.N) >= 0 ? NV.indexOf(cfg.N) : 3;
  $('alpha').max = AV.length - 1; $('alpha').min = 0; $('alpha').value = Math.max(0, AV.indexOf(cfg.alpha));
  $('foot').innerHTML = models.footer;
  const who = envMeta.agent_name || 'agent';
  $('follow-label').textContent = `camera follows the ${who}`; $('follow').checked = view.follow;
  $('follow').closest('label').style.display = !features.map && !(envMeta.camera && envMeta.camera.follow) ? 'none' : '';
  $('tally-head').textContent = features.successTask ? 'reached' : 'mean return';
  $('tally-head2').textContent = features.successTask ? 'mean steps' : 'episodes';
  bindUI(); syncUI(); sendCfg();
  $('loading').classList.add('hidden');
  if (CAPTURE) setupCapture(); else { requestAnimationFrame(frame); showIntro(); }
  view.canvas.addEventListener('pointerdown', ev => { view._down = [ev.clientX, ev.clientY]; });
  view.canvas.addEventListener('pointerup', ev => {
    if (!features.clickGoal || !view._down || Math.hypot(ev.clientX - view._down[0], ev.clientY - view._down[1]) > 8 || ev.button !== 0) return;
    const p = view.pickPoint(ev, features.clickGoal); if (!p) return;
    if (ev.shiftKey && features.clickStart) setStart(p); else setGoal(p);
  });
  view.canvas.addEventListener('pointermove', ev => {
    if (features.clickGoal !== 'table') return;
    view.canvas.style.cursor = view.hover(ev.buttons ? null : ev, 'table') ? 'crosshair' : '';
  });
  view.canvas.addEventListener('pointerleave', () => { view.hover(null); view.canvas.style.cursor = ''; });
}

function free(p) {
  if (!envMeta.maze_map) return true;
  const [x, y] = p, u = envMeta.maze_unit, mm = envMeta.maze_map;
  const i = Math.floor((y + envMeta.offset_y + u / 2) / u), j = Math.floor((x + envMeta.offset_x + u / 2) / u);
  return i >= 0 && i < mm.length && j >= 0 && j < mm[0].length && mm[i][j] === 0;
}
function showMarker(mk) {
  if (!mk) return;
  view.setGoal(mk.goal); if (mk.start) view.setStart(mk.start);
  if (map) { map.goal = mk.goal; if (mk.start) map.start = mk.start; map.draw(); }
}
function setGoal(p) {
  if (!free(p)) { banner('That point is inside a wall', 'fail'); return; }
  taskIdx = null; markTask();
  worker.postMessage({ type: 'goal', p, restart: $('restart-on-goal').checked });
}
function setStart(p) {
  if (!free(p)) { banner('That point is inside a wall', 'fail'); return; }
  taskIdx = null; markTask();
  worker.postMessage({ type: 'start', p });
}
function markTask() {
  document.querySelectorAll('#tasks button').forEach((b, k) => b.classList.toggle('on', k === taskIdx));
  // knobs follow the active preset (or the custom value), the slider of the active family is highlighted
  const t = taskIdx !== null && taskIdx !== undefined ? envMeta.tasks[taskIdx] : knobTask;
  for (const kn of features.knobs || []) {
    const row = $('knob-' + kn.id); if (!row) continue;
    const on = !!t && t.family === kn.family && t.params && t.params[kn.param] !== undefined;
    row.classList.toggle('on', on);
    if (on) { row.querySelector('input').value = t.params[kn.param]; row.querySelector('b').textContent = fmtKnob(kn, t.params[kn.param]); }
  }
}
const fmtKnob = (kn, v) => `${(+v).toFixed(kn.step < 0.1 ? 2 : kn.step < 1 ? 1 : 0)} ${kn.unit.replace('^2', '²')}`;
// A slider of a reward family: any value is a new task, the critic only reads new reward labels.
function buildKnobs() {
  for (const kn of features.knobs || []) {
    const row = document.createElement('div'); row.className = 'slider'; row.id = 'knob-' + kn.id; row.title = kn.note || '';
    row.innerHTML = `<span>${kn.name}</span><input type="range" min="${kn.min}" max="${kn.max}" step="${kn.step}"><b></b>`;
    const inp = row.querySelector('input');
    const preset = envMeta.tasks.find(t => t.family === kn.family && t.params[kn.param] !== undefined);
    inp.value = preset ? preset.params[kn.param] : kn.min; row.querySelector('b').textContent = fmtKnob(kn, inp.value);
    inp.oninput = () => { row.querySelector('b').textContent = fmtKnob(kn, inp.value); };
    inp.onchange = () => {
      clearTimeout(pendingAuto);
      knobTask = { family: kn.family, params: { [kn.param]: +inp.value } }; taskIdx = null;
      worker.postMessage({ type: 'task', k: knobTask, restart: $('restart-on-goal').checked });
    };
    $('knobs').appendChild(row);
  }
}

function bindUI() {
  $('policy').onchange = () => {
    cfg.policy = $('policy').value; cfg.actor = isActor(cfg.policy) ? defaultBc : cfg.policy;
    if (isActor(cfg.policy)) cfg.hexec = 1;                  // an RL actor outputs one action per step
    syncUI(); sendCfg();
  };
  for (const b of document.querySelectorAll('#mode button')) b.onclick = () => { guidedMode = cfg.mode = b.dataset.mode; syncUI(); sendCfg(); applyTaskAlpha(); };
  for (const b of document.querySelectorAll('#guide-toggle button')) b.onclick = () => { cfg.mode = b.dataset.g === 'on' ? guidedMode : 'unguided'; syncUI(); sendCfg(); applyTaskAlpha(); };
  for (const b of document.querySelectorAll('#hexec button')) b.onclick = () => { cfg.hexec = +b.dataset.h; syncUI(); sendCfg(); };
  $('n').oninput = () => { cfg.N = NV[+$('n').value]; syncUI(); sendCfg(); };
  $('alpha').oninput = () => { cfg.alpha = AV[+$('alpha').value]; syncUI(); sendCfg(); };
  $('ghost-policy').onchange = () => { cfg.ghostPolicy = $('ghost-policy').value; syncUI(); sendCfg(); };
  $('refs').onchange = () => { if (map) { map.showRefs = $('refs').checked; map.draw(); } };
  $('follow').onchange = () => { view.follow = $('follow').checked; };
  $('play').onclick = () => { if (playing) worker.postMessage({ type: 'pause' }); else { clearTimeout(pendingAuto); worker.postMessage({ type: 'play' }); } };
  $('step').onclick = () => worker.postMessage({ type: 'step' });
  $('reset').onclick = () => { clearTimeout(pendingAuto); worker.postMessage({ type: 'reset', play: false }); };
  $('speed').onchange = () => { speed = +$('speed').value; worker.postMessage({ type: 'speed', speed }); };
  window.addEventListener('keydown', ev => { if (ev.code === 'Space' && ev.target.tagName !== 'SELECT') { ev.preventDefault(); $('play').click(); } });
}

const isActor = id => baselines.some(b => b.id === id);
// Display names: the guided policy, and the gray run (whose policy is that of ghostOf, as in the worker).
function policyName(c = cfg) {
  const a = models.actors.find(x => x.id === c.policy) || baselines.find(x => x.id === c.policy);
  return a ? a.short || a.label : c.policy;
}
function ghostOf(c = cfg) { const g = c.ghostPolicy || 'same'; return g === 'same' ? (isActor(c.policy) ? c.policy : 'bc') : g === 'base' ? 'bc' : g; }
function ghostName(c = cfg) {
  const g = ghostOf(c);
  if (g === 'bc') { const a = models.actors.find(x => x.id === c.actor); return `${a ? a.short : 'Flow BC'}, without ABI`; }
  const b = baselines.find(x => x.id === g); return `${b ? b.short || b.label : g}${g === c.policy ? ', without ABI' : ''}`;
}
function guidedName(c = cfg) { return c.mode === 'unguided' ? `${policyName(c)}, without ABI` : `${policyName(c)} + ABI`; }

function syncUI() {
  const on = cfg.mode !== 'unguided', ext = isActor(cfg.policy);
  for (const b of document.querySelectorAll('#guide-toggle button')) b.classList.toggle('on', (b.dataset.g === 'on') === on);
  for (const b of document.querySelectorAll('#mode button')) {
    b.classList.toggle('on', b.dataset.mode === (ext ? 'bfn' : guidedMode));
    b.disabled = (ext && b.dataset.mode !== 'bfn') || (!models.qgf && b.dataset.mode.startsWith('qgf'));
  }
  for (const b of document.querySelectorAll('#hexec button')) { b.classList.toggle('on', +b.dataset.h === cfg.hexec); b.disabled = ext && +b.dataset.h !== 1; }
  $('n-val').textContent = ext ? 16 : guidedMode === 'qgf_bfn' ? '4+1' : guidedMode === 'bfn' ? cfg.N : 1;
  $('n-row').classList.toggle('off', ext || guidedMode !== 'bfn');
  $('a-val').textContent = cfg.alpha; $('a-row').classList.toggle('off', ext || !guidedMode.startsWith('qgf'));
  const same = ghostOf() === (ext ? cfg.policy : 'bc');
  $('mode-note').textContent = (ext ? (on ? ACTOR_NOTE(policyName()) : `The ${policyName()} alone: its own action at every step, no critic.`) : MODE_NOTE[cfg.mode](cfg))
    + (on || !same ? '' : ' The gray run is the same policy with the same noise, so from a new episode on the two runs overlap.');
  $('method-note').textContent = ext ? 'ABI guides an RL actor by best-of-16 selection, as in the paper.'
    : guidedMode.startsWith('qgf') && models.taskAlpha ? 'On the paper\'s tasks α is set to the value selected in the paper.' : '';
  $('hud-mk').textContent = guidedName();
  document.body.classList.toggle('unguided', !on); view.setGuided(on); if (map) map.col = null;   // gray without ABI
  const a = models.actors.find(x => x.id === cfg.policy) || baselines.find(x => x.id === cfg.policy);
  $('policy-note').textContent = a ? (a.note || '') + (ghostWarn[cfg.policy] ? ' ' + ghostWarn[cfg.policy] : '') : '';
  const gb = baselines.find(x => x.id === ghostOf());
  $('ghost-note').textContent = gb && gb.id !== cfg.policy ? (gb.note || '') + (ghostWarn[gb.id] ? ' ' + ghostWarn[gb.id] : '') : 'The gray run starts from the same state as the green run.';
  $('scores-note').textContent = ext ? `The 16 actions of the ${policyName()} at the last decision, each scored as the mean over its 4 completed chunks. The highlighted one is executed.`
    : 'The N proposals of the policy at the last decision. The highlighted one is executed.';
  $('hud-gk').textContent = ghostName();
  markTask();
}
// QGF strengths selected per task on development episodes in the paper; a clicked goal keeps the slider value.
function applyTaskAlpha() {
  const ta = models.taskAlpha; if (!ta || alphaTask === null || !cfg.mode.startsWith('qgf')) return;
  const a = (ta[cfg.mode === 'qgf' ? 'qgf' : 'qgf_bfn'] || [])[alphaTask];
  if (a !== undefined && a !== null) {
    cfg.alpha = a;
    if (AV.indexOf(a) < 0) { AV.push(a); AV.sort((x, y) => x - y); $('alpha').max = AV.length - 1; }
    $('alpha').value = AV.indexOf(a); syncUI(); sendCfg();
  }
}
function sendCfg() { worker.postMessage({ type: 'config', cfg: { ...cfg } }); }

function onFrames(m) {
  for (let k = 0; k < m.n; k++) queue.push({ dt: m.dt, main: m.main.subarray(k * frameLen, (k + 1) * frameLen), ghost: m.ghost ? m.ghost.subarray(k * frameLen, (k + 1) * frameLen) : null });
  // When the display falls behind, skip poses but keep every trail point, so trails stay continuous.
  if (CAPTURE) { if (queue.length > CAP_Q && !held) { held = true; worker.postMessage({ type: 'hold', on: true }); } }
  else if (queue.length > 1500) while (queue.length > 200) skim(queue.shift());
}
// The worker runs ahead of the display. Messages about a step are queued behind that step's frames, so the
// HUD, scores and banners change when the agent on screen gets there.
function later(fn) { if (queue.length) queue.push({ ev: fn }); else fn(); }
function skim(f) {
  if (f.ev) { f.ev(); return; }
  view.pushTrail(view.trail, ...view.trackPoint(f.main));
  if (f.ghost) view.pushTrail(view.ghostTrail, ...view.trackPoint(f.ghost));
  if (map) map.track([f.main[0], f.main[1]], f.ghost ? [f.ghost[0], f.ghost[1]] : null);
}
function apply(f) {
  view.applyPoses(f.main, false);
  if (f.ghost) view.applyPoses(f.ghost, true);
  if (map) map.update([f.main[0], f.main[1]], f.ghost ? [f.ghost[0], f.ghost[1]] : null);
}
function frame(now) {
  if (playing && speed > 0) { stats.ticks++; if (!queue.length) stats.starved++; }
  if (queue.length) {
    if (speed === 0 || cursor === null) { cursor = 0; lastNow = now; }
    if (speed === 0) { while (queue.length > 1) skim(queue.shift()); const f = queue.shift(); f.ev ? f.ev() : apply(f); }
    else {
      cursor += (now - lastNow) / 1000 * speed; lastNow = now;    // in seconds of simulated time
      while (queue.length && (queue[0].ev || cursor >= queue[0].dt)) { const f = queue.shift(); if (f.ev) f.ev(); else { apply(f); cursor -= f.dt; } }
      if (!queue.length) cursor = null;
    }
  } else cursor = null;
  view.render();
  requestAnimationFrame(frame);
}

function onStatus(m) {
  playing = m.playing; done = m.done;
  $('play').textContent = playing ? 'Pause' : 'Play';
  later(() => showStatus(m));
}
function hudText(r) { return r.hud.filter(([k]) => k !== 'step' && k !== 'time').map(([k, v]) => `${k} ${v}`).join(' · '); }
function showStatus(m) {
  if (CAPTURE && window.__cap) window.__cap.status = { t: m.t, done: m.done, success: m.success, ret: m.ret, ghost: m.ghost && { t: m.ghost.t, done: m.ghost.done, success: m.ghost.success, ret: m.ghost.ret } };
  $('hud-t').textContent = m.t;
  $('hud-lines').textContent = hudText(m);
  if (m.ghost) $('hud-g').textContent = hudText(m.ghost);
  if (m.decision) {
    $('hud-ms').textContent = `Last decision took ${m.decision.ms.toFixed(0)} ms (${MODE_LABEL[m.decision.mode]}, N=${m.decision.N}).`; drawScores(m.decision);
    if (m.decision.scores) { values.push([m.t, m.decision.scores[m.decision.chosen]]); drawValues(); }
  }
}

function onEpisodeStart(m) {
  for (const f of queue) if (f.ev) f.ev();   // frames still queued belong to the previous episode: drop them,
  queue.length = 0; cursor = null;           // but keep its results (tally)
  values = []; drawValues();
  view.clearTrails(); if (map) map.clearTrails(); showMarker(m.marker); done = false; hideBanner();
  if (m.task !== undefined) { taskIdx = m.task; markTask(); }
}
function onLegStart(m) {
  later(() => {
    showMarker(m.marker); done = false; values = []; drawValues();
    banner(features.clickGoal ? 'New goal: continuing from the current position' : `Task switched to ${taskName(m.task, m.knob)}: continuing from the current state`, '');
  });
}

function taskName(task, knob) {
  if (task !== null && task !== undefined) return tasks[task].label;
  if (knob) { const kn = (features.knobs || []).find(x => x.family === knob.family); if (kn) return `${kn.name} ${fmtKnob(kn, knob.params[kn.param])}`; }
  return 'custom';
}
function key(c) {
  const tn = taskName(c.task, c.knob), t = tn !== 'custom' ? `${tn} · ` : '';
  if (isActor(c.policy)) return `${t}${policyName(c)} · ${c.mode === 'unguided' ? 'without ABI' : '+ ABI, best-of-16'}`;
  return `${t}${policyName(c)} · ${MODE_LABEL[c.mode]}${c.mode === 'bfn' ? ' N=' + c.N : ''}${c.mode.startsWith('qgf') ? ' α=' + c.alpha : ''} · H<sub>exec</sub>=${c.hexec}`;
}
function ghostKey(c) {
  const g = ghostOf(c), tn = taskName(c.task, c.knob), t = tn !== 'custom' ? `${tn} · ` : '';
  if (g === 'bc') return key({ ...c, policy: c.actor, mode: 'unguided' }) + ' (gray)';
  return `${t}${ghostName(c)} (gray)`;
}
function record(k, r) {
  const s = tally.get(k) || { n: 0, s: 0, steps: 0, ret: 0 }; s.n++; s.ret += r.ret; if (r.success) { s.s++; s.steps += r.t; } tally.set(k, s);
}
function onEpisodeEnd(m) {
  if (CAPTURE && window.__cap) window.__cap.ended = { main: m.main, ghost: m.ghost, capFrame };
  const sec = s => (s * (envMeta.control_dt || envMeta.control_timestep || 0.1)).toFixed(1);
  let msg;
  if (features.successTask) {
    msg = m.main.success ? `Reached the goal in ${m.main.t} steps (${sec(m.main.t)} s)` : `Goal not reached within ${m.main.t} steps`;
    if (m.ghost) msg += m.ghost.success ? ` · ${ghostName(m.cfg)}: reached in ${m.ghost.t}` : ` · ${ghostName(m.cfg)}: not reached`;
  } else {
    msg = `Episode return ${m.main.ret.toFixed(1)}`;
    if (m.ghost) msg += ` · ${ghostName(m.cfg)}: ${m.ghost.ret.toFixed(1)}`;
  }
  banner(msg, features.successTask ? (m.main.success ? 'ok' : 'fail') : 'ok', 6000);
  const c = { ...m.cfg, task: m.task, knob: m.knob };
  record(key(c), m.main);
  if (m.ghost) record(ghostKey(c), m.ghost);
  drawTally();
  if ($('auto').checked) pendingAuto = setTimeout(() => worker.postMessage({ type: 'reset', play: true }), 1500);
}
function drawTally() {
  const tb = document.querySelector('#tally tbody'); tb.innerHTML = '';
  $('tally-empty').hidden = tally.size > 0;
  for (const [k, r] of tally) {
    const tr = document.createElement('tr');
    tr.innerHTML = features.successTask
      ? `<td>${k}</td><td class="num">${r.s}/${r.n}</td><td class="num">${r.s ? (r.steps / r.s).toFixed(0) : '–'}</td>`
      : `<td>${k}</td><td class="num">${(r.ret / r.n).toFixed(1)}</td><td class="num">${r.n}</td>`;
    tb.appendChild(tr);
  }
}

// A comparison actor infers its task from its own inference samples; with no sample at the goal it has no task.
function onGhostInfo(m) {
  ghostWarn[m.id] = m.valid ? '' : `None of its ${m.total} inference samples reaches this goal, so its task inference gives no task (z = 0).`;
  syncUI();
}
function onGoalInfo(m) {
  if (map && m.mask) map.setLabelled(m.mask);
  showMarker(m.marker);
  if (m.task !== null && m.task !== undefined) { taskIdx = m.task; knobTask = null; markTask(); }
  else if (m.knob) { taskIdx = null; knobTask = m.knob; markTask(); }
  if (m.settingTask !== null && m.settingTask !== undefined && m.settingTask !== alphaTask) { alphaTask = m.settingTask; applyTaskAlpha(); }
  const what = features.clickGoal ? 'goal' : 'task';
  if (m.meanLabel !== undefined && !features.clickGoal) {
    $('goal-note').textContent = `The task sets the reward label of each of the ${m.total} reference states (mean label ${m.meanLabel.toFixed(3)}). ABI reads the task from them, nothing is retrained.`;
    return;
  }
  $('goal-note').textContent = m.count > 0
    ? `${m.count} of ${m.total} reference states carry a nonzero reward label for this ${what}. ABI reads the task from them, nothing is retrained.`
    : 'No reference state lies within the goal radius here, so the critic has no signal for this goal. Move it closer to the dots.';
}

let bannerTimer = null;
function banner(text, cls, ms = 3500) {
  if (CAPTURE) { capBanner = { text, cls, until: capFrame + Math.round(ms / 1000 * CAP_FPS) }; return; }
  const b = $('banner'); b.textContent = text; b.className = cls || ''; b.hidden = false;
  clearTimeout(bannerTimer); bannerTimer = setTimeout(hideBanner, ms);
}
function hideBanner() { $('banner').hidden = true; }

function drawScores(d) {
  const c = $('scores'), dpr = Math.min(window.devicePixelRatio || 1, 2), w = c.clientWidth;
  c.width = w * dpr; c.height = 90 * dpr; const g = c.getContext('2d'); g.scale(dpr, dpr);
  g.clearRect(0, 0, w, 90);
  const s = Array.from(d.scores || []); if (!s.length) return;
  const lo = Math.min(...s), hi = Math.max(...s), span = hi - lo || 1, n = s.length, bw = Math.max(2, (w - 4) / n - 2);
  s.forEach((v, k) => {
    const h = 8 + 70 * (v - lo) / span, x = 2 + k * (bw + 2);
    g.fillStyle = k === d.chosen ? '#76b900' : '#4a5568'; g.fillRect(x, 86 - h, bw, h);
  });
  g.fillStyle = '#98a2b3'; g.font = '11px system-ui'; g.fillText(`min ${lo.toPrecision(3)}  max ${hi.toPrecision(3)}`, 4, 11);
}

function drawValues() {
  const c = $('value'), dpr = Math.min(window.devicePixelRatio || 1, 2), w = c.clientWidth, H = 70;
  c.width = w * dpr; c.height = H * dpr; const g = c.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, w, H);
  g.strokeStyle = '#2a3240'; g.strokeRect(0.5, 0.5, w - 1, H - 1);
  if (values.length < 2) return;
  const vs = values.map(v => v[1]), lo = Math.min(...vs), hi = Math.max(...vs), span = hi - lo || 1;
  const t0 = values[0][0], t1 = Math.max(values[values.length - 1][0], t0 + 50);
  g.strokeStyle = '#76b900'; g.lineWidth = 1.5; g.beginPath();
  values.forEach(([t, v], k) => { const x = 4 + (w - 8) * (t - t0) / (t1 - t0), y = H - 6 - (H - 18) * (v - lo) / span; k ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.stroke();
  g.fillStyle = '#98a2b3'; g.font = '11px system-ui'; g.fillText(`${lo.toPrecision(3)} … ${hi.toPrecision(3)}`, 6, 12);
}

function showIntro() {
  let seen = false; try { seen = localStorage.getItem('abi-intro-seen') === '1'; } catch (e) { /* storage blocked */ }
  if (seen) return;
  $('intro').hidden = false;
  $('intro-ok').onclick = () => { $('intro').hidden = true; try { localStorage.setItem('abi-intro-seen', '1'); } catch (e) { /* ignore */ } };
}

// Video capture: each call of __cap.frame() shows the next 1/30 s × cspeed of simulated time and returns the
// frame (canvas plus caption, legend and banner) as a JPEG data URL, or null while the worker is behind.
function setupCapture() {
  // ?capture&ui keeps the page's UI (the recorder takes page screenshots), otherwise only the 3-D view is captured
  if (!params.has('ui')) document.body.classList.add('capture');
  if (view.clickArea) view.clickArea.visible = false;      // the clickable-area marking is a UI aid, not part of the scene
  speed = 0; worker.postMessage({ type: 'speed', speed: 0 });
  const out = document.createElement('canvas'), g = out.getContext('2d');
  const font = (px, w = 400) => `${w} ${px}px "DejaVu Sans", system-ui, sans-serif`;
  function pill(x, y, lines, s) {
    let w = 0, h = 14 * s;
    for (const [t, px, wt] of lines) { g.font = font(px * s, wt); w = Math.max(w, g.measureText(t).width); h += px * s * 1.35; }
    g.fillStyle = 'rgba(15,18,24,0.78)'; g.beginPath(); g.roundRect(x, y, w + 28 * s, h, 10 * s); g.fill();
    let yy = y + 7 * s;
    for (const [t, px, wt, col] of lines) { g.font = font(px * s, wt); g.fillStyle = col || '#e6e9ef'; g.textBaseline = 'top'; g.fillText(t, x + 14 * s, yy + px * s * 0.12); yy += px * s * 1.35; }
  }
  window.__cap = {
    caption: null, legend: null, ended: null, clean: false, ui: params.has('ui'),   // clean: no overlays on the frame
    post: m => worker.postMessage(m), banner: (t, c, ms) => banner(t, c, ms), status: null,
    state: () => ({ queue: queue.length, playing, held, capFrame }),
    frame() {
      const need = CAP_SPEED / CAP_FPS;
      let have = 0; for (const f of queue) if (!f.ev) { have += f.dt; if (have >= need) break; }
      if (have < need - 1e-9 && playing) {     // not enough poses yet: let the simulation run (a short ?cq queue can be
        if (held) { held = false; worker.postMessage({ type: 'hold', on: false }); }   // held with too few poses for a frame)
        return null;
      }
      cursor = (cursor || 0) + need;
      while (queue.length && (queue[0].ev || cursor >= queue[0].dt - 1e-9)) { const f = queue.shift(); if (f.ev) f.ev(); else { apply(f); cursor -= f.dt; } }
      if (!queue.length) cursor = 0;
      if (held && queue.length < CAP_Q_LOW) { held = false; worker.postMessage({ type: 'hold', on: false }); }
      view.render(); capFrame++;
      if (window.__cap.ui) return true;
      const W = view.canvas.width, H = view.canvas.height, s = H / 720;
      if (out.width !== W || out.height !== H) { out.width = W; out.height = H; }
      g.drawImage(view.canvas, 0, 0);
      if (window.__cap.clean) return out.toDataURL('image/jpeg', 0.92);
      const c = window.__cap.caption;
      if (c) pill(24 * s, 22 * s, [[c.title, 28, 700], ...(c.sub ? [[c.sub, 19, 400, '#c5ccd8']] : [])], s);
      const lg = window.__cap.legend || [guidedName(), ghostName()];
      g.font = font(19 * s, 600);
      const lw = Math.max(g.measureText(lg[0]).width, g.measureText(lg[1]).width) + 62 * s, lh = 72 * s, lx = 24 * s, ly = H - lh - 22 * s;
      g.fillStyle = 'rgba(15,18,24,0.78)'; g.beginPath(); g.roundRect(lx, ly, lw, lh, 10 * s); g.fill();
      [[lg[0], cfg.mode === 'unguided' ? '#a9b0a6' : '#76b900'], [lg[1], '#a9b0a6']].forEach(([t, col], k) => {
        const y = ly + 14 * s + k * 26 * s;
        g.fillStyle = col; g.beginPath(); g.arc(lx + 24 * s, y + 10 * s, 8 * s, 0, 2 * Math.PI); g.fill();
        g.fillStyle = '#e6e9ef'; g.textBaseline = 'top'; g.fillText(t, lx + 42 * s, y);
      });
      if (CAP_SPEED !== 1) { g.font = font(18 * s, 600); const t = `${CAP_SPEED}× speed`, tw = g.measureText(t).width;
        g.fillStyle = 'rgba(15,18,24,0.7)'; g.beginPath(); g.roundRect(W - tw - 52 * s, 22 * s, tw + 28 * s, 34 * s, 8 * s); g.fill();
        g.fillStyle = '#c5ccd8'; g.fillText(t, W - tw - 38 * s, 30 * s); }
      if (capBanner && capFrame <= capBanner.until) {
        g.font = font(22 * s, 600); const tw = g.measureText(capBanner.text).width, bx = (W - tw) / 2 - 18 * s, by = ly - 58 * s;
        g.fillStyle = 'rgba(22,27,35,0.92)'; g.strokeStyle = capBanner.cls === 'ok' ? '#76b900' : capBanner.cls === 'fail' ? '#ef4f63' : '#2a3240';
        g.lineWidth = 2 * s; g.beginPath(); g.roundRect(bx, by, tw + 36 * s, 44 * s, 9 * s); g.fill(); g.stroke();
        g.fillStyle = capBanner.cls === 'ok' ? '#b9f0cf' : capBanner.cls === 'fail' ? '#ffc2ca' : '#e6e9ef'; g.fillText(capBanner.text, bx + 18 * s, by + 10 * s);
      }
      return out.toDataURL('image/jpeg', 0.92);
    },
  };
}
