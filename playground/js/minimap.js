// Top-down map of the maze: walls, reference states that carry the goal's reward label, start,
// goal, both agents and their trails. Click sets the goal, shift+click sets the start.
export class MiniMap {
  constructor(canvas, meta, refXY, onGoal, onStart) {
    this.c = canvas; this.g = canvas.getContext('2d'); this.meta = meta; this.refXY = refXY;
    this.goal = null; this.start = null; this.ant = null; this.ghostXY = null; this.trail = []; this.ghostTrail = [];
    this.labelled = null; this.showRefs = true; this.bg = document.createElement('canvas'); this.bgKey = ''; this.raf = 0;
    const mm = meta.maze_map, u = meta.maze_unit;
    this.x0 = -meta.offset_x - u / 2; this.y0 = -meta.offset_y - u / 2;
    this.w = mm[0].length * u; this.h = mm.length * u;
    const pick = ev => {
      const r = canvas.getBoundingClientRect();
      const px = (ev.clientX - r.left) / r.width, py = (ev.clientY - r.top) / r.height;
      return [this.x0 + px * this.w, this.y0 + (1 - py) * this.h];
    };
    canvas.addEventListener('click', ev => { const p = pick(ev); (ev.shiftKey ? onStart : onGoal)(p); });
    canvas.addEventListener('contextmenu', ev => { ev.preventDefault(); onStart(pick(ev)); });
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }
  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2), r = this.c.getBoundingClientRect();
    if (!r.width) return;
    this.c.width = Math.round(r.width * dpr); this.c.height = Math.round(r.width * dpr * this.h / this.w);
    this.c.style.height = `${r.width * this.h / this.w}px`;
    this.draw();
  }
  toPx(x, y) { return [(x - this.x0) / this.w * this.c.width, (1 - (y - this.y0) / this.h) * this.c.height]; }
  css(name) { return getComputedStyle(this.c).getPropertyValue(name).trim(); }
  setLabelled(mask) { this.labelled = mask; this.labelVer = (this.labelVer || 0) + 1; this.draw(); }
  clearTrails() { this.trail = []; this.ghostTrail = []; }
  // Poses arrive many times per display frame: redraw at most once per animation frame.
  update(ant, ghostXY) {
    this.track(ant, ghostXY);
    if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.draw(); });
  }
  track(ant, ghostXY) {
    if (ant) { this.ant = ant; const l = this.trail[this.trail.length - 1]; if (!l || Math.hypot(l[0] - ant[0], l[1] - ant[1]) > 0.05) this.trail.push(ant); }
    this.ghostXY = ghostXY;
    if (ghostXY) { const l = this.ghostTrail[this.ghostTrail.length - 1]; if (!l || Math.hypot(l[0] - ghostXY[0], l[1] - ghostXY[1]) > 0.05) this.ghostTrail.push(ghostXY); }
  }
  colors() {
    if (!this.col) this.col = Object.fromEntries(['--map-floor', '--map-wall', '--map-ref', '--goal', '--start', '--ant', '--ghost'].map(k => [k, this.css(k)]));
    return this.col;
  }
  // Walls and reference states change only with the size, the goal labels or the checkbox: cache them.
  background() {
    const key = `${this.c.width}x${this.c.height}|${this.showRefs}|${this.labelVer || 0}`;
    if (key === this.bgKey) return this.bg;
    this.bgKey = key; this.bg.width = this.c.width; this.bg.height = this.c.height;
    const g = this.bg.getContext('2d'), C = this.colors(), mm = this.meta.maze_map, u = this.meta.maze_unit, s = this.c.width / this.w;
    g.fillStyle = C['--map-floor']; g.fillRect(0, 0, this.c.width, this.c.height);
    g.fillStyle = C['--map-wall'];
    for (let i = 0; i < mm.length; i++) for (let j = 0; j < mm[0].length; j++) if (mm[i][j] === 1) {
      const [px, py] = this.toPx(j * u - this.meta.offset_x - u / 2, i * u - this.meta.offset_y + u / 2);
      g.fillRect(px - 0.5, py - 0.5, u * s + 1, u * s + 1);
    }
    if (this.showRefs && this.refXY) {
      const R = this.refXY.length / 2;
      g.fillStyle = C['--map-ref'];
      for (let k = 0; k < R; k++) { const [px, py] = this.toPx(this.refXY[2 * k], this.refXY[2 * k + 1]); g.fillRect(px - 1, py - 1, 2, 2); }
      if (this.labelled) {
        g.fillStyle = C['--goal'];
        for (let k = 0; k < R; k++) if (this.labelled[k]) { const [px, py] = this.toPx(this.refXY[2 * k], this.refXY[2 * k + 1]); g.beginPath(); g.arc(px, py, 2.6, 0, 7); g.fill(); }
      }
    }
    return this.bg;
  }
  draw() {
    const g = this.g, C = this.colors(), s = this.c.width / this.w;
    g.drawImage(this.background(), 0, 0);
    const line = (pts, col) => {
      if (pts.length < 2) return;
      g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
      pts.forEach((p, k) => { const [px, py] = this.toPx(p[0], p[1]); k ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
    };
    line(this.ghostTrail, C['--ghost']); line(this.trail, C['--ant']);
    const dot = (p, col, r, ring) => {
      if (!p) return; const [px, py] = this.toPx(p[0], p[1]);
      g.beginPath(); g.arc(px, py, r, 0, 7);
      if (ring) { g.strokeStyle = col; g.lineWidth = 2.5; g.stroke(); } else { g.fillStyle = col; g.fill(); }
    };
    dot(this.start, C['--start'], 6, true);
    if (this.goal) { dot(this.goal, C['--goal'], this.meta.goal_tol * s, true); dot(this.goal, C['--goal'], 3); }
    dot(this.ghostXY, C['--ghost'], 4.5); dot(this.ant, C['--ant'], 5);
  }
}
