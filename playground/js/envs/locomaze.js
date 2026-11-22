// OGBench locomaze (AntMaze) behind the demo's common environment interface (see ENV_INTERFACE.md).
// The physics and the success test are LocoMaze in ../sim.js. A task fixes a start cell and a goal cell, a click
// can move the goal (or, with shift, the start) anywhere on the free floor.
import { LocoMaze } from '../sim.js';

export class Env extends LocoMaze {
  constructor(mj, xml, meta) {
    super(mj, xml, meta);
    this.features = { map: true, clickGoal: 'floor', clickStart: true, taskSwitch: 'restart', successTask: true };
    this.taskIdx = 0; this.start = [0, 0]; this.ret = 0; this.lastReward = 0;
    this.setTask(0);
  }
  tasks() { return this.meta.tasks.map((t, k) => ({ label: `Task ${k + 1}`, title: `OGBench evaluation task ${k + 1}: start cell (${t.init_ij}) → goal cell (${t.goal_ij})` })); }
  setTask(k) {
    const t = this.meta.tasks[k]; this.taskIdx = k;
    this.start = this.ijToXy(...t.init_ij); this.setGoal(this.ijToXy(...t.goal_ij));
  }
  setStart(xy) { this.start = [xy[0], xy[1]]; this.taskIdx = null; }
  reset(rng) { super.reset(this.start, rng); this.ret = 0; this.lastReward = 0; }
  copyStateFrom(other) { super.copyStateFrom(other); this.ret = 0; this.lastReward = 0; }
  step(action, onSubstep) {
    const s = super.step(action, onSubstep);
    this.lastReward = s ? 1 : 0; this.ret += this.lastReward;
    return s;
  }
  newLeg() { this.t = 0; this.success = false; this.done = false; this.ret = 0; }
  hud() { return [['distance to goal', this.success ? 'reached' : this.distToGoal().toFixed(2)]]; }
  marker() { return { goal: this.goal, start: this.start }; }
}
