// three.js view of a MuJoCo scene (z-up). Static geoms are built once, moving geoms are updated
// from flattened [pos(3), xmat(9)] poses. A second translucent copy of the moving geoms shows the
// unguided comparison run ("ghost").
import * as THREE from 'three';
import { OrbitControls } from '../vendor/three/controls/OrbitControls.js';

const GEOM = { PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };

const PALETTE = {
  floorA: 0x1d2430, floorB: 0x262f3d, wall: 0xc9ccd3, wallTop: 0xe4e6ea,
  ant: 0xd9922e, antJoint: 0xb5741c, ghost: 0x6fa8ff, goal: 0xe8475a, start: 0x5cc28c,
  trail: 0xf0b45a, ghostTrail: 0x7fb2ff, sky: 0x10141b,
};

function checkerTexture(n = 32, ny = n) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d'), s = 256 / 2;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    g.fillStyle = (i + j) % 2 ? '#262f3d' : '#1d2430'; g.fillRect(i * s, j * s, s, s);
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(n, ny); t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function geometryFor(type, size, mesh) {
  switch (type) {
    case GEOM.MESH: {
      if (!mesh) return null;
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(mesh.vert), 3));
      g.setIndex(Array.from(mesh.face)); g.computeVertexNormals();
      return g;
    }
    case GEOM.SPHERE: return new THREE.SphereGeometry(size[0], 24, 16);
    case GEOM.CAPSULE: { const g = new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 16); g.rotateX(Math.PI / 2); return g; }
    case GEOM.CYLINDER: { const g = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 32); g.rotateX(Math.PI / 2); return g; }
    case GEOM.BOX: return new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    case GEOM.ELLIPSOID: { const g = new THREE.SphereGeometry(1, 24, 16); g.scale(size[0], size[1], size[2]); return g; }
    default: return null;
  }
}

function setPose(obj, p, o) {
  obj.matrix.set(p[o + 3], p[o + 4], p[o + 5], p[o],
                 p[o + 6], p[o + 7], p[o + 8], p[o + 1],
                 p[o + 9], p[o + 10], p[o + 11], p[o + 2],
                 0, 0, 0, 1);
  obj.matrixWorldNeedsUpdate = true;
}

export class SceneView {
  constructor(canvas, desc, meta) {
    this.canvas = canvas; this.meta = meta;
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFShadowMap;
    r.outputColorSpace = THREE.SRGBColorSpace;
    const scene = this.scene = new THREE.Scene();
    scene.background = new THREE.Color(PALETTE.sky);
    const cam = this.camera = new THREE.PerspectiveCamera(45, 1, 0.02, 400);
    cam.up.set(0, 0, 1);
    let span;
    this.palette = !!meta.maze_map;                 // the maze keeps the demo's own colours, other scenes use the model's
    if (meta.maze_map) {
      const mm = meta.maze_map, u = meta.maze_unit;
      this.center = new THREE.Vector3((mm[0].length - 1) * u / 2 - meta.offset_x, (mm.length - 1) * u / 2 - meta.offset_y, 0);
      span = Math.max(mm.length, mm[0].length) * u;
      cam.position.set(this.center.x, this.center.y - 0.75 * span, 1.05 * span);
    } else {
      // MuJoCo free-camera convention: the camera looks along (cos el cos az, cos el sin az, sin el) at lookat.
      const c = { ...(meta.camera || {}), ...(meta.view || {}) }, t = c.lookat || c.target || [0, 0, 0.5], d = c.distance || 4;
      const el = (c.elevation ?? -20) * Math.PI / 180, az = (c.azimuth ?? 90) * Math.PI / 180;
      this.center = new THREE.Vector3(...t); span = c.span || 2 * d;
      cam.position.set(t[0] - d * Math.cos(el) * Math.cos(az), t[1] - d * Math.cos(el) * Math.sin(az), t[2] - d * Math.sin(el));
    }
    this.span = span;
    this.controls = new OrbitControls(cam, canvas);
    this.controls.target.copy(this.center); this.controls.enableDamping = true; this.controls.update();
    this.follow = !!(meta.camera && meta.camera.follow);
    // the trail follows the first moving geom (the ant's torso) or the manipulated object
    const fg = desc.follow ? desc.geoms.find(g => g.name === desc.follow) : null;
    this.track = desc.cubeGeom !== undefined ? Math.max(0, desc.dynamic.indexOf(desc.cubeGeom)) : fg ? Math.max(0, desc.dynamic.indexOf(fg.id)) : 0;
    this.trailStep = meta.kind === 'manipspace' ? 0.002 : 0.02; this.trailZ = meta.kind === 'manipspace' ? null : 0.02;

    scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242c, 0.9));
    const sun = this.sun = new THREE.DirectionalLight(0xffffff, 1.6);
    this.sunOffset = new THREE.Vector3(-0.25 * span, -0.45 * span, span);
    sun.position.copy(this.center).add(this.sunOffset); sun.target.position.copy(this.center);
    sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera; sc.left = sc.bottom = -span * 0.75; sc.right = sc.top = span * 0.75; sc.near = 0.05 * span; sc.far = 4 * span;
    scene.add(sun, sun.target);

    this.dynamic = desc.dynamic; this.mainGeoms = []; this.ghostGeoms = [];
    const ghostMat = new THREE.MeshStandardMaterial({ color: PALETTE.ghost, transparent: true, opacity: 0.38, roughness: 0.6, depthWrite: false });
    for (const g of desc.geoms) {
      if (g.type === GEOM.PLANE) {
        // planes are infinite for collisions: narrow or short floors (DMC strips) are drawn at least 50 m wide
        const wide = !this.palette && meta.kind !== 'manipspace' ? 50 : 0;
        const hx = Math.max(g.size[0] || 200, wide), hy = Math.max(g.size[1] || 200, wide);
        const cell = meta.floor_cell || (this.palette ? 1 : meta.kind === 'manipspace' ? 0.25 : 1);
        const geo = new THREE.PlaneGeometry(2 * hx, 2 * hy);
        const mat = new THREE.MeshStandardMaterial({ map: checkerTexture(Math.max(1, Math.round(hx / cell)), Math.max(1, Math.round(hy / cell))), roughness: 0.95 });
        const mesh = new THREE.Mesh(geo, mat); mesh.matrixAutoUpdate = false; setPose(mesh, [...g.pos, ...g.mat], 0); mesh.receiveShadow = true; scene.add(mesh);
        continue;
      }
      if (g.name === 'target' && meta.maze_map) { this.buildGoal(g); continue; }
      if (g.rgba && g.rgba[3] === 0) continue;                     // invisible in the model
      if (g.visible === false || (g.group !== undefined && g.group >= 3)) continue;   // collision-only groups
      if (desc.targetGeom !== undefined && g.id === desc.targetGeom) continue;        // drawn as the goal marker
      const mesh0 = typeof g.mesh === 'number' ? (g.mesh >= 0 && desc.meshes ? desc.meshes[g.mesh] : null) : g.mesh;
      const geo = geometryFor(g.type, g.size, mesh0); if (!geo) continue;
      let color, opacity = 1;
      if (this.palette) { color = PALETTE.wall; if (g.body !== 0) color = /ankle/.test(g.name) ? PALETTE.antJoint : PALETTE.ant; }
      else { color = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]); opacity = g.rgba[3]; }
      const mat = new THREE.MeshStandardMaterial({ color, roughness: g.body !== 0 ? 0.45 : 0.85, metalness: 0.0, transparent: opacity < 1, opacity });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true; mesh.receiveShadow = true; mesh.matrixAutoUpdate = false;
      setPose(mesh, [...g.pos, ...g.mat], 0);
      scene.add(mesh);
      if (g.body !== 0) {
        mesh.userData.dyn = desc.dynamic.indexOf(g.id); this.mainGeoms.push(mesh);
        const gm = new THREE.Mesh(geo, ghostMat); gm.matrixAutoUpdate = false; gm.visible = false; gm.renderOrder = 2;
        gm.userData.dyn = mesh.userData.dyn; setPose(gm, [...g.pos, ...g.mat], 0); scene.add(gm); this.ghostGeoms.push(gm);
      }
    }
    if (!this.goal) this.buildGoal(null);
    // Manipulation: the area where a click places the target (OGBench target bounds) is marked on the table, and a ring
    // previews the target under the pointer.
    const ct = meta.click_targets;
    if (ct && ct.bounds_xy) {
      const [[x0, y0], [x1, y1]] = ct.bounds_xy, z = 0.0015;
      const fill = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0), new THREE.MeshBasicMaterial({ color: PALETTE.goal, transparent: true, opacity: 0.1, depthWrite: false }));
      fill.position.set((x0 + x1) / 2, (y0 + y1) / 2, z);
      const edge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([x, y]) => new THREE.Vector3(x, y, z))),
                                      new THREE.LineBasicMaterial({ color: PALETTE.goal, transparent: true, opacity: 0.6 }));
      this.clickArea = new THREE.Group(); this.clickArea.add(fill, edge); this.clickArea.renderOrder = 1; scene.add(this.clickArea);
      const rad = (meta.success && meta.success.threshold) || 0.04;
      this.hoverMarker = new THREE.Mesh(new THREE.RingGeometry(0.75 * rad, rad, 40), new THREE.MeshBasicMaterial({ color: PALETTE.goal, transparent: true, opacity: 0.75, side: THREE.DoubleSide, depthWrite: false }));
      this.hoverMarker.visible = false; scene.add(this.hoverMarker);
    }
    this.trail = this.makeTrail(PALETTE.trail, 0.95); this.ghostTrail = this.makeTrail(PALETTE.ghostTrail, 0.8);
    this.startMarker = new THREE.Mesh(new THREE.RingGeometry(0.35, 0.5, 32), new THREE.MeshBasicMaterial({ color: PALETTE.start, side: THREE.DoubleSide }));
    this.startMarker.position.z = 0.02; this.startMarker.visible = !!meta.maze_map; scene.add(this.startMarker);
    this.raycaster = new THREE.Raycaster(); this.floorPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
  }

  buildGoal(g) {
    const grp = new THREE.Group();
    // 3-D target: the success region (a sphere of the success radius) around a small solid core
    const rad = (this.meta.success && this.meta.success.threshold) || 0.04;
    this.goalBox = new THREE.Group();
    this.goalBox.add(new THREE.Mesh(new THREE.SphereGeometry(rad, 32, 16), new THREE.MeshBasicMaterial({ color: PALETTE.goal, transparent: true, opacity: 0.22, depthWrite: false })));
    this.goalBox.add(new THREE.Mesh(new THREE.SphereGeometry(rad * 0.15, 16, 8), new THREE.MeshBasicMaterial({ color: PALETTE.goal })));
    this.goalBox.visible = false; this.scene.add(this.goalBox);
    if (!g) { this.goal = grp; grp.visible = false; this.scene.add(grp); return; }
    const disk = new THREE.Mesh(new THREE.CircleGeometry(this.meta.goal_tol, 48), new THREE.MeshBasicMaterial({ color: PALETTE.goal, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
    disk.position.z = 0.03;
    const ring = new THREE.Mesh(new THREE.RingGeometry(this.meta.goal_tol * 0.92, this.meta.goal_tol, 48), new THREE.MeshBasicMaterial({ color: PALETTE.goal, side: THREE.DoubleSide }));
    ring.position.z = 0.035;
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 3, 12), new THREE.MeshBasicMaterial({ color: PALETTE.goal, transparent: true, opacity: 0.6 }));
    beam.rotation.x = Math.PI / 2; beam.position.z = 1.5;
    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.22, 20, 12), new THREE.MeshStandardMaterial({ color: PALETTE.goal, emissive: 0x5a0a14 }));
    cap.position.z = 3.0;
    grp.add(disk, ring, beam, cap); this.goal = grp; this.scene.add(grp);
  }

  makeTrail(color, opacity) {
    const max = 20000, geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * max), 3));
    geo.setDrawRange(0, 0);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
    line.frustumCulled = false; this.scene.add(line);
    return { line, geo, n: 0, max };
  }
  // The tracked point of a pose frame: [x, y, z].
  trackPoint(poses) { const o = 12 * this.track; return [poses[o], poses[o + 1], poses[o + 2]]; }
  pushTrail(tr, x, y, z) {
    if (tr.n >= tr.max) return;
    const a = tr.geo.attributes.position, zz = this.trailZ !== null ? this.trailZ : z;
    if (tr.n && Math.hypot(a.getX(tr.n - 1) - x, a.getY(tr.n - 1) - y, a.getZ(tr.n - 1) - zz) < this.trailStep) return;   // skip points that do not move
    a.setXYZ(tr.n, x, y, zz); tr.n += 1;
    a.needsUpdate = true; tr.geo.setDrawRange(0, tr.n);
  }
  clearTrails() { for (const tr of [this.trail, this.ghostTrail]) { tr.n = 0; tr.geo.setDrawRange(0, 0); } }

  setGoal(p) {
    if (!p) { this.goal.visible = false; this.goalBox.visible = false; return; }
    if (p.length === 3) { this.goalBox.position.set(p[0], p[1], p[2]); this.goalBox.visible = true; this.goal.visible = false; }
    else { this.goal.position.set(p[0], p[1], 0); this.goal.visible = this.goal.children.length > 0; }
  }
  setStart(xy) { if (xy) this.startMarker.position.set(xy[0], xy[1], 0.02); }
  setGhostVisible(v) { for (const m of this.ghostGeoms) m.visible = v; this.ghostTrail.line.visible = v; }

  applyPoses(poses, ghost) {
    const list = ghost ? this.ghostGeoms : this.mainGeoms;
    for (let k = 0; k < list.length; k++) setPose(list[k], poses, 12 * list[k].userData.dyn);
    const p = this.trackPoint(poses);
    this.pushTrail(ghost ? this.ghostTrail : this.trail, p[0], p[1], p[2]);
    if (!ghost) this.antXY = [p[0], p[1]];
  }

  floorPoint(ev) { return this.pickPoint(ev, 'floor'); }
  // Clicked point on the floor (z = 0, returns [x, y]) or on the work surface (z = meta.pick_z, returns [x, y, z_goal]).
  pickPoint(ev, kind) {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const ct = this.meta.click_targets || {};       // {bounds_xy: [[xmin, ymin], [xmax, ymax]], z}
    const z = kind === 'floor' ? 0 : (ct.z ?? 0), p = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -z), p)) return null;
    if (kind === 'floor') return [p.x, p.y];
    const b = ct.bounds_xy;
    if (b) {
      // a click near the area snaps to its edge, a click far from it is ignored
      const m = ct.snap_margin ?? 0.15;
      if (p.x < b[0][0] - m || p.x > b[1][0] + m || p.y < b[0][1] - m || p.y > b[1][1] + m) return null;
      p.x = Math.min(b[1][0], Math.max(b[0][0], p.x)); p.y = Math.min(b[1][1], Math.max(b[0][1], p.y));
    }
    return [p.x, p.y, z];
  }

  // Preview of a click on the work surface (ev = null hides it); returns whether a click there places a target.
  hover(ev, kind) {
    if (!this.hoverMarker) return false;
    const p = ev ? this.pickPoint(ev, kind) : null;
    this.hoverMarker.visible = !!p;
    if (p) this.hoverMarker.position.set(p[0], p[1], 0.002);
    return !!p;
  }

  resize() {
    const el = this.canvas.parentElement, w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  render() {
    if (this.follow && this.antXY) {
      const t = new THREE.Vector3(this.antXY[0], this.antXY[1], this.controls.target.z);
      const d = t.clone().sub(this.controls.target).multiplyScalar(0.12);
      this.controls.target.add(d); this.camera.position.add(d);
      this.sun.position.copy(this.controls.target).add(this.sunOffset); this.sun.target.position.copy(this.controls.target);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
