/**
 * 360° 3D workflow of the research model (Model Architecture page).
 *
 * The pipeline stands on a round stage: one platform per workflow stage, modality A in
 * the back lane, modality B in the front lane, joined at fusion. Data flows along +x.
 *   • WebGL   — the objects (real input thumbnails, conv feature-map stacks, pooling
 *               funnels, h vectors, α / 1 / × spheres, fusion, classifier, class tiles),
 *               the connections, and "tensor packets" that travel along every connection
 *               to show the direction of the forward pass (an illustration — nothing is
 *               executed on the page).
 *   • CSS3D   — camera-facing labels (crisp text at any zoom, theme-aware), drawn on top.
 * Every shape, parameter count and class code comes from the ArchGraph (generated from
 * the code + checkpoints); research numbers come from research.ts via the page.
 * The camera orbits freely (full 360°), can auto-rotate, and flies to a stage or node.
 */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS3DRenderer, CSS3DSprite } from "three/examples/jsm/renderers/CSS3DRenderer.js";
import media from "../data/architecture.media.json";
import { fmtShape, sectionOf, stageInfo, stageOf, type ArchGraph, type ArchNode, type SectionKey, type StageKey } from "./model";

export interface SceneTheme {
  ink: string;
  muted: string;
  line: string;
  surface: string;
  a: string;
  b: string;
  joint: string;
  forward: string;
  gradient: string;
  warning: string;
  arrow: string;
  sphere: string;
  panel: string;
  border: string;
  card: string;
  zoneA: string;
  zoneB: string;
  zoneInt: string;
  zoneFus: string;
  intInk: string;
  fusInk: string;
  chip: string;
  dark: boolean;
}

export interface SceneState {
  selected: string | null;
  path: { nodes: Set<string>; edges: Set<string> } | null;
  flowNode: string | null;
  flowEdge: string | null;
  overlay: boolean;
  /** The section (platform) lifted out of the stage; its part labels are shown. */
  raised: SectionKey | null;
}

export interface SceneCallbacks {
  onHover: (id: string | null, clientX: number, clientY: number) => void;
  onSelect: (id: string | null) => void;
  /** The auto-orbit stopped (user grabbed the camera) or started. */
  onOrbitChange?: (on: boolean) => void;
}

export type ViewPreset = "front" | "top" | "side" | "back";

/* ── world layout (y up · data flows along +x · lane A behind, lane B in front) ── */
const Y = 0.66; // height of the data-flow lines
const LANE = { a: -1.8, b: 1.8 } as const;
const X = {
  raw: -10.8,
  mid: -8.75,
  fin: -6.75,
  block1: -4.25,
  block2: -2.9,
  block3: -1.65,
  gap: -0.6,
  tmean: 0.15,
  fc: 0.95,
  h: 1.8,
  alpha: 3.2,
  times: 4.2,
  fusion: 6.0,
  head: 8.2,
  output: 10.3,
} as const;
const SCENE_BOX = new THREE.Box3(new THREE.Vector3(-13.1, 0, -3.9), new THREE.Vector3(12.2, 2.2, 3.9));
const FLOOR_R = 13.8;
const LS = 0.0084; // world units per CSS px of a label
const LIFT = 0.95; // how far a selected section rises
const VIEWS: Record<ViewPreset, [az: number, pol: number]> = {
  front: [-0.42, 1.0],
  top: [0, 0.06],
  side: [-1.42, 1.18],
  back: [Math.PI - 0.42, 1.0],
};
const INTRO: [number, number] = [-1.25, 1.12];

const ASSET = (f: string) => `${import.meta.env.BASE_URL}arch/${f}`;
const cap = (s: string) => s[0].toUpperCase() + s.slice(1);
const keyOf = (id: string) => id.replace(/^[ab]_/, "");
const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const dirOf = (az: number, pol: number) => V(Math.sin(pol) * Math.sin(az), Math.cos(pol), Math.sin(pol) * Math.cos(az)).normalize();
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const WAVE = ["#e07a2f", "#3f8fd0", "#3fa66a", "#c9a227", "#8e6cc9", "#d0527a"];

interface NodeVis {
  meshes: THREE.Mesh[];
}
interface EdgeSpec {
  kind: "line" | "fusion";
  /** Points at rest, each tagged with the section whose lift it follows. */
  pts: { p: THREE.Vector3; s: SectionKey }[];
}
interface EdgeRec {
  key: string;
  spec: EdgeSpec;
  secs: Set<SectionKey>;
  curve: THREE.Curve<THREE.Vector3>;
  len: number;
  tubes: THREE.Mesh[];
  tip: THREE.Mesh;
  color: THREE.Color;
  first: number; // index of its first packet
  n: number;
}
/** ground = stage titles and lane tags (always shown) · glyph = α / 1 / × (always) ·
 *  node = part labels (only for the raised section) · overlay = research overlay tags. */
type LabelKind = "ground" | "glyph" | "node" | "overlay";
interface Label {
  obj: CSS3DSprite;
  wrap: HTMLDivElement;
  kind: LabelKind;
  section: SectionKey | null;
  hidden: boolean;
  op: number;
}
interface Zone {
  stage: StageKey;
  section: SectionKey;
  mesh: THREE.Mesh;
  fill: string;
}
interface Flight {
  t0: number;
  dur: number;
  fromT: THREE.Vector3;
  toT: THREE.Vector3;
  fromS: THREE.Spherical;
  toS: THREE.Spherical;
}

export class ArchitectureScene {
  private gl: THREE.WebGLRenderer;
  private css: CSS3DRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private objs = new THREE.Group();
  private overlayGroup = new THREE.Group();
  private labelGroup = new THREE.Group();
  /** One group per section; a selected section's group rises by LIFT. */
  private sections = new Map<SectionKey, { group: THREE.Group; lift: number; target: number }>();
  private cur: THREE.Object3D = this.objs;
  private curSec: SectionKey | null = null;
  private hitMat = new THREE.MeshBasicMaterial({ visible: false });
  private vis = new Map<string, NodeVis>();
  private boxes = new Map<string, THREE.Box3>();
  private anchors = new Map<string, { in: THREE.Vector3; out: THREE.Vector3 }>();
  private pickables: THREE.Object3D[] = [];
  private edges: EdgeRec[] = [];
  private edgeByKey = new Map<string, EdgeRec>();
  private labels: Label[] = [];
  private zones: Zone[] = [];
  private beacons: { mesh: THREE.Mesh; y: number; phase: number }[] = [];
  private packets: THREE.InstancedMesh | null = null;
  private halos: THREE.InstancedMesh | null = null;
  private ovPaths: { curve: THREE.Curve<THREE.Vector3>; n: number; first: number; speed: number }[] = [];
  private ovPackets: THREE.InstancedMesh | null = null;
  private tex = new Map<string, THREE.Texture>();
  private dynTex: THREE.Texture[] = [];
  private loader = new THREE.TextureLoader();
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private downAt: { x: number; y: number } | null = null;
  private hoverId: string | null = null;
  private frame = 0;
  private running = false;
  private inView = true;
  private lastT = 0;
  private flight: Flight | null = null;
  private resizeObs: ResizeObserver;
  private io: IntersectionObserver | null = null;
  private theme!: SceneTheme;
  private graph: ArchGraph | null = null;
  private state: SceneState = { selected: null, path: null, flowNode: null, flowEdge: null, overlay: false, raised: null };
  private reduceMotion: boolean;
  private disposed = false;
  private introDone = false;
  private dummy = new THREE.Object3D();
  private tmpC = new THREE.Color();

  constructor(private host: HTMLElement, private cb: SceneCallbacks) {
    this.gl = new THREE.WebGLRenderer({ antialias: true, alpha: true }); // throws without WebGL
    this.gl.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.gl.setClearColor(0x000000, 0);
    this.gl.shadowMap.enabled = true;
    this.gl.shadowMap.type = THREE.PCFSoftShadowMap;
    Object.assign(this.gl.domElement.style, { position: "absolute", inset: "0", width: "100%", height: "100%", display: "block" });
    this.gl.domElement.setAttribute("aria-hidden", "true");
    host.appendChild(this.gl.domElement);

    // labels sit above the WebGL layer and let pointer events through to the canvas
    this.css = new CSS3DRenderer();
    Object.assign(this.css.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none", overflow: "hidden" });
    host.appendChild(this.css.domElement);

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 400);
    this.controls = new OrbitControls(this.camera, this.gl.domElement);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 2.2;
    this.controls.maxDistance = 75;
    this.controls.minPolarAngle = 0.05; // full 360° around; from straight above down to the floor
    this.controls.maxPolarAngle = Math.PI / 2 - 0.03;
    this.controls.autoRotateSpeed = 0.75; // one revolution ≈ 80 s
    this.controls.addEventListener("change", () => this.requestRender());

    this.reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    this.scene.add(this.objs, this.overlayGroup, this.labelGroup);

    const el = this.gl.domElement;
    el.addEventListener("pointermove", this.onMove);
    el.addEventListener("pointerdown", this.onDown);
    el.addEventListener("pointerup", this.onUp);
    el.addEventListener("pointerleave", this.onLeave);
    el.addEventListener("dblclick", this.onDbl);
    el.addEventListener("wheel", this.stopOrbit, { passive: true });
    document.addEventListener("visibilitychange", this.onVisibility);
    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(host);
    if (typeof IntersectionObserver !== "undefined") {
      this.io = new IntersectionObserver(([e]) => {
        this.inView = e.isIntersecting;
        this.kick();
      });
      this.io.observe(host);
    }
    this.resize();
  }

  /* ── public API ───────────────────────────────────────────────── */

  setGraph(graph: ArchGraph, theme: SceneTheme) {
    this.graph = graph;
    this.theme = theme;
    this.rebuild();
    if (!this.introDone && !this.reduceMotion) {
      // brief fly-in from the side so the depth of the scene is obvious, then a slow orbit
      this.introDone = true;
      this.place(dirOf(...INTRO), 1.25);
      this.flyToView("front", 2200);
      this.setAutoRotate(true);
    } else {
      this.introDone = true;
      this.place(dirOf(...VIEWS.front), 1);
    }
  }

  setTheme(theme: SceneTheme) {
    if (!this.graph) return;
    this.theme = theme;
    this.rebuild();
    this.requestRender();
  }

  setState(state: SceneState) {
    this.state = state;
    this.applyState();
  }

  resetCamera() {
    this.flyToView("front");
  }

  fitView() {
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.flyTo(SCENE_BOX.getCenter(new THREE.Vector3()), dir, this.fitDistance(SCENE_BOX, dir) * 1.02);
  }

  setView(v: ViewPreset) {
    this.flyToView(v);
  }

  setAutoRotate(on: boolean) {
    const want = on && !this.reduceMotion;
    if (this.controls.autoRotate !== want) {
      this.controls.autoRotate = want;
      this.cb.onOrbitChange?.(want);
    }
    this.kick();
  }

  /** Fly the camera to a node, a figure component ("concept:…") or a whole stage ("stage:…"). */
  focus(id: string | null) {
    if (!id || !this.graph) return;
    let box: THREE.Box3 | undefined;
    if (id.startsWith("stage:")) {
      const s = id.slice(6) as StageKey;
      box = new THREE.Box3();
      for (const z of this.zones) if (z.stage === s) box.expandByObject(z.mesh);
      for (const [nid, b] of this.boxes) if (stageOf(this.graph, nid) === s) box.union(b);
      if (box.isEmpty()) return;
    } else {
      box = this.boxes.get(id)?.clone().translate(V(0, LIFT, 0));
    }
    if (!box) return;
    const size = box.getSize(new THREE.Vector3());
    box.expandByVector(V(Math.max(0, (3.4 - size.x) / 2), Math.max(0, (1.4 - size.y) / 2), Math.max(0, (3.4 - size.z) / 2)));
    const cur = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target));
    const dir = dirOf(cur.theta, THREE.MathUtils.clamp(cur.phi, 0.62, 1.2));
    this.setAutoRotate(false);
    this.flyTo(box.getCenter(new THREE.Vector3()), dir, this.fitDistance(box, dir) * 1.12);
  }

  /* ── camera ───────────────────────────────────────────────────── */

  private fitDistance(box: THREE.Box3, dir: THREE.Vector3) {
    const center = box.getCenter(new THREE.Vector3());
    const d = dir.clone().normalize();
    const view = new THREE.Matrix4().lookAt(d, new THREE.Vector3(), new THREE.Vector3(0, 1, 0));
    const right = new THREE.Vector3().setFromMatrixColumn(view, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(view, 1);
    const tanV = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
    const tanH = tanV * this.camera.aspect;
    let dist = 1;
    for (let i = 0; i < 8; i++) {
      const c = V(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).sub(center);
      dist = Math.max(dist, Math.abs(c.dot(right)) / tanH + c.dot(d), Math.abs(c.dot(up)) / tanV + c.dot(d));
    }
    return dist;
  }

  /** Jump (no animation) to a direction at `zoom` × the fitting distance. */
  private place(dir: THREE.Vector3, zoom: number) {
    const c = SCENE_BOX.getCenter(new THREE.Vector3());
    this.controls.target.copy(c);
    this.camera.position.copy(c).addScaledVector(dir, this.fitDistance(SCENE_BOX, dir) * zoom);
    this.controls.update();
    this.requestRender();
  }

  private flyToView(v: ViewPreset, dur = 1100) {
    const dir = dirOf(...VIEWS[v]);
    if (v === "top") this.setAutoRotate(false);
    this.flyTo(SCENE_BOX.getCenter(new THREE.Vector3()), dir, this.fitDistance(SCENE_BOX, dir) * (v === "top" ? 0.98 : 1.0), dur);
  }

  private flyTo(target: THREE.Vector3, dir: THREE.Vector3, dist: number, dur = 1000) {
    const toPos = target.clone().addScaledVector(dir.clone().normalize(), dist);
    if (this.reduceMotion || dur <= 0) {
      this.controls.target.copy(target);
      this.camera.position.copy(toPos);
      this.controls.update();
      this.requestRender();
      return;
    }
    const fromS = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target));
    const toS = new THREE.Spherical().setFromVector3(toPos.clone().sub(target));
    let dt = toS.theta - fromS.theta;
    dt = ((((dt + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI; // shortest way round
    toS.theta = fromS.theta + dt;
    this.flight = { t0: performance.now(), dur, fromT: this.controls.target.clone(), toT: target.clone(), fromS, toS };
    this.kick();
  }

  /* ── construction ─────────────────────────────────────────────── */

  private rebuild() {
    this.clear(this.objs);
    this.clear(this.overlayGroup);
    this.clear(this.labelGroup);
    this.dynTex.forEach((t) => t.dispose());
    this.dynTex = [];
    this.hitMat = new THREE.MeshBasicMaterial({ visible: false });
    this.vis.clear();
    this.boxes.clear();
    this.anchors.clear();
    this.pickables = [];
    this.edges = [];
    this.edgeByKey.clear();
    this.labels = [];
    this.zones = [];
    this.beacons = [];
    this.ovPaths = [];
    this.packets = this.halos = this.ovPackets = null;
    this.sections.clear();
    this.enter(null);
    const g = this.graph!;
    this.buildStage(g);
    for (const br of ["a", "b"] as const) this.buildLane(g, br);
    this.buildIntervention(g);
    this.buildFusion(g);
    this.buildClassifier(g);
    this.buildOutput(g);
    this.buildEdges(g);
    this.buildBeacons(g);
    this.buildOverlay(g);
    this.buildPackets();
    this.applyState();
  }

  /** Subsequent objects and labels belong to `key` (null = static, never lifted). */
  private enter(key: SectionKey | null) {
    this.curSec = key;
    if (!key) {
      this.cur = this.objs;
      return;
    }
    let sec = this.sections.get(key);
    if (!sec) {
      const group = new THREE.Group();
      this.objs.add(group);
      sec = { group, lift: 0, target: 0 };
      this.sections.set(key, sec);
    }
    this.cur = sec.group;
  }

  private std(color: string, opts: Partial<THREE.MeshStandardMaterialParameters> = {}) {
    return new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.05, transparent: true, ...opts });
  }

  private tint(color: string, t: number) {
    return `#${new THREE.Color(color).lerp(new THREE.Color(this.theme.dark ? "#0d1117" : "#ffffff"), t).getHexString()}`;
  }

  /** Adds a mesh that casts a shadow; registers it for highlight / dimming under `id`. */
  private put<T extends THREE.Object3D>(o: T, id?: string): T {
    o.traverse((c) => {
      if (c instanceof THREE.Mesh) c.castShadow = true;
    });
    this.cur.add(o);
    if (id && o instanceof THREE.Mesh) {
      const v = this.vis.get(id) ?? { meshes: [] };
      v.meshes.push(o);
      o.userData.baseOpacity ??= (o.material as THREE.Material).opacity;
      this.vis.set(id, v);
    }
    return o;
  }

  /** Invisible pick box around a node's objects; also its camera-focus box. */
  private register(id: string, objs: THREE.Object3D[], pad = 0.1) {
    const box = new THREE.Box3();
    for (const o of objs) {
      o.updateMatrixWorld(true);
      box.expandByObject(o);
    }
    box.expandByScalar(pad);
    const prev = this.boxes.get(id);
    this.boxes.set(id, prev ? prev.union(box) : box);
    const size = box.getSize(new THREE.Vector3());
    const m = new THREE.Mesh(new THREE.BoxGeometry(size.x, size.y, size.z), this.hitMat);
    m.position.copy(box.getCenter(new THREE.Vector3()));
    m.userData.id = id;
    this.cur.add(m);
    this.pickables.push(m);
  }

  /** Camera-facing label. `anchor` = which point of the label sits at `pos`. */
  private label(html: string, pos: THREE.Vector3, anchor: "bottom" | "center" | "top" = "bottom", kind: LabelKind = "node") {
    const wrap = document.createElement("div");
    wrap.className = "a3-anchor";
    const inner = document.createElement("div");
    inner.className = `a3-in a3-in--${anchor}`;
    inner.innerHTML = html;
    wrap.appendChild(inner);
    const obj = new CSS3DSprite(wrap);
    obj.position.copy(pos);
    obj.scale.setScalar(LS);
    (kind === "overlay" ? this.overlayGroup : this.curSec ? this.cur : this.labelGroup).add(obj);
    this.labels.push({ obj, wrap, kind, section: kind === "overlay" ? null : this.curSec, hidden: false, op: 1 });
    return inner;
  }

  private chip(id: string, title: string, sub: string, pos: THREE.Vector3) {
    this.label(`<div class="a3-chip" data-node="${id}"><span class="a3-chip-t">${title}</span>${sub ? `<span class="a3-chip-s">${sub}</span>` : ""}</div>`, pos);
  }

  private image(file: string, nearest = false) {
    let t = this.tex.get(file);
    if (!t) {
      t = this.loader.load(ASSET(file), () => this.requestRender());
      t.colorSpace = THREE.SRGBColorSpace;
      if (nearest) t.magFilter = THREE.NearestFilter;
      t.anisotropy = 4;
      this.tex.set(file, t);
    }
    return t;
  }

  /** One frame of a horizontal strip image. */
  private frameTex(file: string, i: number, n: number, nearest = false) {
    const t = this.image(file, nearest).clone();
    t.repeat.set(1 / n, 1);
    t.offset.set(i / n, 0);
    this.dynTex.push(t);
    return t;
  }

  private canvasTex(w: number, h: number, draw: (c: CanvasRenderingContext2D) => void) {
    const cv = document.createElement("canvas");
    cv.width = w;
    cv.height = h;
    const ctx = cv.getContext("2d");
    if (ctx) draw(ctx);
    const t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    this.dynTex.push(t);
    return t;
  }

  /** A textured card standing upright, slightly turned towards the default camera. */
  private card(map: THREE.Texture, w: number, h: number, pos: THREE.Vector3, id: string, turn = -0.32) {
    const g = new THREE.Group();
    const back = new THREE.Mesh(new THREE.BoxGeometry(w + 0.07, h + 0.07, 0.035), this.std(this.theme.card));
    const face = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map, transparent: true, side: THREE.DoubleSide }));
    face.position.z = 0.02;
    const face2 = face.clone();
    face2.position.z = -0.02;
    face2.rotation.y = Math.PI;
    g.add(back, face, face2);
    g.position.copy(pos);
    g.rotation.y = turn;
    this.put(g);
    for (const m of [back, face, face2]) {
      const v = this.vis.get(id) ?? { meshes: [] };
      m.userData.baseOpacity = 1;
      v.meshes.push(m);
      this.vis.set(id, v);
    }
    return g;
  }

  /* stage: floor + one platform per workflow stage ------------------- */

  private platform(x0: number, x1: number, z0: number, z1: number, fill: string, edge: string, stage: StageKey, section: SectionKey) {
    this.enter(section);
    const r = 0.38;
    const s = new THREE.Shape();
    // shape is drawn in (x, -z); rotateX(-90°) maps it onto the floor
    s.moveTo(x0 + r, -z0);
    s.lineTo(x1 - r, -z0);
    s.quadraticCurveTo(x1, -z0, x1, -z0 - r);
    s.lineTo(x1, -z1 + r);
    s.quadraticCurveTo(x1, -z1, x1 - r, -z1);
    s.lineTo(x0 + r, -z1);
    s.quadraticCurveTo(x0, -z1, x0, -z1 + r);
    s.lineTo(x0, -z0 - r);
    s.quadraticCurveTo(x0, -z0, x0 + r, -z0);
    const geo = new THREE.ExtrudeGeometry(s, { depth: 0.1, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.03, bevelSegments: 2, curveSegments: 6 });
    geo.rotateX(-Math.PI / 2);
    const mesh = new THREE.Mesh(geo, this.std(fill, { roughness: 0.85, transparent: false }));
    mesh.receiveShadow = true;
    mesh.castShadow = true; // a raised platform throws a shadow on the stage
    mesh.userData.id = `section:${section}`;
    this.cur.add(mesh);
    this.pickables.push(mesh);
    const pts = s.getPoints(8).map((p) => V(p.x, 0.125, -p.y));
    const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: edge }));
    this.cur.add(line);
    this.zones.push({ stage, section, mesh, fill });
  }

  private zoneTitle(stage: StageKey, section: SectionKey, title: string, sub: string, pos: THREE.Vector3) {
    this.enter(section);
    this.label(`<div class="a3-zone" data-stage="${stage}">${title}<span>${sub}</span></div>`, pos, "top", "ground");
  }

  private buildStage(g: ArchGraph) {
    const T = this.theme;
    this.scene.children.filter((c) => c instanceof THREE.Light).forEach((l) => this.scene.remove(l));
    const hemi = new THREE.HemisphereLight(0xffffff, T.dark ? 0x30343c : 0xe9e0d2, T.dark ? 1.35 : 1.9);
    const sun = new THREE.DirectionalLight(0xffffff, T.dark ? 1.5 : 1.7);
    sun.position.set(-7, 15, 10);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -14, right: 14, top: 14, bottom: -14, near: 1, far: 40 });
    sun.shadow.bias = -0.0004;
    sun.shadow.radius = 4;
    this.scene.add(hemi, sun);

    const floorCol = `#${new THREE.Color(T.surface).lerp(new THREE.Color(T.line), T.dark ? 0.35 : 0.45).getHexString()}`;
    const floor = new THREE.Mesh(new THREE.CircleGeometry(FLOOR_R, 128), this.std(floorCol, { roughness: 1, transparent: false }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.002;
    floor.receiveShadow = true;
    this.objs.add(floor);
    const grid = new THREE.PolarGridHelper(FLOOR_R, 24, 7, 128, T.border, T.border);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = T.dark ? 0.45 : 0.6;
    grid.position.y = 0.001;
    this.objs.add(grid);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(FLOOR_R, 0.05, 8, 160), this.std(T.border, { transparent: false }));
    rim.rotation.x = Math.PI / 2;
    this.objs.add(rim);

    const za = LANE.a;
    const zb = LANE.b;
    this.platform(-12.05, -9.75, za - 1.3, zb + 1.3, T.panel, T.border, "inputs", "inputs");
    this.platform(-9.6, -5.55, za - 1.3, zb + 1.3, T.panel, T.border, "preprocessing", "preprocessing");
    this.platform(-5.35, 2.4, za - 1.3, -0.45, T.zoneA, T.border, "encoding", "enc_a");
    this.platform(-5.35, 2.4, 0.45, zb + 1.3, T.zoneB, T.border, "encoding", "enc_b");
    this.platform(2.55, 4.85, za - 1.3, zb + 1.3, T.zoneInt, T.border, "intervention", "intervention");
    this.platform(5.0, 7.0, za - 1.95, zb + 1.95, T.zoneFus, T.border, "fusion", "fusion");
    this.platform(7.15, 9.25, -1.75, 1.75, T.panel, T.border, "classifier", "classifier");
    this.platform(9.4, 12.05, -2.45, 2.45, T.panel, T.border, "output", "output");

    const front = zb + 1.5;
    const head = g.nodes.find((n) => n.id === "head")!;
    this.zoneTitle("inputs", "inputs", "Inputs", `${g.modalities.a} + ${g.modalities.b}`, V(-10.9, 0.14, front));
    this.zoneTitle("preprocessing", "preprocessing", "Preprocessing", "deterministic · not learned", V(-7.6, 0.14, front));
    this.zoneTitle(
      "encoding",
      "enc_b",
      `Encoders ℰ<sub>A</sub> · ℰ<sub>B</sub>`,
      `${g.encoders.a} ${g.params.encoder_a.toLocaleString("en-US")} · ${g.encoders.b} ${g.params.encoder_b.toLocaleString("en-US")} params`,
      V(-1.5, 0.14, front),
    );
    // lane tags at the start of each modality lane (static ground labels)
    this.enter(null);
    for (const br of ["a", "b"] as const) {
      const pf = g.nodes.find((n) => n.branch === br && n.per_frame)?.per_frame;
      this.label(
        `<div class="a3-lane a3-lane--${br}">${br.toUpperCase()} · ${g.modalities[br]}${pf ? `<span>per frame × ${pf}, shared weights</span>` : ""}</div>`,
        V(-12.55, 0.16, LANE[br]),
        "center",
        "ground",
      );
    }
    this.zoneTitle("intervention", "intervention", "Feature-scale intervention", "experiments only · α = 1 here", V(3.7, 0.14, front));
    this.zoneTitle("fusion", "fusion", "Fusion 𝓕", `${g.fusion} in this model · ${g.params.fusion} params`, V(6.0, 0.14, zb + 2.15));
    this.zoneTitle("classifier", "classifier", "Classifier", `${head.params.toLocaleString("en-US")} params`, V(8.2, 0.14, 1.95));
    this.zoneTitle("output", "output", "Output ŷ", `${stageInfo(g, "output").detail}`, V(10.75, 0.14, 2.65));
    this.enter(null);
  }

  /* one modality lane: inputs → preprocessing → encoder ----------------- */

  private buildLane(g: ArchGraph, br: "a" | "b") {
    const z = LANE[br];
    const mod = g.modalities[br];
    const col = br === "a" ? this.theme.a : this.theme.b;
    const [raw, mid, fin] = g.nodes.filter((n) => n.branch === br && n.segment === "pipeline");

    // raw input
    this.enter("inputs");
    const rawObjs = this.rawObject(mod, z, raw.id, col);
    this.register(raw.id, rawObjs);
    this.anchors.set(raw.id, { in: V(X.raw - 0.9, Y, z), out: V(X.raw + 0.92, Y, z) });
    const rawSub = mod === "audio" || mod === "video" ? "recording (variable length)" : mod === "depth" ? "depth video (H × W × T)" : "accelerometer + gyroscope (T × 6)";
    this.chip(raw.id, `${cap(mod)}`, rawSub, V(X.raw, 1.42, z));

    // preprocessing step (capsule = deterministic, not learned)
    this.enter("preprocessing");
    const cap3 = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.7, 6, 18), this.std(this.theme.card, { roughness: 0.25, opacity: 0.95 }));
    cap3.rotation.z = Math.PI / 2;
    cap3.position.set(X.mid, Y, z);
    this.put(cap3, mid.id);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.022, 8, 40), this.std(col));
    ring.rotation.y = Math.PI / 2;
    ring.position.set(X.mid, Y, z);
    this.put(ring, mid.id);
    this.register(mid.id, [cap3]);
    this.anchors.set(mid.id, { in: V(X.mid - 0.58, Y, z), out: V(X.mid + 0.58, Y, z) });
    this.chip(mid.id, mid.short, mid.out_shape ? fmtShape(mid.out_shape) : "", V(X.mid, 1.06, z));

    // model input tensor (real preprocessing output)
    const finObjs = this.finObject(mod, z, fin.id);
    this.register(fin.id, finObjs);
    this.anchors.set(fin.id, { in: V(X.fin - 0.95, Y, z), out: V(X.fin + 0.95, Y, z) });
    this.chip(fin.id, fin.short, fmtShape(fin.out_shape), V(X.fin, 1.42, z));

    this.buildEncoder(g, br, col);
    this.enter(null);
  }

  private rawObject(mod: string, z: number, id: string, col: string): THREE.Object3D[] {
    const T = this.theme;
    const out: THREE.Object3D[] = [];
    const base = this.put(new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.06, 0.62), this.std(T.card)), id);
    base.position.set(X.raw, 0.16, z);
    out.push(base);
    if (mod === "audio") {
      // envelope of the real (TTS) speech sample used for the log-mel thumbnail
      const env = media.audio_envelope as number[];
      const bars = new THREE.InstancedMesh(new THREE.BoxGeometry(0.017, 1, 0.12), this.std(col), env.length);
      env.forEach((v, i) => {
        const h = Math.max(0.03, Math.abs(v)) * 0.82;
        this.dummy.position.set(X.raw - 0.8 + (i / (env.length - 1)) * 1.6, Y, z);
        this.dummy.scale.set(1, h, 1);
        this.dummy.updateMatrix();
        bars.setMatrixAt(i, this.dummy.matrix);
      });
      this.put(bars, id);
      out.push(bars);
    } else if (mod === "video") {
      // generic film-strip icon (the raw colour video is not shown)
      const map = this.canvasTex(512, 220, (c) => {
        c.fillStyle = "#2c2f36";
        c.fillRect(0, 0, 512, 220);
        c.fillStyle = "#f2efe9";
        for (let i = 0; i < 12; i++) {
          c.fillRect(14 + i * 42, 12, 22, 18);
          c.fillRect(14 + i * 42, 190, 22, 18);
        }
        c.fillStyle = "#4b5160";
        for (let i = 0; i < 4; i++) c.fillRect(16 + i * 124, 42, 108, 136);
        c.fillStyle = "#f2efe9";
        c.beginPath();
        c.moveTo(226, 72);
        c.lineTo(300, 110);
        c.lineTo(226, 148);
        c.fill();
      });
      out.push(this.card(map, 1.55, 0.67, V(X.raw, Y, z), id, -0.3));
    } else if (mod === "depth") {
      // three raw depth frames of UTD-MHAD test trial a3_s4_t2
      for (let i = 0; i < 3; i++) out.push(this.card(this.frameTex("utd_depth_raw.png", i, 3), 0.86, 0.645, V(X.raw - 0.22 + i * 0.22, Y + 0.05 - i * 0.05, z - 0.18 + i * 0.18), id, -0.3));
    } else {
      // raw inertial channels (same trial), one ribbon per channel
      (media.inertial_raw as number[][]).forEach((ch, ci) => {
        const lo = Math.min(...ch);
        const hi = Math.max(...ch);
        const pts = ch.map((v, i) => V(X.raw - 0.8 + (i / (ch.length - 1)) * 1.6, Y - 0.32 + ((v - lo) / (hi - lo || 1)) * 0.64, z - 0.25 + ci * 0.1));
        const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 160, 0.012, 5), this.std(WAVE[ci % WAVE.length]));
        this.put(tube, id);
        out.push(tube);
      });
    }
    return out;
  }

  private finObject(mod: string, z: number, id: string): THREE.Object3D[] {
    if (mod === "audio") return [this.card(this.image("cremad_logmel.png"), 1.65, 0.56, V(X.fin, Y, z), id)];
    if (mod === "video" || mod === "depth") {
      const [file, n, s] = mod === "video" ? ["cremad_frames.png", 8, 0.5] : ["utd_depth.png", 16, 0.44];
      const out: THREE.Object3D[] = [];
      const step = mod === "video" ? 0.12 : 0.065;
      for (let i = 0; i < (n as number); i++) {
        const k = i - ((n as number) - 1) / 2;
        out.push(this.card(this.frameTex(file as string, i, n as number, mod === "depth"), s as number, s as number, V(X.fin + k * step, Y, z - k * step * 0.55), id, -0.55));
      }
      return out;
    }
    // standardized inertial tensor [6, 128] as a heat map (real values, same trial)
    const zc = media.inertial_z as number[][];
    const map = this.canvasTex(512, 192, (c) => {
      const cw = 512 / zc[0].length;
      const rh = 192 / zc.length;
      zc.forEach((row, r) =>
        row.forEach((v, i) => {
          const t = Math.max(-1, Math.min(1, v / 2.5));
          const a = t < 0 ? [59, 110, 165] : [207, 122, 62];
          const w = Math.abs(t);
          c.fillStyle = `rgb(${Math.round(250 + (a[0] - 250) * w)},${Math.round(248 + (a[1] - 248) * w)},${Math.round(244 + (a[2] - 244) * w)})`;
          c.fillRect(i * cw, r * rh, cw + 0.6, rh + 0.6);
        }),
      );
    });
    return [this.card(map, 1.65, 0.62, V(X.fin, Y, z), id)];
  }

  private buildEncoder(g: ArchGraph, br: "a" | "b", col: string) {
    this.enter(br === "a" ? "enc_a" : "enc_b");
    const z = LANE[br];
    const nodes = g.nodes.filter((n) => n.branch === br && n.segment === "encoder");
    const NAMES: Record<string, string> = { block1: "Conv 1", block2: "Conv 2", block3: "Conv 3", gap: "GAP", tmean: "mean over t", fc: `FC → h<sub>${br.toUpperCase()}</sub>` };
    for (const n of nodes) {
      const key = keyOf(n.id);
      const cx = X[key as keyof typeof X];
      const s = n.out_shape!;
      const objs: THREE.Object3D[] = [];
      let top = Y + 0.4;
      if (key.startsWith("block")) {
        // feature-map stack: plates ∝ log2(channels); plate size ∝ log2(map size)
        const plates = Math.max(3, Math.round(Math.log2(s[0])) - 1);
        const is2d = s.length === 3;
        const hh = 0.24 + 0.1 * Math.log2(s[1]);
        const dd = is2d ? 0.24 + 0.1 * Math.log2(s[2]) : 0.22;
        const dx = 0.11;
        const copies = n.per_frame ? 3 : 1; // per-frame encoders: shared weights, one copy per frame
        for (let c = copies - 1; c >= 0; c--) {
          for (let i = 0; i < plates; i++) {
            const geo = new THREE.BoxGeometry(0.06, hh, dd);
            const shade = this.tint(col, 0.06 + 0.5 * (i / (plates - 1)));
            const m = new THREE.Mesh(geo, this.std(shade, { opacity: c === 0 ? 1 : 0.22, roughness: 0.4 }));
            m.position.set(cx - ((plates - 1) / 2) * dx + i * dx + c * 0.07, Y + c * 0.07, z - c * 0.16 * Math.sign(z));
            this.put(m, n.id);
            if (c === 0) {
              objs.push(m);
              const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.45 }));
              e.position.copy(m.position);
              this.cur.add(e);
            }
          }
        }
        top = Y + hh / 2;
        const half = ((plates - 1) / 2) * dx + 0.05;
        this.anchors.set(n.id, { in: V(cx - half, Y, z), out: V(cx + half, Y, z) });
      } else if (key === "gap" || key === "tmean") {
        // pooling funnel: wide input side, narrow output side
        const big = key === "gap" ? 0.33 : 0.26;
        const geo = new THREE.CylinderGeometry(big, 0.09, 0.42, 28, 1, true);
        geo.rotateZ(Math.PI / 2); // +y (wide end) → −x
        const m = new THREE.Mesh(geo, this.std(this.tint(col, key === "gap" ? 0.18 : 0.32), { side: THREE.DoubleSide, roughness: 0.35 }));
        m.position.set(cx, Y, z);
        this.put(m, n.id);
        objs.push(m);
        top = Y + big;
        this.anchors.set(n.id, { in: V(cx - 0.22, Y, z), out: V(cx + 0.22, Y, z) });
      } else {
        // FC slab + the h vector it produces (column of cells, 64-d)
        const slab = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.82, 0.56), this.std(this.tint(col, 0.02), { roughness: 0.35 }));
        slab.position.set(cx, Y, z);
        this.put(slab, n.id);
        objs.push(slab);
        for (let i = 0; i < 8; i++) {
          const cell = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.095, 0.15), this.std(this.tint(col, 0.15 + (i % 2) * 0.12), { roughness: 0.3 }));
          cell.position.set(X.h, Y - 0.39 + i * 0.112, z);
          this.put(cell, n.id);
          objs.push(cell);
        }
        top = Y + 0.46;
        this.anchors.set(n.id, { in: V(cx - 0.06, Y, z), out: V(cx + 0.06, Y, z) });
        this.label(`<div class="a3-glyph-h" data-node="${n.id}"><i>h</i><sub>${br.toUpperCase()}</sub></div>`, V(X.h, Y + 0.56, z));
      }
      this.register(n.id, objs);
      const shape = `${n.per_frame ? `${n.per_frame}× ` : ""}${fmtShape(s)}`;
      this.chip(n.id, NAMES[key], shape, V(cx, top + 0.24, z));
    }
  }

  /* intervention: α on the branch the experiments scaled, 1 on the other ------------ */

  private buildIntervention(g: ArchGraph) {
    this.enter("intervention");
    const T = this.theme;
    const objs: THREE.Object3D[] = [];
    for (const br of ["a", "b"] as const) {
      const z = LANE[br];
      const scaled = br === g.scaledBranch;
      const r = scaled ? 0.3 : 0.25;
      const s = this.put(new THREE.Mesh(new THREE.SphereGeometry(r, 40, 28), this.std(T.sphere, { roughness: 0.28 })), "concept:intervention");
      s.position.set(X.alpha, Y, z);
      const x = this.put(new THREE.Mesh(new THREE.SphereGeometry(0.2, 32, 22), this.std("#f7f3ee", { roughness: 0.35 })), "concept:intervention");
      x.position.set(X.times, Y, z);
      objs.push(s, x);
      this.label(`<div class="a3-glyph">${scaled ? "α" : "1"}</div>`, V(X.alpha, Y, z), "center", "glyph");
      this.label(`<div class="a3-glyph a3-glyph--dark">×</div>`, V(X.times, Y, z), "center", "glyph");
    }
    // dashed intervention link between the α and 1 spheres
    const a = V(X.alpha, Y, LANE.a + 0.36);
    const b = V(X.alpha, Y, LANE.b - 0.32);
    const mat = new THREE.MeshBasicMaterial({ color: T.intInk });
    for (let i = 0; i < 15; i += 2) {
      const seg = new THREE.LineCurve3(a.clone().lerp(b, i / 15), a.clone().lerp(b, (i + 1) / 15));
      objs.push(this.put(new THREE.Mesh(new THREE.TubeGeometry(seg, 1, 0.016, 6), mat)));
    }
    for (const [p, dir] of [
      [a, -1],
      [b, 1],
    ] as [THREE.Vector3, number][]) {
      const cone = this.put(new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.13, 12), mat));
      cone.position.copy(p);
      cone.rotation.x = (dir * Math.PI) / 2;
      objs.push(cone);
    }
    this.register("concept:intervention", objs, 0.06);
    const scaledName = g.modalities[g.scaledBranch];
    this.label(
      `<div class="a3-chip a3-chip--int" data-node="concept:intervention"><span class="a3-chip-tag">experiments only</span><span class="a3-chip-t">α scales ${scaledName}</span><span class="a3-chip-s">× 1 on ${g.modalities[g.scaledBranch === "a" ? "b" : "a"]}</span></div>`,
      V(X.alpha + 0.5, Y + 0.62, 0),
    );
  }

  /* fusion: Sum (this model) + the two alternatives tested only in the synthetic experiment */

  private buildFusion(g: ArchGraph) {
    this.enter("fusion");
    const T = this.theme;
    const col = T.fusInk;
    const objs: THREE.Object3D[] = [];
    const ring = this.put(new THREE.Mesh(new THREE.TorusGeometry(0.46, 0.075, 18, 56), this.std(col, { roughness: 0.3 })), "fusion");
    ring.rotation.y = Math.PI / 2;
    ring.position.set(X.fusion, Y, 0);
    objs.push(ring);
    for (const rot of [0, Math.PI / 2]) {
      const bar = this.put(new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.62, 16), this.std(col, { roughness: 0.3 })), "fusion");
      bar.position.set(X.fusion, Y, 0);
      bar.rotation.x = rot;
      objs.push(bar);
    }
    const core = this.put(new THREE.Mesh(new THREE.SphereGeometry(0.34, 32, 24), this.std(this.tint(col, 0.55), { opacity: 0.35, roughness: 0.1 })), "fusion");
    core.position.set(X.fusion, Y, 0);
    objs.push(core);
    this.register("fusion", objs);
    this.anchors.set("fusion", { in: V(X.fusion - 0.5, Y, 0), out: V(X.fusion + 0.5, Y, 0) });
    const f = nodeShape(g, "fusion");
    this.chip("fusion", `${cap(g.fusion)} fusion · <i>h</i><sub>A</sub> + <i>h</i><sub>B</sub>`, f, V(X.fusion, Y + 0.68, 0));

    // alternatives: ghosted, tagged, never connected to the data flow
    const alt = (id: string, z: number, name: string, formula: string, build: () => THREE.Mesh[]) => {
      const ms = build();
      ms.forEach((m) => {
        m.position.z += z;
        this.put(m, id);
      });
      this.register(id, ms, 0.12);
      this.label(
        `<div class="a3-chip a3-chip--ghost" data-node="${id}"><span class="a3-chip-tag">synthetic exp. only</span><span class="a3-chip-t">${name}</span><span class="a3-chip-s">${formula}</span></div>`,
        V(X.fusion, Y + 0.48, z),
      );
    };
    const ghost = () => this.std(col, { opacity: 0.38, roughness: 0.3 });
    alt("concept:fusion_concat", LANE.a - 1.2, "Concat", "[h<sub>A</sub> ‖ h<sub>B</sub>]", () =>
      [-0.13, 0.13].map((dz) => {
        const m = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.66, 0.2), ghost());
        m.position.set(X.fusion, Y, dz);
        return m;
      }),
    );
    alt("concept:fusion_gated", LANE.b + 1.2, "Gated", "g<sub>A</sub>⊙h<sub>A</sub> + g<sub>B</sub>⊙h<sub>B</sub>", () => {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.26, 28, 20), ghost());
      s.position.set(X.fusion, Y, 0);
      const valves = [-0.36, 0.36].map((dz) => {
        const v = new THREE.Mesh(new THREE.TorusGeometry(0.12, 0.035, 10, 28), ghost());
        v.position.set(X.fusion, Y, dz);
        return v;
      });
      return [s, ...valves];
    });
  }

  /* classifier: one linear layer (no hidden layer) ---------------------------- */

  private buildClassifier(g: ArchGraph) {
    this.enter("classifier");
    const head = g.nodes.find((n) => n.id === "head")!;
    const col = this.theme.arrow;
    const objs: THREE.Object3D[] = [];
    const ins: THREE.Vector3[] = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) ins.push(V(X.head - 0.55, Y - 0.38 + r * 0.38, -0.38 + c * 0.38));
    const nOut = Math.min(g.nClasses, 9);
    const rows = 3;
    const colsOut = Math.ceil(nOut / rows);
    const outs: THREE.Vector3[] = [];
    for (let i = 0; i < nOut; i++) {
      const r = i % rows;
      const c = Math.floor(i / rows);
      outs.push(V(X.head + 0.55, Y - 0.38 + r * 0.38, (c - (colsOut - 1) / 2) * 0.38));
    }
    for (const p of ins) {
      const s = this.put(new THREE.Mesh(new THREE.SphereGeometry(0.07, 18, 12), this.std(this.theme.fusInk, { roughness: 0.35 })), "head");
      s.position.copy(p);
      objs.push(s);
    }
    for (const p of outs) {
      const s = this.put(new THREE.Mesh(new THREE.SphereGeometry(0.08, 18, 12), this.std(col, { roughness: 0.35 })), "head");
      s.position.copy(p);
      objs.push(s);
    }
    const pos: number[] = [];
    for (const a of ins) for (const b of outs) pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    this.cur.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.35 })));
    this.register("head", objs, 0.14);
    this.anchors.set("head", { in: V(X.head - 0.7, Y, 0), out: V(X.head + 0.7, Y, 0) });
    const schematic = g.nClasses > nOut ? ` · ${nOut} of ${g.nClasses} units drawn` : "";
    this.chip("head", `Linear ${head.in_shape![0]} → ${head.out_shape![0]}`, `no hidden layer${schematic}`, V(X.head, Y + 0.62, 0));
  }

  /* output: softmax → class tiles (codes only; no scores are shown) ------------ */

  private buildOutput(g: ArchGraph) {
    this.enter("output");
    const T = this.theme;
    const objs: THREE.Object3D[] = [];
    const disc = this.put(new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.05, 48), this.std(T.chip, { opacity: 0.85, roughness: 0.25 })), "output");
    disc.rotation.z = Math.PI / 2;
    disc.position.set(X.output - 0.55, Y, 0);
    objs.push(disc);
    this.anchors.set("output", { in: V(X.output - 0.6, Y, 0), out: V(X.output - 0.5, Y, 0) });
    const n = g.nClasses;
    const rows = n > 9 ? 3 : 1;
    const perRow = Math.ceil(n / rows);
    const spread = THREE.MathUtils.degToRad(n > 9 ? 120 : 110);
    const R = n > 9 ? 1.55 : 1.35;
    const tw = n > 9 ? 0.3 : 0.5;
    const th = n > 9 ? 0.24 : 0.32;
    const cx = X.output - 0.55;
    g.classCodes.forEach((code, i) => {
      const r = Math.floor(i / perRow);
      const k = i % perRow;
      const ang = -spread / 2 + (perRow === 1 ? 0.5 : k / (perRow - 1)) * spread;
      const map = this.canvasTex(256, 192, (c) => {
        c.fillStyle = T.card;
        c.fillRect(0, 0, 256, 192);
        c.strokeStyle = T.chip;
        c.lineWidth = 14;
        c.strokeRect(7, 7, 242, 178);
        c.fillStyle = T.ink;
        c.font = `700 ${code.length > 3 ? 70 : 84}px Inter, system-ui, sans-serif`;
        c.textAlign = "center";
        c.textBaseline = "middle";
        c.fillText(code, 128, 100);
      });
      const tile = this.put(new THREE.Mesh(new THREE.BoxGeometry(tw, th, 0.04), [null, null, null, null, 0, 0].map((x) => (x === null ? this.std(T.card) : new THREE.MeshBasicMaterial({ map, transparent: true }))) as THREE.Material[]), "output");
      tile.position.set(cx + Math.cos(ang) * R, Y + (rows === 1 ? 0 : (1 - r) * (th + 0.08)), Math.sin(ang) * R);
      tile.lookAt(cx - 2, tile.position.y, 0);
      tile.rotateY(Math.PI);
      objs.push(tile);
    });
    this.register("output", objs, 0.1);
    this.chip("output", "Softmax → ŷ", `${n} class probabilities${g.classNamesAvailable ? "" : " · codes only"}`, V(X.output + 0.25, Y + (rows === 1 ? 0.6 : 0.85), 0));
  }

  /* connections + packets --------------------------------------------------- */

  private liftOf(sec: SectionKey) {
    return this.sections.get(sec)?.lift ?? 0;
  }

  private buildEdges(g: ArchGraph) {
    this.enter(null); // connections are rebuilt from the sections' current lift
    const T = this.theme;
    const S = (id: string) => sectionOf(g, id)!;
    for (const e of g.edges) {
      const key = `${e.from}->${e.to}`;
      const from = g.nodes.find((n) => n.id === e.from)!;
      const color = new THREE.Color(from.branch === "a" ? T.a : from.branch === "b" ? T.b : T.fusInk);
      let spec: EdgeSpec;
      if (e.to === "fusion") {
        // FC → h → (α or 1) → × → fusion, as in the experiments' scale operator position
        const z = LANE[from.branch as "a" | "b"];
        const es = S(e.from);
        spec = {
          kind: "fusion",
          pts: [
            { p: this.anchors.get(e.from)!.out, s: es },
            { p: V(X.h - 0.09, Y, z), s: es },
            { p: V(X.h + 0.09, Y, z), s: es },
            { p: V(X.alpha - 0.3, Y, z), s: "intervention" },
            { p: V(X.alpha + 0.3, Y, z), s: "intervention" },
            { p: V(X.times - 0.2, Y, z), s: "intervention" },
            { p: V(X.times + 0.2, Y, z), s: "intervention" },
            { p: V(X.fusion - 0.5, Y, z * 0.16), s: "fusion" },
          ],
        };
      } else {
        spec = {
          kind: "line",
          pts: [
            { p: this.anchors.get(e.from)!.out, s: S(e.from) },
            { p: this.anchors.get(e.to)!.in, s: S(e.to) },
          ],
        };
      }
      const tip = new THREE.Mesh(new THREE.ConeGeometry(0.055, 0.14, 12), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 }));
      this.objs.add(tip);
      const rec: EdgeRec = { key, spec, secs: new Set(spec.pts.map((x) => x.s)), curve: new THREE.LineCurve3(), len: 1, tubes: [], tip, color, first: 0, n: 2 };
      this.layoutEdge(rec);
      rec.n = Math.max(2, Math.round(rec.len / 0.8));
      this.edges.push(rec);
      this.edgeByKey.set(key, rec);
    }
  }

  /** (Re)builds an edge's curve, tubes and arrow tip from the current section lifts. */
  private layoutEdge(e: EdgeRec) {
    const P = e.spec.pts.map(({ p, s: sec }) => V(p.x, p.y + this.liftOf(sec), p.z));
    const L = (a: THREE.Vector3, b: THREE.Vector3) => new THREE.LineCurve3(a, b);
    let visible: THREE.Curve<THREE.Vector3>[];
    if (e.spec.kind === "line") {
      e.curve = L(P[0], P[1]);
      visible = [e.curve];
    } else {
      const [p0, p1, p2, p3, p4, p5, p6, p7] = P;
      const bend = new THREE.QuadraticBezierCurve3(p6, V(X.fusion - 1.05, (p6.y + p7.y) / 2, p6.z * 0.92), p7);
      const path = new THREE.CurvePath<THREE.Vector3>();
      [L(p0, p1), L(p1, p2), L(p2, p3), L(p3, p4), L(p4, p5), L(p5, p6), bend].forEach((c) => path.add(c));
      e.curve = path;
      visible = [L(p0, p1), L(p2, p3), L(p4, p5), bend];
    }
    visible.forEach((c, i) => {
      const geo = new THREE.TubeGeometry(c, c instanceof THREE.LineCurve3 ? 1 : 24, 0.02, 8);
      const m = e.tubes[i];
      if (m) {
        m.geometry.dispose();
        m.geometry = geo;
      } else {
        const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: e.color, transparent: true, opacity: 0.5 }));
        this.objs.add(mesh);
        e.tubes.push(mesh);
      }
    });
    const tan = e.curve.getTangent(1).normalize();
    e.tip.position.copy(e.curve.getPoint(1)).addScaledVector(tan, -0.07);
    e.tip.quaternion.setFromUnitVectors(V(0, 1, 0), tan);
    e.len = e.curve.getLength();
  }

  /** Eases every section towards its target height; reshapes the connections that touch it. */
  private stepLifts(dt: number) {
    const changed = new Set<SectionKey>();
    for (const [k, sec] of this.sections) {
      const d = sec.target - sec.lift;
      if (Math.abs(d) < 1e-4) continue;
      sec.lift = dt <= 0 || Math.abs(d) < 0.004 ? sec.target : sec.lift + d * Math.min(1, dt * 7);
      sec.group.position.y = sec.lift;
      changed.add(k);
    }
    if (changed.size) for (const e of this.edges) if ([...e.secs].some((x) => changed.has(x))) this.layoutEdge(e);
  }

  private buildPackets() {
    let total = 0;
    for (const e of this.edges) {
      e.first = total;
      total += e.n;
    }
    const geo = new THREE.BoxGeometry(0.1, 0.1, 0.1);
    this.packets = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color: 0xffffff }), total);
    this.halos = new THREE.InstancedMesh(new THREE.SphereGeometry(0.11, 14, 10), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28, depthWrite: false }), total);
    this.packets.frustumCulled = this.halos.frustumCulled = false;
    this.objs.add(this.packets, this.halos);
    let ov = 0;
    for (const p of this.ovPaths) {
      p.first = ov;
      ov += p.n;
    }
    if (ov) {
      this.ovPackets = new THREE.InstancedMesh(new THREE.OctahedronGeometry(0.085), new THREE.MeshBasicMaterial({ color: 0xffffff }), ov);
      this.ovPackets.frustumCulled = false;
      this.overlayGroup.add(this.ovPackets);
      this.ovPaths.forEach((p, k) => {
        this.tmpC.set(k === 0 ? this.theme.forward : k === 1 ? this.theme.gradient : this.theme.ink);
        for (let i = 0; i < p.n; i++) this.ovPackets!.setColorAt(p.first + i, this.tmpC);
      });
    }
    this.updatePackets(0);
  }

  /** Positions every packet at time `t` (seconds). Static when motion is reduced. */
  private updatePackets(t: number) {
    if (!this.packets || !this.halos) return;
    const { path, flowEdge } = this.state;
    const ink = new THREE.Color(this.theme.ink);
    for (const e of this.edges) {
      const on = !path || path.edges.has(e.key);
      const hot = e.key === flowEdge;
      for (let i = 0; i < e.n; i++) {
        const u = this.reduceMotion ? (i + 0.5) / e.n : (t * (hot ? 1.15 : 0.75) / e.len + i / e.n) % 1;
        const p = e.curve.getPointAt(u);
        const s = !on ? 0 : hot ? 1.45 : flowEdge ? 0.75 : 1;
        this.dummy.position.copy(p);
        this.dummy.rotation.set(t * 1.3 + i, t * 0.9 + i * 0.5, 0);
        this.dummy.scale.setScalar(s);
        this.dummy.updateMatrix();
        this.packets.setMatrixAt(e.first + i, this.dummy.matrix);
        this.dummy.scale.setScalar(s * (1 + 0.18 * Math.sin(t * 4 + i)));
        this.dummy.updateMatrix();
        this.halos.setMatrixAt(e.first + i, this.dummy.matrix);
        this.tmpC.copy(hot ? ink : e.color);
        if (!hot) this.tmpC.lerp(new THREE.Color(this.theme.dark ? "#ffffff" : "#000000"), 0.12);
        this.packets.setColorAt(e.first + i, this.tmpC);
        this.halos.setColorAt(e.first + i, hot ? ink : e.color);
      }
    }
    this.packets.instanceMatrix.needsUpdate = this.halos.instanceMatrix.needsUpdate = true;
    if (this.packets.instanceColor) this.packets.instanceColor.needsUpdate = true;
    if (this.halos.instanceColor) this.halos.instanceColor.needsUpdate = true;
    if (this.ovPackets && this.state.overlay) {
      for (const p of this.ovPaths) {
        const len = p.curve.getLength();
        for (let i = 0; i < p.n; i++) {
          const u = this.reduceMotion ? (i + 0.5) / p.n : (t * p.speed / len + i / p.n) % 1;
          this.dummy.position.copy(p.curve.getPointAt(u));
          this.dummy.rotation.set(t, t * 1.4, 0);
          this.dummy.scale.setScalar(1);
          this.dummy.updateMatrix();
          this.ovPackets.setMatrixAt(p.first + i, this.dummy.matrix);
        }
      }
      this.ovPackets.instanceMatrix.needsUpdate = true;
    }
  }

  /* research components that are not layers: floating, clickable markers ------ */

  private buildBeacons(g: ArchGraph) {
    const T = this.theme;
    const zs = LANE[g.scaledBranch];
    const make = (id: string, geo: THREE.BufferGeometry, color: string, pos: THREE.Vector3, tag: string, title: string) => {
      this.enter(sectionOf(g, id));
      const m = this.put(new THREE.Mesh(geo, this.std(color, { roughness: 0.25, metalness: 0.15 })), id);
      m.position.copy(pos);
      this.beacons.push({ mesh: m, y: pos.y, phase: this.beacons.length * 1.7 });
      const stem = new THREE.Mesh(
        new THREE.CylinderGeometry(0.008, 0.008, pos.y - 0.12, 6),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.45 }),
      );
      stem.position.set(pos.x, (pos.y + 0.12) / 2, pos.z);
      this.cur.add(stem);
      this.register(id, [m], 0.22);
      this.label(
        `<div class="a3-chip a3-chip--beacon" data-node="${id}"><span class="a3-chip-tag">${tag}</span><span class="a3-chip-t">${title}</span></div>`,
        V(pos.x, pos.y + 0.34, pos.z),
      );
    };
    make("concept:variants", new THREE.OctahedronGeometry(0.24), T.sphere, V(X.alpha + 0.5, 2.25, 0), "experiments only", "Intervention variants");
    make("concept:optimizer", new THREE.TorusKnotGeometry(0.15, 0.05, 80, 10), T.chip, V(X.block2, 2.45, zs), "analysis", "Optimizer analysis · Adam vs SGD");
    make("concept:utility", new THREE.DodecahedronGeometry(0.22), T.fusInk, V(X.fusion + 0.35, 2.3, 1.25), "analysis", "Fusion utility metrics");
    this.enter(null);
  }

  /* conceptual research overlay (forward / gradient / update pathways) --------- */

  private buildOverlay(g: ArchGraph) {
    const T = this.theme;
    const zs = LANE[g.scaledBranch];
    const tag = (text: string, pos: THREE.Vector3, tone: string) =>
      this.label(`<div class="a3-chip a3-chip--path a3-chip--${tone}"><span class="a3-chip-tag">concept</span><span class="a3-chip-t">${text}</span></div>`, pos, "bottom", "overlay");
    const path = (pts: THREE.Vector3[], color: string, dash: boolean) => {
      const curve = new THREE.CatmullRomCurve3(pts, false, "catmullrom", 0.35);
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85 });
      const n = dash ? 46 : 1;
      for (let i = 0; i < n; i++) {
        if (dash && i % 2) continue;
        const seg = dash ? new THREE.LineCurve3(curve.getPointAt(i / n), curve.getPointAt((i + 1) / n)) : curve;
        this.overlayGroup.add(new THREE.Mesh(new THREE.TubeGeometry(seg, dash ? 1 : 140, 0.026, 8), mat));
      }
      const cone = new THREE.Mesh(new THREE.ConeGeometry(0.075, 0.2, 14), mat);
      cone.position.copy(curve.getPoint(1));
      cone.quaternion.setFromUnitVectors(V(0, 1, 0), curve.getTangent(1).normalize());
      this.overlayGroup.add(cone);
      return curve;
    };
    // forward pathway: the scaled magnitude reaches the sum, the logits and the softmax
    const fwd = path([V(X.alpha, Y + 0.35, zs), V(X.alpha + 0.9, 1.75, zs * 0.8), V(X.fusion, 1.95, zs * 0.25), V(X.head, 1.85, 0), V(X.output - 0.55, Y + 0.6, 0)], T.forward, false);
    tag("forward pathway", V(X.fusion + 1.1, 2.05, zs * 0.15), "forward");
    // gradient pathway: from the loss back through the sum and α into the scaled encoder
    const grad = path(
      [V(X.output - 0.55, Y + 0.9, 0), V(X.head, 2.65, 0), V(X.fusion, 2.85, zs * 0.4), V(X.alpha, 2.55, zs), V(X.block3, 2.05, zs), V(X.block1, Y + 0.62, zs)],
      T.gradient,
      true,
    );
    tag("gradient pathway", V(X.fusion - 0.9, 2.95, zs * 0.55), "gradient");
    // update pathway: the optimizer turns gradients into weight updates of the scaled encoder
    const upd = new THREE.CatmullRomCurve3([V(X.block2, 2.3, zs), V(X.block2 - 0.55, 1.6, zs * 1.15), V(X.block1, Y + 0.55, zs * 1.05)]);
    const dotMat = new THREE.MeshBasicMaterial({ color: T.ink });
    for (let i = 0; i <= 22; i++) {
      const d = new THREE.Mesh(new THREE.SphereGeometry(0.024, 8, 6), dotMat);
      d.position.copy(upd.getPointAt(i / 22));
      this.overlayGroup.add(d);
    }
    tag("update pathway (optimizer)", V(X.block2 - 1.05, 1.65, zs * 1.2), "ink");
    this.ovPaths = [
      { curve: fwd, n: 7, first: 0, speed: 1.0 },
      { curve: grad, n: 9, first: 0, speed: 1.0 },
      { curve: upd, n: 3, first: 0, speed: 0.6 },
    ];
  }

  /* ── state ─────────────────────────────────────────────────────── */

  private applyState() {
    if (!this.graph) return;
    const { selected, path, flowNode, flowEdge, overlay, raised } = this.state;
    const T = this.theme;
    const active = new Set([selected, flowNode, this.hoverId].filter(Boolean) as string[]);
    for (const [k, sec] of this.sections) sec.target = k === raised ? LIFT : 0;
    if (this.reduceMotion) this.stepLifts(0);
    for (const l of this.labels) {
      l.hidden = l.kind === "overlay" ? !overlay : l.kind === "node" ? l.section !== raised : false;
      l.obj.visible = !l.hidden;
    }
    for (const [id, v] of this.vis) {
      const onPath = !path || path.nodes.has(id) || (id.startsWith("concept:") && !flowNode);
      const on = active.has(id);
      for (const m of v.meshes) {
        const mat = m.material as THREE.MeshStandardMaterial | THREE.MeshBasicMaterial | THREE.Material[];
        for (const mm of Array.isArray(mat) ? mat : [mat]) {
          if (mm instanceof THREE.MeshStandardMaterial) {
            mm.emissive.set(on ? T.ink : "#000000");
            mm.emissiveIntensity = on ? (T.dark ? 0.35 : 0.22) : 0;
          }
          mm.opacity = (onPath ? 1 : 0.14) * (m.userData.baseOpacity ?? 1);
        }
      }
    }
    for (const e of this.edges) {
      const onPath = !path || path.edges.has(e.key);
      for (const m of [...e.tubes, e.tip]) {
        const mm = m.material as THREE.MeshBasicMaterial;
        mm.color.set(e.key === flowEdge ? T.ink : e.color);
        mm.opacity = onPath ? (e.key === flowEdge ? 0.95 : 0.5) : 0.08;
      }
    }
    const stage = stageOf(this.graph, flowNode ?? selected);
    const hoverSec = sectionOf(this.graph, this.hoverId);
    for (const z of this.zones) {
      const mat = z.mesh.material as THREE.MeshStandardMaterial;
      mat.emissive.set(T.ink);
      const on = z.section === raised || (!raised && z.stage === stage);
      mat.emissiveIntensity = on ? (T.dark ? 0.13 : 0.07) : z.section === hoverSec ? (T.dark ? 0.07 : 0.035) : 0;
    }
    const root = this.css.domElement;
    root.querySelectorAll<HTMLElement>("[data-node]").forEach((el) => {
      const id = el.getAttribute("data-node")!;
      el.classList.toggle("is-active", active.has(id));
      el.classList.toggle("is-dim", !!path && !id.startsWith("concept:") && !path.nodes.has(id));
    });
    root.querySelectorAll<HTMLElement>("[data-stage]").forEach((el) => el.classList.toggle("is-active", el.getAttribute("data-stage") === stage));
    this.overlayGroup.visible = overlay;
    if (this.reduceMotion) this.updatePackets(0);
    this.kick();
    this.requestRender();
  }

  /* ── animation loop (paused when off-screen or the tab is hidden) ───────── */

  private kick() {
    if (this.reduceMotion || this.disposed || this.running || !this.inView || document.hidden || !this.graph) return;
    this.running = true;
    this.lastT = performance.now();
    requestAnimationFrame(this.loop);
  }

  private loop = (now: number) => {
    if (this.disposed || !this.inView || document.hidden) {
      this.running = false;
      return;
    }
    const dt = Math.min(0.1, (now - this.lastT) / 1000);
    this.lastT = now;
    if (this.flight) {
      const f = this.flight;
      const p = ease(Math.min(1, (now - f.t0) / f.dur));
      this.controls.target.lerpVectors(f.fromT, f.toT, p);
      const s = new THREE.Spherical(
        f.fromS.radius * Math.pow(f.toS.radius / f.fromS.radius, p),
        THREE.MathUtils.lerp(f.fromS.phi, f.toS.phi, p),
        THREE.MathUtils.lerp(f.fromS.theta, f.toS.theta, p),
      );
      this.camera.position.copy(this.controls.target).add(new THREE.Vector3().setFromSpherical(s));
      if (p >= 1) this.flight = null;
    }
    this.controls.update(dt);
    this.stepLifts(dt);
    const t = now / 1000;
    this.updatePackets(t);
    for (const b of this.beacons) {
      b.mesh.position.y = b.y + Math.sin(t * 1.6 + b.phase) * 0.07;
      b.mesh.rotation.y = t * 0.8 + b.phase;
    }
    this.render();
    requestAnimationFrame(this.loop);
  };

  /* ── events ─────────────────────────────────────────────────────── */

  private pickAt(cx: number, cy: number): string | null {
    const r = this.gl.domElement.getBoundingClientRect();
    this.pointer.set(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables, false);
    if (!hits.length) return null;
    // prefer the smallest box under the cursor (e.g. a beacon in front of a zone)
    hits.sort((a, b) => a.distance - b.distance);
    return (hits[0].object.userData.id as string) ?? null;
  }

  private stopOrbit = () => {
    this.flight = null;
    this.setAutoRotate(false);
  };

  private onMove = (ev: PointerEvent) => {
    const id = this.pickAt(ev.clientX, ev.clientY);
    this.gl.domElement.style.cursor = id ? "pointer" : this.downAt ? "grabbing" : "grab";
    if (id !== this.hoverId) {
      this.hoverId = id;
      this.applyState();
    }
    this.cb.onHover(id, ev.clientX, ev.clientY);
  };

  private onDown = (ev: PointerEvent) => {
    this.downAt = { x: ev.clientX, y: ev.clientY };
    this.stopOrbit();
  };

  private onUp = (ev: PointerEvent) => {
    if (!this.downAt) return;
    const moved = Math.hypot(ev.clientX - this.downAt.x, ev.clientY - this.downAt.y);
    this.downAt = null;
    if (moved < 5) this.cb.onSelect(this.pickAt(ev.clientX, ev.clientY));
  };

  private onDbl = (ev: MouseEvent) => {
    const id = this.pickAt(ev.clientX, ev.clientY);
    if (id) this.focus(id);
  };

  private onLeave = () => {
    this.downAt = null;
    if (this.hoverId) {
      this.hoverId = null;
      this.applyState();
    }
    this.cb.onHover(null, 0, 0);
  };

  private onVisibility = () => this.kick();

  /* ── rendering ──────────────────────────────────────────────────── */

  private resize() {
    const w = Math.max(1, this.host.clientWidth);
    const h = Math.max(1, this.host.clientHeight);
    this.gl.setSize(w, h, false);
    this.css.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  requestRender() {
    if (this.frame || this.running || this.disposed) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.render();
    });
  }

  private render() {
    if (this.disposed) return;
    this.camera.updateMatrixWorld();
    // depth cue: labels on the far side of the stage fade; labels behind the camera hide
    const tdist = this.camera.position.distanceTo(this.controls.target);
    const v = new THREE.Vector3();
    for (const l of this.labels) {
      // CSS3DRenderer owns element.style.display (from obj.visible), so hide via visibility
      l.obj.visible = !l.hidden;
      if (l.hidden) {
        l.op = -1;
        continue;
      }
      l.obj.getWorldPosition(v).applyMatrix4(this.camera.matrixWorldInverse);
      const behind = v.z > -0.4;
      const rel = (-v.z - tdist) / 9;
      const op = behind ? 0 : THREE.MathUtils.clamp(1 - Math.max(0, rel) * 1.1, 0.16, 1);
      if (Math.abs(op - l.op) > 0.03) {
        l.op = op;
        l.wrap.style.opacity = op.toFixed(2);
        l.wrap.style.visibility = behind ? "hidden" : "";
      }
    }
    this.gl.render(this.scene, this.camera);
    this.css.render(this.scene, this.camera);
  }

  private clear(group: THREE.Group) {
    const geos = new Set<THREE.BufferGeometry>();
    const mats = new Set<THREE.Material>();
    group.traverse((o) => {
      if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments || o instanceof THREE.Line) {
        geos.add(o.geometry);
        (Array.isArray(o.material) ? o.material : [o.material]).forEach((m: THREE.Material) => mats.add(m));
      }
      if (o instanceof CSS3DSprite) o.element.remove();
    });
    geos.forEach((g) => g.dispose());
    mats.forEach((m) => m.dispose());
    group.clear();
  }

  dispose() {
    this.disposed = true;
    this.running = false;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.resizeObs.disconnect();
    this.io?.disconnect();
    document.removeEventListener("visibilitychange", this.onVisibility);
    const el = this.gl.domElement;
    el.removeEventListener("pointermove", this.onMove);
    el.removeEventListener("pointerdown", this.onDown);
    el.removeEventListener("pointerup", this.onUp);
    el.removeEventListener("pointerleave", this.onLeave);
    el.removeEventListener("dblclick", this.onDbl);
    el.removeEventListener("wheel", this.stopOrbit);
    this.controls.dispose();
    this.clear(this.objs);
    this.clear(this.overlayGroup);
    this.clear(this.labelGroup);
    this.dynTex.forEach((t) => t.dispose());
    this.tex.forEach((t) => t.dispose());
    this.hitMat.dispose();
    this.gl.dispose();
    this.gl.forceContextLoss();
    el.remove();
    this.css.domElement.remove();
  }
}

function nodeShape(g: ArchGraph, id: string): string {
  const n: ArchNode | undefined = g.nodes.find((x) => x.id === id);
  if (!n) return "";
  return n.id === "fusion" ? `${fmtShape(n.in_shapes?.a)} + ${fmtShape(n.in_shapes?.b)} → ${fmtShape(n.out_shape)}` : fmtShape(n.out_shape);
}
