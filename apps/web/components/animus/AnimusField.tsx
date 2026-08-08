"use client";

import { useEffect, useRef } from "react";

/**
 * AnimusField — the reconstruction lattice behind the console.
 *
 * A fixed, inert, full-viewport canvas: faint low-poly plates drifting at the
 * back, a slow constellation of nodes and hairline edges in front, a handful of
 * "scanned" nodes breathing on top. Every colour comes from the --animus-field-*
 * custom properties, so the field re-skins with the theme instead of fighting it.
 *
 * The whole thing is imperative — no React state, no re-renders, one rAF loop.
 */

type FieldNode = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  /** parallax weight, 0.45..1 — front nodes swing further with the pointer */
  depth: number;
  /** 0 = inert, otherwise pulse rate in rad/s */
  pulse: number;
  phase: number;
};

type Plate = {
  /** triangle in normalized viewport units: x0,y0,x1,y1,x2,y2 */
  pts: number[];
  drift: number;
  phase: number;
  amp: number;
};

const TAU = Math.PI * 2;
const MAX_DPR = 2;
const AREA_PER_NODE = 16000;
const MIN_NODES = 40;
const MAX_NODES = 140;
/** pointer-driven shift, css px — nodes lean in, plates lean out */
const NODE_PARALLAX = 16;
const PLATE_PARALLAX = 7;
/** parallax follow rate, 1/s — high enough to feel live, low enough to never jitter */
const EASE = 2.6;
/** clamp frame delta so a stalled tab can't teleport the field */
const MAX_DT = 0.05;

function rand(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

export function AnimusField() {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const root = document.documentElement;
    const reduceQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

    let w = 0;
    let h = 0;
    let link2 = 0;
    let margin = 0;
    const nodes: FieldNode[] = [];
    let plates: Plate[] = [];

    // Screen-space scratch, allocated once — the edge pass reads it, never the node objects.
    const sx = new Float32Array(MAX_NODES);
    const sy = new Float32Array(MAX_NODES);

    // ── theme ────────────────────────────────────────────────────────────────
    const color = {
      node: "transparent",
      line: "transparent",
      poly: "transparent",
      glow: "transparent",
    };

    const readColors = () => {
      const cs = getComputedStyle(root);
      // "transparent" fallback: an empty string is an invalid canvas style and
      // would silently leave the previous colour in place.
      const read = (name: string) => cs.getPropertyValue(name).trim() || "transparent";
      color.node = read("--animus-field-node");
      color.line = read("--animus-field-line");
      color.poly = read("--animus-field-poly");
      color.glow = read("--animus-field-glow");
    };

    // ── geometry ─────────────────────────────────────────────────────────────
    let cols = 1;
    let rows = 1;

    const spawn = (i: number): FieldNode => {
      const speed = rand(2, 6); // css px/s — ~15s to cross 60px
      const dir = Math.random() * TAU;
      // Jittered grid, not pure random: uniform randomness clumps into blobs and
      // leaves dead voids. Stratifying keeps the lattice evenly woven.
      const col = i % cols;
      const row = Math.floor(i / cols) % rows;
      return {
        x: ((col + rand(0.12, 0.88)) * w) / cols,
        y: ((row + rand(0.12, 0.88)) * h) / rows,
        vx: Math.cos(dir) * speed,
        vy: Math.sin(dir) * speed,
        r: rand(1, 1.6),
        depth: rand(0.45, 1),
        pulse: 0,
        phase: Math.random() * TAU,
      };
    };

    const wrap = (n: FieldNode) => {
      if (n.x < -margin) n.x = w + margin;
      else if (n.x > w + margin) n.x = -margin;
      if (n.y < -margin) n.y = h + margin;
      else if (n.y > h + margin) n.y = -margin;
    };

    const buildPlates = () => {
      const count = 3 + Math.floor(Math.random() * 4); // 3..6
      plates = [];
      for (let i = 0; i < count; i++) {
        const cx = rand(-0.05, 1.05);
        const cy = rand(-0.05, 1.05);
        const base = Math.random() * TAU;
        const rx = rand(0.26, 0.52);
        const ry = rand(0.26, 0.52);
        const pts: number[] = [];
        // Even thirds with a little jitter — keeps the triangles fat, never slivers.
        for (let k = 0; k < 3; k++) {
          const a = base + (k * TAU) / 3 + rand(-0.34, 0.34);
          pts.push(cx + Math.cos(a) * rx, cy + Math.sin(a) * ry);
        }
        plates.push({
          pts,
          drift: rand(0.02, 0.05), // 125..315s period
          phase: Math.random() * TAU,
          amp: rand(30, 70),
        });
      }
    };

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
      w = canvas.clientWidth || window.innerWidth;
      h = canvas.clientHeight || window.innerHeight;
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const target = clamp(Math.round((w * h) / AREA_PER_NODE), MIN_NODES, MAX_NODES);
      // Link range tracks actual spacing, so a clamped 4K field keeps the same
      // visual density instead of falling apart into dust.
      const spacing = Math.sqrt((w * h) / target);
      const link = clamp(spacing * 1.28, 110, 200);
      link2 = link * link;
      margin = link;
      cols = Math.max(1, Math.round(Math.sqrt((target * w) / h)));
      rows = Math.max(1, Math.ceil(target / cols));

      while (nodes.length > target) nodes.pop();
      while (nodes.length < target) nodes.push(spawn(nodes.length));
      for (const n of nodes) wrap(n); // a shrunk viewport can strand nodes outside

      // Spread the "scanned" nodes across the field — index order would stack
      // them all in the first grid row.
      const active = clamp(Math.round(nodes.length * 0.05), 2, 6);
      const stride = nodes.length / active;
      for (const n of nodes) n.pulse = 0;
      for (let k = 0; k < active; k++) {
        const idx = Math.min(nodes.length - 1, Math.floor(k * stride + rand(0, stride)));
        nodes[idx].pulse = rand(0.33, 0.5); // 12..19s breath
      }
    };

    // ── pointer parallax ─────────────────────────────────────────────────────
    let tx = 0;
    let ty = 0;
    let cx = 0;
    let cy = 0;

    const onPointer = (e: PointerEvent) => {
      if (!w || !h) return;
      tx = clamp((e.clientX / w) * 2 - 1, -1, 1);
      ty = clamp((e.clientY / h) * 2 - 1, -1, 1);
    };

    // ── render ───────────────────────────────────────────────────────────────
    const draw = (time: number) => {
      const count = nodes.length;
      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = color.poly;
      const plx = -cx * PLATE_PARALLAX;
      const ply = -cy * PLATE_PARALLAX;
      for (const p of plates) {
        // Sine sway rather than linear drift: no wrap, so nothing ever pops.
        const dx = Math.sin(time * p.drift + p.phase) * p.amp + plx;
        const dy = Math.cos(time * p.drift * 0.8 + p.phase) * p.amp * 0.55 + ply;
        ctx.beginPath();
        ctx.moveTo(p.pts[0] * w + dx, p.pts[1] * h + dy);
        ctx.lineTo(p.pts[2] * w + dx, p.pts[3] * h + dy);
        ctx.lineTo(p.pts[4] * w + dx, p.pts[5] * h + dy);
        ctx.closePath();
        ctx.fill();
      }

      const nox = cx * NODE_PARALLAX;
      const noy = cy * NODE_PARALLAX;
      for (let i = 0; i < count; i++) {
        const n = nodes[i];
        sx[i] = n.x + nox * n.depth;
        sy[i] = n.y + noy * n.depth;
      }

      // Edges. Squared distance only — no sqrt in the hot pair loop.
      ctx.strokeStyle = color.line;
      ctx.lineWidth = 0.7;
      for (let i = 0; i < count; i++) {
        const xi = sx[i];
        const yi = sy[i];
        for (let j = i + 1; j < count; j++) {
          const dx = xi - sx[j];
          const dy = yi - sy[j];
          const d2 = dx * dx + dy * dy;
          if (d2 >= link2) continue;
          const t = 1 - d2 / link2;
          ctx.globalAlpha = t; // linear falloff — reaches 0 at the threshold, so no popping
          ctx.beginPath();
          ctx.moveTo(xi, yi);
          ctx.lineTo(sx[j], sy[j]);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;

      // Inert nodes: one path, one fill.
      ctx.fillStyle = color.node;
      ctx.beginPath();
      for (let i = 0; i < count; i++) {
        const n = nodes[i];
        if (n.pulse) continue;
        ctx.moveTo(sx[i] + n.r, sy[i]);
        ctx.arc(sx[i], sy[i], n.r, 0, TAU);
      }
      ctx.fill();

      // Scanned nodes: slow breath plus a thin ring.
      ctx.fillStyle = color.glow;
      ctx.strokeStyle = color.glow;
      ctx.lineWidth = 0.8;
      for (let i = 0; i < count; i++) {
        const n = nodes[i];
        if (!n.pulse) continue;
        const p = 0.5 + 0.5 * Math.sin(time * n.pulse + n.phase);
        ctx.globalAlpha = 0.42 + 0.34 * p;
        ctx.beginPath();
        ctx.arc(sx[i], sy[i], n.r + 0.35, 0, TAU);
        ctx.fill();
        ctx.globalAlpha = 0.05 + 0.15 * p;
        ctx.beginPath();
        ctx.arc(sx[i], sy[i], 3 + 2.2 * p, 0, TAU);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    };

    // ── loop ─────────────────────────────────────────────────────────────────
    let raf = 0;
    let last = -1;
    let elapsed = 0;

    const frame = (ts: number) => {
      raf = requestAnimationFrame(frame);
      const now = ts / 1000;
      const dt = last < 0 ? 0 : Math.min(now - last, MAX_DT);
      last = now;
      elapsed += dt;

      const k = 1 - Math.exp(-dt * EASE);
      cx += (tx - cx) * k;
      cy += (ty - cy) * k;

      for (const n of nodes) {
        n.x += n.vx * dt;
        n.y += n.vy * dt;
        wrap(n);
      }

      draw(elapsed);
    };

    const start = () => {
      if (raf || reduceQuery.matches || document.hidden) return;
      last = -1;
      raf = requestAnimationFrame(frame);
    };

    const stop = () => {
      if (!raf) return;
      cancelAnimationFrame(raf);
      raf = 0;
    };

    // ── listeners ────────────────────────────────────────────────────────────
    let resizeRaf = 0;
    const onResize = () => {
      if (resizeRaf) return;
      resizeRaf = requestAnimationFrame(() => {
        resizeRaf = 0;
        resize();
        if (!raf) draw(elapsed); // paused or reduced-motion: refresh the held frame
      });
    };

    const onVisibility = () => {
      if (document.hidden) stop();
      else start();
    };

    const onMotion = () => {
      if (reduceQuery.matches) {
        stop();
        draw(elapsed);
      } else {
        start();
      }
    };

    const themeObserver = new MutationObserver(() => {
      readColors();
      if (!raf) draw(elapsed);
    });

    readColors();
    resize();
    buildPlates();
    draw(elapsed);
    start();

    themeObserver.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    window.addEventListener("resize", onResize);
    window.addEventListener("pointermove", onPointer, { passive: true });
    document.addEventListener("visibilitychange", onVisibility);
    reduceQuery.addEventListener("change", onMotion);

    return () => {
      stop();
      if (resizeRaf) cancelAnimationFrame(resizeRaf);
      themeObserver.disconnect();
      window.removeEventListener("resize", onResize);
      window.removeEventListener("pointermove", onPointer);
      document.removeEventListener("visibilitychange", onVisibility);
      reduceQuery.removeEventListener("change", onMotion);
    };
  }, []);

  return (
    <canvas
      ref={ref}
      aria-hidden="true"
      // Inline rather than utility classes: the field must be correctly placed
      // and inert on its own, independent of stylesheet load order.
      style={{
        position: "fixed",
        inset: 0,
        width: "100%",
        height: "100%",
        zIndex: 0,
        pointerEvents: "none",
      }}
    />
  );
}
