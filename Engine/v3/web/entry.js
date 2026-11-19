// StudyMate v3 browser renderer: Excalidraw -> SVG -> hand-drawn animation, frame by frame.
// Build (only needed if you change this file):
//   npx esbuild v3/web/entry.js --bundle --format=iife --minify --outfile=v3/web/bundle.js \
//       --define:process.env.NODE_ENV='"production"'
import { exportToSvg } from "@excalidraw/excalidraw";

let ITEMS = [];      // one per element: {g, tl, strokes:[{p,len}], fills:[], kind}
let LAST = "";

window.loadScene = async (doc, timeline, W, H) => {
  const svg = await exportToSvg({
    elements: doc.elements,
    appState: { exportBackground: true, viewBackgroundColor: "#ffffff", exportWithDarkMode: false,
                exportPadding: 0 },
    files: doc.files || {},
  });
  const stage = document.getElementById("stage");
  stage.innerHTML = "";
  stage.appendChild(svg);
  const groups = [...svg.children].filter((n) => n.tagName.toLowerCase() === "g");
  // the first element is an invisible 1600x900 frame at (0,0): use it to pin the viewBox
  const m = /translate\(([-\d.e]+)[ ,]+([-\d.e]+)\)/.exec(groups[0]?.getAttribute("transform") || "");
  const ox = m ? parseFloat(m[1]) : 0, oy = m ? parseFloat(m[2]) : 0;
  svg.setAttribute("viewBox", `${ox} ${oy} ${W} ${H}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");
  svg.querySelectorAll(":scope > rect").forEach((r) => {
    r.setAttribute("x", ox - 10); r.setAttribute("y", oy - 10);
    r.setAttribute("width", W + 20); r.setAttribute("height", H + 20);
  });
  const matched = groups.length === timeline.length;
  if (!matched) console.warn(`element/group mismatch: ${groups.length} groups vs ${timeline.length}`);
  ITEMS = groups.map((g, i) => {
    const tl = matched ? timeline[i]
      : timeline[Math.min(timeline.length - 1, Math.round(i * timeline.length / groups.length))];
    const strokes = [], fills = [];
    g.querySelectorAll("path, line, polyline, ellipse, circle, rect").forEach((p) => {
      const stroke = p.getAttribute("stroke");
      if (stroke && stroke !== "none" && p.getTotalLength) {
        let len = 0;
        try { len = p.getTotalLength(); } catch (e) {}
        if (len > 0) { strokes.push({ p, len }); return; }
      }
      fills.push(p);
    });
    const kind = g.querySelector("text") ? "text" : g.querySelector("image") ? "image" : "shape";
    return { g, tl, strokes, fills, kind };
  });
  await document.fonts.ready;
  LAST = "";
  window.seek(0);
  return { groups: groups.length, timeline: timeline.length, matched };
};

const clamp = (x) => Math.max(0, Math.min(1, x));
const ease = (x) => 1 - Math.pow(1 - x, 2);

window.seek = (t) => {
  const sig = [];
  for (const it of ITEMS) {
    const { s, d, e, fade } = it.tl;
    let p = clamp((t - s) / Math.max(d, 0.001));
    let hide = e != null ? clamp((t - e) / 0.25) : 0;
    p = Math.round(ease(p) * 60) / 60;
    hide = Math.round(hide * 20) / 20;
    sig.push(p, hide);
    const g = it.g;
    g.style.opacity = p <= 0 ? 0 : (1 - hide);
    if (p <= 0 || p >= 1 || fade) {
      g.style.clipPath = "";
      for (const st of it.strokes) { st.p.style.strokeDasharray = ""; st.p.style.strokeDashoffset = ""; }
      for (const f of it.fills) f.style.opacity = "";
      if (fade && p > 0) g.style.opacity = p * (1 - hide);
      continue;
    }
    if (it.kind === "text") {                       // handwriting: reveal left to right
      g.style.clipPath = `inset(-20% ${(1 - p) * 100}% -20% -5%)`;
    } else if (it.kind === "image") {
      g.style.opacity = p;
    } else {                                        // draw strokes, then fill
      for (const st of it.strokes) {
        st.p.style.strokeDasharray = `${st.len} ${st.len}`;
        st.p.style.strokeDashoffset = `${st.len * (1 - Math.min(1, p * 1.15))}`;
      }
      for (const f of it.fills) f.style.opacity = clamp((p - 0.45) / 0.55);
    }
  }
  const key = sig.join(",");
  const changed = key !== LAST;
  LAST = key;
  return changed;
};
window.READY = true;
