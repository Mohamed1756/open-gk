#!/usr/bin/env python3.11
"""
Blind deal for gaze-yaw rating: stratified A/B/overlap split + rating pages.

Reads data/gaze_rating/{manifest,key}.json. The dealer (this script) sees the
sealed key for stratification; raters receive only blind rows. Writes
labels_A.json / labels_B.json skeletons and rate_A.html / rate_B.html
self-contained rating pages (open in a browser, no server needed).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RATING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gaze rating __RATER__</title>
<style>
body{font-family:sans-serif;max-width:900px;margin:1rem auto;padding:0 1rem}
img{max-width:100%;max-height:55vh;border:1px solid #888;cursor:crosshair}
.row{margin:.6rem 0}.bar{height:10px;background:#eee}#fill{height:10px;background:#2a2;width:0%}
button{padding:.4rem .9rem;margin-right:.4rem}input[type=number]{width:5rem}
.hint{color:#555;font-size:.9rem}
</style>
</head>
<body>
<h2>Rater __RATER__ — head yaw labeling</h2>
<p class="hint">Convention: y-down screen coords, 0&deg; = right, 90&deg; = down, &minus;90&deg; = up.
View &isin; front/side/back. Head size: click crown then chin on the crop.</p>
<div class="bar"><div id="fill"></div></div>
<p id="prog"></p>
<img id="crop" alt="keeper crop">
<div class="row">Yaw (deg): <input id="yaw" type="range" min="-180" max="180" step="1" value="90">
<input id="yawnum" type="number" min="-180" max="180" step="1" value="90"></div>
<div class="row">View: <select id="view">
<option value="">?</option><option>front</option><option>side</option><option>back</option>
</select></div>
<div class="row">Head px: <input id="head" type="number" min="1" step="1">
<span class="hint">click crown, then chin</span></div>
<div class="row">
<button id="prev">&larr; Prev</button>
<button id="save">Save &amp; next &rarr;</button>
<button id="export">Export labels_</button><span id="rater"></span><span>.json</span>
</div>
<script>
const ITEMS = __ITEMS__;
const RATER = "__RATER__";
const storeKey = "gaze_" + RATER + "_v1";
let state = JSON.parse(localStorage.getItem(storeKey) || "{}");
let idx = 0, clicks = [];
const $ = id => document.getElementById(id);
function persist(){ localStorage.setItem(storeKey, JSON.stringify(state)); }
function firstOpen(){ for (let i = 0; i < ITEMS.length; i++) if (!state[ITEMS[i].id]) return i; return 0; }
function render(){
  const it = ITEMS[idx], saved = state[it.id] || {};
  $("crop").src = "crops/" + it.crop;
  $("yaw").value = $("yawnum").value = (saved.yaw_deg ?? 90);
  $("view").value = saved.view || "";
  $("head").value = saved.head_px ?? "";
  clicks = [];
  const done = Object.keys(state).length;
  $("fill").style.width = (100 * done / ITEMS.length) + "%";
  $("prog").textContent = (idx + 1) + " / " + ITEMS.length + " — saved " + done;
}
$("yaw").oninput = e => $("yawnum").value = e.target.value;
$("yawnum").oninput = e => $("yaw").value = e.target.value;
$("crop").onclick = e => {
  const r = e.target.getBoundingClientRect();
  const sx = e.target.naturalWidth / r.width, sy = e.target.naturalHeight / r.height;
  clicks.push([(e.clientX - r.left) * sx, (e.clientY - r.top) * sy]);
  if (clicks.length === 2) {
    const dx = clicks[1][0] - clicks[0][0], dy = clicks[1][1] - clicks[0][1];
    $("head").value = Math.round(Math.hypot(dx, dy));
    clicks = [];
  }
};
$("save").onclick = () => {
  const yaw = parseFloat($("yawnum").value), view = $("view").value, head = parseFloat($("head").value);
  if (!isFinite(yaw) || !view || !isFinite(head)) { alert("Fill yaw, view, head px first."); return; }
  state[ITEMS[idx].id] = {yaw_deg: yaw, view: view, head_px: head, saved_at: new Date().toISOString()};
  persist();
  idx = (idx + 1) % ITEMS.length;
  render();
};
$("prev").onclick = () => { idx = (idx - 1 + ITEMS.length) % ITEMS.length; render(); };
$("export").onclick = () => {
  const rows = ITEMS.map(it => Object.assign({id: it.id, crop: it.crop, rater: RATER}, state[it.id] || null));
  const blob = new Blob([JSON.stringify(rows, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_" + RATER + ".json";
  a.click();
};
$("rater").textContent = RATER;
idx = firstOpen();
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/gaze_rating")
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    work = Path(args.dir)
    manifest = json.loads((work / "manifest.json").read_text())
    key = json.loads((work / "key.json").read_text())
    by_ep: dict[str, list[dict]] = {}
    for row in manifest:
        by_ep.setdefault(key[row["id"]]["episode"], []).append(row)

    shared: list[dict] = []
    only_a: list[dict] = []
    only_b: list[dict] = []
    per_ep_overlap = max(1, round(args.overlap / max(1, len(by_ep))))
    for ep_idx, (ep, rows) in enumerate(sorted(by_ep.items())):
        rng.shuffle(rows)
        n_shared = min(per_ep_overlap, len(rows) // 3)
        shared.extend(rows[:n_shared])
        rest = rows[n_shared:]
        half = len(rest) // 2
        if ep_idx % 2 == 0:
            only_a.extend(rest[:half])
            only_b.extend(rest[half:])
        else:
            only_b.extend(rest[:half])
            only_a.extend(rest[half:])

    set_a = only_a + shared
    set_b = only_b + shared
    rng.shuffle(set_a)
    rng.shuffle(set_b)

    for rater, subset in (("A", set_a), ("B", set_b)):
        (work / f"labels_{rater}.json").write_text(json.dumps(subset, indent=2))
        page = RATING_PAGE.replace("__RATER__", rater).replace(
            "__ITEMS__", json.dumps(subset)
        )
        (work / f"rate_{rater}.html").write_text(page)

    print(f"A: {len(set_a)} rows ({len(only_a)} solo + {len(shared)} shared)")
    print(f"B: {len(set_b)} rows ({len(only_b)} solo + {len(shared)} shared)")
    print("Open rate_A.html / rate_B.html in a browser. Export to labels_<rater>.json.")


if __name__ == "__main__":
    main()
