// Video page: upload -> background job -> poll progress -> show processed video, stats, downloads.
const $ = (id) => document.getElementById(id);

async function call(url, opts) {
  const res = await fetch(url, opts);
  let body;
  try { body = await res.json(); } catch { body = {}; }
  if (!res.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `HTTP ${res.status}`);
  return body;
}

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

(() => {
  let jobId = null;
  let timer = null;

  const num = (id) => Number($(id).value);
  const n2 = (x) => (x === null || x === undefined) ? "–" : Number(x).toFixed(2);

  function vstatus(msg, kind = "info") {
    const el = $("vstatus");
    el.className = `status ${kind}`;
    el.textContent = msg;
    if (!msg) el.classList.add("hidden");
  }

  function busy(on) {
    $("vstart").disabled = on;
    $("vcancel").disabled = !on;
    $("vprogress-wrap").classList.toggle("hidden", !on);
  }

  $("vfile").addEventListener("change", () => {
    const f = $("vfile").files[0];
    const v = $("vorig");
    if (v.src) URL.revokeObjectURL(v.src);
    v.removeAttribute("src");
    v.onerror = () => vstatus("The browser cannot preview this file (e.g. AVI); it can still be processed.", "info");
    if (f) v.src = URL.createObjectURL(f);   // local preview; AVI and some codecs can't play in browsers
  });

  $("vform").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = $("vfile").files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    fd.append("threshold", num("vthreshold"));
    fd.append("margin", num("vmargin"));
    fd.append("det_confidence", num("vdet"));
    fd.append("frame_stride", num("vstride"));
    fd.append("reid_interval", num("vinterval"));
    fd.append("min_votes", num("vvotes"));
    fd.append("imgsz", num("vimgsz"));
    fd.append("max_frames", num("vmax"));
    $("vresults").innerHTML = "";
    $("vout").removeAttribute("src");
    $("vdownload").classList.add("hidden");
    $("vprogress").removeAttribute("value");
    $("vprogress-text").textContent = "";
    busy(true);
    vstatus(`Uploading ${(f.size / 1e6).toFixed(1)} MB…`);
    try {
      const job = await call("/api/video/jobs", { method: "POST", body: fd });
      jobId = job.id;
      vstatus("Queued…");
      poll();
    } catch (err) {
      busy(false);
      vstatus(`Error: ${err.message}`, "error");
    }
  });

  $("vcancel").addEventListener("click", async () => {
    if (!jobId) return;
    try { await call(`/api/video/jobs/${jobId}/cancel`, { method: "POST" }); vstatus("Cancelling…"); }
    catch (err) { vstatus(`Error: ${err.message}`, "error"); }
  });

  async function poll() {
    clearTimeout(timer);
    let job;
    try { job = await call(`/api/video/jobs/${jobId}`); }
    catch (err) { busy(false); vstatus(`Error: ${err.message}`, "error"); return; }
    const p = job.progress || {};
    if (job.state === "queued") vstatus("Queued (another video is being processed)…");
    if (job.state === "running") {
      if (p.fraction != null) $("vprogress").value = p.fraction; else $("vprogress").removeAttribute("value");
      const pct = p.fraction != null ? `${(100 * p.fraction).toFixed(1)}% · ` : "";
      const of = p.frames_total_est ? ` / ~${p.frames_total_est}` : "";
      $("vprogress-text").textContent =
        `${pct}${p.frames_processed ?? 0}${of} frames · ${n2(p.processing_fps)} fps · ${p.tracks_so_far ?? 0} tracks so far`;
      vstatus("Processing…");
    }
    if (job.state === "queued" || job.state === "running") { timer = setTimeout(poll, 700); return; }
    busy(false);
    if (job.state === "error") { vstatus(`Error: ${job.error}`, "error"); return; }
    vstatus(job.state === "cancelled" ? "Cancelled — partial results below." : "Done.");
    render(job);
  }

  function stat(label, value, hint = "") {
    return `<div class="stat"><div class="sv">${value}</div><div class="sl">${label}${hint ? `<br><small>${hint}</small>` : ""}</div></div>`;
  }

  function render(job) {
    const s = job.summary;
    if (!s) return;
    const c = s.counts, t = s.timing, d = s.diagnostics;
    const files = job.files || {};
    const vid = s.outputs.video;
    if (vid && files[vid]) {
      $("vout").src = files[vid];
      $("vdownload-btn").href = `${files[vid]}?download=1`;
      $("vdownload").classList.remove("hidden");
    }
    const ids = c.matched_known_identities.length ? c.matched_known_identities.map(esc).join(", ") : "none";
    const dl = Object.entries(files).map(([n, u]) => `<a class="btn" href="${u}?download=1">${esc(n)}</a>`).join(" ");
    const trackRows = (job.tracks || []).map((r) => `<tr class="${r.final_status.toLowerCase()}">
      <td>${r.track_id}</td><td>${r.final_status === "PENDING" ? "UNKNOWN (insufficient evidence)" : esc(r.final_identity)}</td>
      <td>${r.final_score ?? "–"}</td><td>${r.reid_observations}</td><td>${r.frames_visible}</td>
      <td>${r.first_frame}–${r.last_frame}</td><td>${r.label_changes}</td></tr>`).join("");
    $("vresults").innerHTML = `
      ${s.outputs.browser_playable === false ? `<div class="panel warn">The output was written as ${esc(s.outputs.video_codec)}; use the download link if it does not play here.</div>` : ""}
      <div class="stats">
        ${stat("frames processed", s.frames_processed, `of ${s.frames_read} read · stride ${s.frame_stride}`)}
        ${stat("person detections", c.person_detections, "boxes over all processed frames")}
        ${stat("person tracks", c.unique_track_ids, `max ${c.max_simultaneous_tracks} at once`)}
        ${stat("known tracks", c.tracks_known, `identities: ${ids}`)}
        ${stat("unknown / rejected tracks", c.tracks_unknown)}
        ${stat("insufficient evidence", c.tracks_insufficient_evidence, "too few usable crops; shown as UNKNOWN")}
        ${stat("processing fps", n2(t.processing_fps), `${n2(t.realtime_factor)}× real time · ${esc(t.device)}`)}
        ${stat("total time", `${n2(t.total_s)} s`, `detect ${n2(t.detector_s)} · re-id ${n2(t.reid_s)} · track ${n2(t.tracker_s)}`)}
      </div>
      <p class="hint">These are counts, <b>not accuracy</b>. Whether the identities are right can only be measured
        against ground truth (<code>scripts/evaluate_video_tracking.py</code>). Scores are cosine similarities.</p>
      <div class="downloads"><span class="hint">All outputs:</span> ${dl}</div>
      <details><summary>Diagnostics</summary>
        <table class="kv">
          <tr><th>checkpoint</th><td>${esc(d.checkpoint)}</td></tr>
          <tr><th>embedding dim / gallery dim</th><td>${d.embedding_dim} / ${d.gallery_dim} ${d.embedding_dim_matches_gallery ? "✔" : "✘"}</td></tr>
          <tr><th>gallery</th><td>${d.gallery_images} images / ${d.gallery_identities} identities (${esc(d.gallery_source)}, ${esc(d.aggregation)})</td></tr>
          <tr><th>threshold / margin</th><td>${d.threshold} / ${d.margin}</td></tr>
          <tr><th>Re-ID crops embedded</th><td>${d.embedded_crops} in ${d.embed_batches} batches · ${c.reid_observations_accepted} accepted individually</td></tr>
          <tr><th>crop size (h × w px)</th><td>${d.crop_height_px ? `${d.crop_height_px.min}–${d.crop_height_px.max} × ${d.crop_width_px.min}–${d.crop_width_px.max} (median ${d.crop_height_px.median} × ${d.crop_width_px.median})` : "–"}</td></tr>
          <tr><th>crops skipped</th><td>${esc(JSON.stringify(d.crops_skipped))}</td></tr>
          <tr><th>adjacent-frame track continuity</th><td>${d.adjacent_frame_track_continuity ?? "–"}</td></tr>
          <tr><th>track length (processed frames)</th><td>${d.track_length_processed_frames ? JSON.stringify(d.track_length_processed_frames) : "–"}</td></tr>
          <tr><th>label changes (all tracks)</th><td>${d.label_changes_total}</td></tr>
          <tr><th>source</th><td>${s.source.resolution.join("×")} @ ${s.source.fps} fps → output ${s.outputs.output_resolution.join("×")}, ${esc(s.outputs.video_codec || "–")}</td></tr>
        </table>
      </details>
      <details><summary>Tracks (${(job.tracks || []).length}${job.tracks_truncated ? ", first 500" : ""})</summary>
        <table class="tracks"><thead><tr><th>track</th><th>final label</th><th>score</th><th>Re-ID samples</th>
          <th>frames</th><th>frame range</th><th>label changes</th></tr></thead><tbody>${trackRows}</tbody></table>
      </details>`;
  }
})();
