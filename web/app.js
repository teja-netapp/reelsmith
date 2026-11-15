(() => {
  const $ = (id) => document.getElementById(id);
  const state = { clips: [], music: null, length: 30, quality: "final", planner: "auto", jobId: null, poll: null, lastVersion: 0 };

  const EXAMPLES = [
    "Cinematic travel montage, warm film look, slow-mo on the best moments, title \"WANDER\"",
    "High-energy hype reel, fast beat cuts, whip pans and flashes, vibrant colours",
    "Moody black & white, slow pacing, letterbox, dip-to-black between scenes",
    "Talking vlog: keep my best lines, bold captions, punch-in zooms on key words",
    "Dreamy golden-hour recap with light leaks and soft dissolves",
  ];
  const REFINES = ["Faster pace", "More slow motion", "Warmer, filmic look", "Open with a stronger hook",
    "Fewer transitions, cleaner cuts", "Add a title at the start", "Make it 15 seconds"];

  // ---------- helpers ----------
  const fmtSize = (b) => b > 1e9 ? (b / 1e9).toFixed(1) + " GB" : (b / 1e6).toFixed(1) + " MB";
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    return r.json();
  }
  function options() {
    return {
      target_duration: state.length, quality: state.quality, planner: state.planner,
      auto_pick: $("optAuto").checked, beat_sync: $("optBeat").checked,
      captions: $("optCap").checked, smart_reframe: $("optReframe").checked,
    };
  }

  // ---------- config pills & AI editor selector ----------
  const AI_NAMES = { auto: "Auto", gemini: "Gemini", ollama: "Local AI", claude: "Claude", heuristic: "Rules" };
  const AI_HINTS = {
    auto: "Uses the best editor that's available: Claude → Gemini → Local AI → rules.",
    gemini: "Cloud. Strongest vision + brief-following. Needs GEMINI_API_KEY (free tier available).",
    ollama: "Free & private on this Mac. Logs every shot first (~3–5 s each, cached), then writes the edit.",
    claude: "Cloud. Needs ANTHROPIC_API_KEY.",
    heuristic: "No AI: keyword rules + quality scores. Instant, but ignores detailed instructions.",
  };
  let providers = {};
  function updateHint() {
    const k = state.planner, st = providers[k];
    let txt = AI_HINTS[k] || "";
    if (k === "auto") {
      const pick = ["claude", "gemini", "ollama"].find((p) => providers[p] && providers[p].available) || "heuristic";
      txt += ` Right now: ${AI_NAMES[pick]}.`;
    } else if (st && !st.available) txt = `Not ready: ${st.detail}. It will fall back to the next editor.`;
    $("aiHint").textContent = txt;
    $("aiHint").classList.toggle("bad", !!(st && !st.available && k !== "auto"));
  }
  api("/api/config").then((c) => {
    providers = c.providers || {};
    if (c.version) $("ver").textContent = "v" + c.version;
    const pv = (k, label) => {
      const s = providers[k] || {};
      return [s.available ? `${label} · ${s.detail}` : `${label}: ${s.detail || "off"}`, s.available ? "ok" : "warn"];
    };
    const pills = [
      pv("gemini", "Gemini"), pv("ollama", "Local AI"),
      ...(providers.claude && providers.claude.available ? [pv("claude", "Claude")] : []),
      [c.whisper ? "Captions ready" : "Whisper not installed", c.whisper ? "ok" : "warn"],
      [c.face_model ? "Face tracking" : "Saliency tracking", c.face_model ? "ok" : "warn"],
      [c.ffmpeg ? `FFmpeg · ${c.encoder}` : "FFmpeg missing!", c.ffmpeg ? "ok" : "warn"],
    ];
    $("pills").innerHTML = pills.map(([t, k]) => `<span class="pill ${k}">${esc(t)}</span>`).join("");
    [...$("aiSeg").children].forEach((b) => {
      const s = providers[b.dataset.v];
      b.classList.toggle("na", !!(s && !s.available));
      b.title = s ? (s.available ? s.detail : "Not ready: " + s.detail) : AI_HINTS.auto;
    });
    let saved = null;
    try { saved = localStorage.getItem("reelsmith.planner"); } catch (e) {}
    const initial = saved || c.default_planner || "auto";
    const btn = [...$("aiSeg").children].find((b) => b.dataset.v === initial);
    if (btn) btn.click(); else updateHint();
  }).catch(() => {});

  // ---------- clips & music ----------
  function renderClips() {
    $("clipList").innerHTML = state.clips.map((f, i) =>
      `<li><span class="name">${i + 1}. ${esc(f.name)}</span><span class="size">${fmtSize(f.size)}</span>
        <button data-up="${i}" title="Move up">↑</button><button data-rm="${i}" title="Remove">✕</button></li>`).join("");
  }
  $("clipList").addEventListener("click", (e) => {
    const rm = e.target.dataset.rm, up = e.target.dataset.up;
    if (rm !== undefined) state.clips.splice(+rm, 1);
    if (up !== undefined && +up > 0) { const [f] = state.clips.splice(+up, 1); state.clips.splice(+up - 1, 0, f); }
    renderClips();
  });
  function addClips(files) {
    for (const f of files) if (f.type.startsWith("video") || /\.(mov|mp4|m4v|mkv|webm|avi|mts|m2ts)$/i.test(f.name)) state.clips.push(f);
    renderClips();
  }
  function setMusic(f) {
    state.music = f;
    $("musicLabel").innerHTML = f ? `<b>${esc(f.name)}</b> <span class="muted">${fmtSize(f.size)}</span> <a href="#" id="rmMusic">remove</a>`
      : `<b>Add a track</b> <span class="muted">MP3 · M4A · WAV</span>`;
    const rm = $("rmMusic");
    if (rm) rm.onclick = (e) => { e.preventDefault(); e.stopPropagation(); setMusic(null); };
  }
  $("clipInput").onchange = (e) => { addClips(e.target.files); e.target.value = ""; };
  $("musicInput").onchange = (e) => { if (e.target.files[0]) setMusic(e.target.files[0]); e.target.value = ""; };
  for (const [id, fn] of [["clipDrop", (fs) => addClips(fs)], ["musicDrop", (fs) => fs[0] && setMusic(fs[0])]]) {
    const el = $(id);
    el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("over"); });
    el.addEventListener("dragleave", () => el.classList.remove("over"));
    el.addEventListener("drop", (e) => { e.preventDefault(); el.classList.remove("over"); fn(e.dataTransfer.files); });
  }

  // ---------- chips & segmented controls ----------
  $("examples").innerHTML = EXAMPLES.map((x) => `<span class="chip">${esc(x.split(",")[0])}</span>`).join("");
  $("examples").onclick = (e) => {
    const i = [...$("examples").children].indexOf(e.target);
    if (i >= 0) $("desc").value = EXAMPLES[i];
  };
  $("refineChips").innerHTML = REFINES.map((x) => `<span class="chip">${esc(x)}</span>`).join("");
  $("refineChips").onclick = (e) => {
    if (!e.target.classList.contains("chip")) return;
    const t = $("feedback");
    t.value = (t.value ? t.value.trim() + ". " : "") + e.target.textContent;
  };
  function seg(id, key, num) {
    $(id).onclick = (e) => {
      if (!e.target.dataset.v) return;
      [...$(id).children].forEach((b) => b.classList.toggle("on", b === e.target));
      state[key] = num ? +e.target.dataset.v : e.target.dataset.v;
    };
  }
  seg("lengthSeg", "length", true);
  seg("qualSeg", "quality", false);
  seg("aiSeg", "planner", false);
  $("aiSeg").addEventListener("click", () => {
    try { localStorage.setItem("reelsmith.planner", state.planner); } catch (e) {}
    updateHint();
  });

  // ---------- submit ----------
  $("goBtn").onclick = () => {
    $("formErr").textContent = "";
    const paths = $("paths").value.trim();
    if (!state.clips.length && !paths) { $("formErr").textContent = "Add at least one video."; return; }
    if ($("optBeat").checked && !state.music) { /* beat sync silently needs music */ }
    const fd = new FormData();
    state.clips.forEach((f) => fd.append("clips", f, f.name));
    if (state.music) fd.append("music", state.music, state.music.name);
    fd.append("description", $("desc").value);
    fd.append("options", JSON.stringify(options()));
    fd.append("paths", paths);
    $("goBtn").disabled = true;
    showBusy("Uploading footage…", 0);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) showBusy(`Uploading footage… ${fmtSize(e.loaded)} / ${fmtSize(e.total)}`, e.loaded / e.total * 0.02); };
    xhr.onload = () => {
      $("goBtn").disabled = false;
      if (xhr.status >= 300) {
        let m = xhr.statusText; try { m = JSON.parse(xhr.responseText).detail; } catch (e) {}
        $("formErr").textContent = m; hideBusy(); return;
      }
      openJob(JSON.parse(xhr.responseText).id);
      loadRecent();
    };
    xhr.onerror = () => { $("goBtn").disabled = false; hideBusy(); $("formErr").textContent = "Upload failed — is the server running?"; };
    xhr.send(fd);
  };

  // ---------- job view ----------
  function showBusy(text, p) {
    $("emptyState").classList.add("hidden");
    $("busy").classList.remove("hidden");
    $("stageText").textContent = text;
    $("barFill").style.width = Math.round(p * 100) + "%";
    $("pct").textContent = Math.round(p * 100) + "%";
  }
  function hideBusy() { $("busy").classList.add("hidden"); }

  function openJob(id) {
    state.jobId = id; state.lastVersion = -1;
    clearInterval(state.poll);
    tick();
    state.poll = setInterval(tick, 1000);
  }

  async function tick() {
    if (!state.jobId) return;
    let j;
    try { j = await api(`/api/jobs/${state.jobId}`); } catch (e) { return; }
    $("logs").textContent = (j.logs || []).join("\n");
    if (j.status === "running" || j.status === "queued") {
      showBusy(j.stage, j.progress);
      $("refineBtn").disabled = true; $("jsonBtn").disabled = true;
    } else {
      hideBusy();
      $("refineBtn").disabled = false; $("jsonBtn").disabled = false;
      clearInterval(state.poll);
      if (j.status === "error") {
        $("emptyState").classList.toggle("hidden", !!j.output_url);
        $("formErr").textContent = "Render failed: " + (j.error || "unknown error") + " (see Activity log)";
        $("logCard").open = true;
      }
    }
    if (j.plan) renderPlan(j);
    if (j.output_url && j.version !== state.lastVersion && j.status === "done") {
      state.lastVersion = j.version;
      const v = $("player");
      v.src = j.output_url; v.controls = true; v.load(); v.play().catch(() => {});
      $("emptyState").classList.add("hidden");
      $("dlBtn").href = `/api/jobs/${j.id}/output?download=1`;
    }
  }

  function renderPlan(j) {
    const p = j.plan, tl = j.timeline || {};
    $("planCard").classList.remove("hidden");
    $("refineCard").classList.remove("hidden");
    $("plannerName").textContent = `Edited by ${p.planner || "Reelsmith"}${j.version ? " · v" + j.version : ""}`;
    $("planTitle").textContent = p.title || "Untitled";
    $("planReason").textContent = p.reasoning || "";
    const stats = [
      `${(tl.duration || p.estimated_duration || 0).toFixed(1)}s`,
      `${p.segments.length} shots`,
      `look: ${p.look.replace(/_/g, " ")}`,
      p.grain > 0.05 ? `grain ${Math.round(p.grain * 100)}%` : null,
      p.letterbox ? "letterbox" : null,
      tl.beats_snapped ? `${tl.beats_snapped} cuts on beat` : null,
      tl.captions ? `${tl.captions} caption words` : null,
    ].filter(Boolean);
    $("planStats").innerHTML = stats.map((s) => `<span class="stat">${esc(s)}</span>`).join("");
    const segs = tl.segments || [];
    $("timeline").innerHTML = p.segments.map((s, i) => {
      const t = segs[i];
      const d = t ? (t.frames / 30).toFixed(1) : ((s.end - s.start) / s.speed).toFixed(1);
      const thumbId = /^c\d+s\d+$/.test(s.shot_id) ? s.shot_id : null;
      const bg = thumbId ? `style="background-image:url('/api/jobs/${j.id}/thumb/${thumbId}')"` : "";
      const tr = i === 0 ? "start" : (t ? t.transition_in : s.transition_in);
      return `<div class="seg-card" title="${esc(s.shot_id)} ${s.start.toFixed(1)}–${s.end.toFixed(1)}s${s.text ? " · text: " + esc(s.text.content) : ""}">
        <div class="th" ${bg}><span>${d}s${s.speed !== 1 ? " · " + s.speed + "×" : ""}</span></div>
        <div class="tr">${esc(tr.replace(/_/g, " "))}</div><div class="mo">${esc(s.motion.replace(/_/g, " "))}${s.text ? " · T" : ""}</div></div>`;
    }).join("");
    const ta = $("planJson");
    if (document.activeElement !== ta) {
      const clean = { ...p }; delete clean.planner; delete clean.warnings; delete clean.estimated_duration;
      ta.value = JSON.stringify(clean, null, 2);
    }
  }

  $("refineBtn").onclick = async () => {
    const fb = $("feedback").value.trim();
    if (!fb || !state.jobId) return;
    try {
      await api(`/api/jobs/${state.jobId}/refine`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback: fb, options: { quality: state.quality, planner: state.planner, captions: $("optCap").checked, beat_sync: $("optBeat").checked, smart_reframe: $("optReframe").checked } }) });
      $("feedback").value = "";
      openJob(state.jobId);
    } catch (e) { $("formErr").textContent = e.message; }
  };
  $("jsonBtn").onclick = async () => {
    let plan;
    try { plan = JSON.parse($("planJson").value); } catch (e) { $("formErr").textContent = "Plan JSON is invalid: " + e.message; return; }
    try {
      await api(`/api/jobs/${state.jobId}/render`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan, options: { quality: state.quality } }) });
      openJob(state.jobId);
    } catch (e) { $("formErr").textContent = e.message; }
  };

  // ---------- recent ----------
  async function loadRecent() {
    try {
      const jobs = await api("/api/jobs");
      $("recentList").innerHTML = jobs.map((j) =>
        `<li data-id="${esc(j.id)}"><span>${esc(j.title || j.description || j.id)}</span><span class="st">${esc(j.status)}</span></li>`).join("")
        || `<li class="muted">No reels yet</li>`;
    } catch (e) {}
  }
  $("recentList").onclick = (e) => {
    const li = e.target.closest("li[data-id]");
    if (li) openJob(li.dataset.id);
  };
  loadRecent();
})();
