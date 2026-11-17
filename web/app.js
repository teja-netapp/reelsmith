(() => {
  const $ = (id) => document.getElementById(id);
  const state = { clips: [], music: null, length: 30, quality: "final", planner: "auto", jobId: null, poll: null,
    lastVersion: 0, versions: [], viewVersion: 0 };

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
      captions: $("optCap").checked, smart_reframe: $("optReframe").checked, auto_color: $("optColor").checked,
      grading: $("optGrade").checked, grade_strength: +$("optStrength").value / 100,
    };
  }

  // ---------- colour grading switch ----------
  const GRADE_HINT = $("gradeHint").innerHTML;
  function syncGrade() {
    const on = $("optGrade").checked;
    $("gradeSub").classList.toggle("off", !on || !!state.briefRaw);
    $("strengthVal").textContent = $("optStrength").value + "%";
    if (!state.briefRaw) $("gradeHint").innerHTML = on ? GRADE_HINT
      : "<b>Off:</b> your footage keeps its original colours. Only transitions, effects, text &amp; captions are added.";
  }
  $("optGrade").onchange = syncGrade;
  $("optStrength").oninput = syncGrade;

  // ---------- your instructions, detected live (they're enforced as hard rules) ----------
  function ruleChips(pub, lead) {
    const all = (pub && pub.rules) || [];
    const only = all.filter((r) => /^Only what you asked for/.test(r.label));
    const rest = all.filter((r) => !/^Only what you asked for/.test(r.label));
    const chip = (label, title) => {
      const grade = /colou?r|grading|look/i.test(label);
      const off = /^(no |don't|cuts only|keep the original|mute|original audio only|nothing)/i.test(label);
      return `<span class="rule${grade ? " grade" : off ? " off" : ""}" title="${esc(title)}">${esc(label)}</span>`;
    };
    const chips = rest.map((r) => chip(r.label, r.source));
    if (only.length) chips.push(chip("Nothing else added", only.map((r) => r.label.replace("Only what you asked for: ", "")).join(" · ")));
    return chips.length ? `<span class="lead">${esc(lead)}</span>${chips.join("")}` : "";
  }
  function detector(textId, outId, lead, revision) {
    let timer = null, last = "";
    const run = async () => {
      const text = $(textId).value.trim();
      if (text === last) return;
      last = text;
      let pub = { rules: [] };
      if (text) {
        try { pub = await api("/api/directives", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, revision }) }); } catch (e) { return; }
      }
      const html = ruleChips(pub, lead);
      $(outId).innerHTML = html;
      $(outId).classList.toggle("hidden", !html);
      if (!revision) {
        state.briefRaw = !!pub.raw_color;
        if (state.briefRaw && $("optGrade").checked) {
          $("gradeHint").innerHTML = "<b>Your brief says not to touch the colours</b>, so this reel keeps the original colours.";
          $("gradeSub").classList.add("off");
        } else { state.briefRaw = false; syncGrade(); }
      }
    };
    $(textId).addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 350); });
    return run;
  }
  const detectBrief = detector("desc", "descRules", "I'll follow:", false);
  detector("feedback", "fbRules", "I'll follow:", true);

  // ---------- config pills & AI editor selector ----------
  const AI_NAMES = { auto: "Auto", gemini: "Gemini", ollama: "Local AI", claude: "Claude", heuristic: "Rules" };
  const AI_HINTS = {
    auto: "Uses the best editor that's available: Claude → Gemini → Local AI → rules.",
    gemini: "Cloud. Strongest vision + brief-following. Add a free API key in Settings.",
    ollama: "Free & private on this Mac. Logs every shot first (~3–5 s each, cached), then writes the edit.",
    claude: "Cloud. Add an Anthropic API key in Settings.",
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
    const none = !["claude", "gemini", "ollama"].some((p) => providers[p] && providers[p].available);
    $("aiHint").innerHTML = esc(txt) + ((st && !st.available) || none
      ? ` <a href="#" class="open-settings">Open Settings</a>` : "");
    $("aiHint").classList.toggle("bad", !!(st && !st.available && k !== "auto"));
  }
  let configLoaded = false;
  async function loadConfig() {
    let c;
    try { c = await api("/api/config"); } catch (e) { return; }
    providers = c.providers || {};
    if (c.version) $("ver").textContent = "v" + c.version;
    const pv = (k, label) => {
      const s = providers[k] || {};
      return [s.available ? `${label} · ${s.detail}` : `${label}: ${s.detail || "off"}`, s.available ? "ok" : "warn", true];
    };
    const pills = [
      pv("gemini", "Gemini"), pv("ollama", "Local AI"),
      ...(providers.claude && providers.claude.available ? [pv("claude", "Claude")] : []),
      [c.whisper ? "Captions ready" : "Whisper not installed", c.whisper ? "ok" : "warn"],
      [c.face_model ? "Face tracking" : "Saliency tracking", c.face_model ? "ok" : "warn"],
      [c.ffmpeg ? `FFmpeg · ${c.encoder}` : "FFmpeg missing!", c.ffmpeg ? "ok" : "warn"],
    ];
    $("pills").innerHTML = pills.map(([t, k, ai]) =>
      `<span class="pill ${k}${ai ? " open-settings" : ""}"${ai ? ' title="Open Settings"' : ""}>${esc(t)}</span>`).join("")
      + `<span class="pill ok open-settings" id="learnPill" title="Taste model: open Settings">Taste model</span>`
      + `<button class="gear open-settings" title="API keys & AI editors">⚙ Settings</button>`;
    api("/api/learning").then((l) => {
      const p = $("learnPill");
      if (!p) return;
      p.textContent = l.enabled ? `Taste model · ${l.edits} edits · ${Math.round(l.confidence * 100)}% sure` : "Taste model: paused";
      p.className = "pill open-settings " + (l.enabled ? "ok" : "warn");
    }).catch(() => {});
    [...$("aiSeg").children].forEach((b) => {
      const s = providers[b.dataset.v];
      b.classList.toggle("na", !!(s && !s.available));
      b.title = s ? (s.available ? s.detail : "Not ready: " + s.detail) : AI_HINTS.auto;
    });
    if (!configLoaded) {
      configLoaded = true;
      let saved = null;
      try { saved = localStorage.getItem("reelsmith.planner"); } catch (e) {}
      const initial = saved || c.default_planner || "auto";
      const btn = [...$("aiSeg").children].find((b) => b.dataset.v === initial);
      if (btn) btn.click(); else updateHint();
    } else updateHint();
  }
  loadConfig();
  document.addEventListener("click", (e) => {
    if (e.target.closest(".open-settings")) { e.preventDefault(); openSettings(); }
  });

  // ---------- Settings: API keys & AI editors ----------
  const SET = { gemini: "g", claude: "c" };
  function setMsg(id, kind, html) {
    const el = $(id);
    el.className = "msg " + (kind || "");
    el.innerHTML = html || "";
  }
  function statusChip(id, kind, text) {
    $(id).className = "st " + kind;
    $(id).textContent = text;
  }
  function fillModels(sel, models, current, autoLabel) {
    const opts = [`<option value="">${esc(autoLabel)}</option>`].concat(
      [...new Set([...(models || []), ...(current ? [current] : [])])].map((m) => `<option value="${esc(m)}">${esc(m)}</option>`));
    sel.innerHTML = opts.join("");
    sel.value = current || "";
  }
  function showCloud(name, s, verified) {
    const p = SET[name], v = verified || s.verified;
    $(p + "Key").value = "";
    $(p + "Key").placeholder = s.configured ? `Saved: ${s.masked} (paste a new key to replace it)` : "Paste your API key";
    $(p + "Remove").disabled = !s.configured;
    $(p + "Test").disabled = !s.configured;
    if (!s.configured) statusChip("st" + name, "off", "Not connected");
    else if (v && v.ok) statusChip("st" + name, "ok", `Connected · ${s.masked}`);
    else if (v && v.auth === false) statusChip("st" + name, "bad", `Key rejected · ${s.masked}`);
    else statusChip("st" + name, "warn", `Key saved · ${s.masked} · not checked yet`);
    if (name === "gemini") {
      // free-text model name with suggestions from the models this key can see
      const models = (v && v.models) || state.geminiModels || [];
      if (v && v.models) state.geminiModels = v.models;
      $("gModels").innerHTML = models.map((m) => `<option value="${esc(m)}"></option>`).join("");
      $("gModel").value = s.model || "";
      $("gModelHint").textContent = s.model
        ? `Using your choice: ${s.model}`
        : (v && v.will_use ? `Auto currently picks: ${v.will_use}` : "Auto prefers gemini-4-hydrogen, then the newest Flash model");
      if (models.length) $("gModelHint").textContent += ` · ${models.length} models available to this key (start typing for suggestions)`;
    } else fillModels($("cModel"), v && v.models, s.model, "Default (claude-sonnet-5-5)");
    $(p + "Shell").classList.toggle("hidden", !s.from_shell);
  }
  function showOllama(o, v) {
    v = v || o.verified;
    $("oHost").value = o.host || "";
    const installed = (v && v.installed) || [];
    const all = [...new Set([...installed, ...(o.recommended || []), o.model].filter(Boolean))];
    $("oModel").innerHTML = all.map((m) => `<option value="${esc(m)}">${esc(m)}${installed.length && !installed.includes(m) ? " (not downloaded)" : ""}</option>`).join("");
    $("oModel").value = o.model;
    $("oModels").innerHTML = [...new Set([...installed, "qwen3-vl:8b-thinking", "qwen3-vl:8b-instruct", "qwen3-vl:4b-instruct"])]
      .map((m) => `<option value="${esc(m)}"></option>`).join("");
    if (o.plan_depth) $("oDepth").value = o.plan_depth;
    if (o.plan_model_set !== undefined) $("oPlanModel").value = o.plan_model_set ? o.plan_model : "";
    if (!v) statusChip("stollama", "warn", "Not checked yet");
    else if (v.ok) statusChip("stollama", "ok", `Connected · Ollama ${v.version || ""}`);
    else if (v.auth) statusChip("stollama", "warn", "Connected · model not downloaded");
    else statusChip("stollama", "bad", "Not reachable");
    if (v) renderOllamaMsg(v);
  }
  function renderOllamaMsg(v) {
    const m = v.model || $("oModel").value;
    let html = esc(v.detail || "");
    if (v.auth && !v.ok) html += `<div class="cmd"><code>ollama pull ${esc(m)}</code><button class="ghost sm" data-copy="ollama pull ${esc(m)}">Copy</button></div>`;
    setMsg("oMsg", v.ok ? "ok" : (v.auth ? "warn" : "bad"), html);
  }
  async function openSettings() {
    $("settingsModal").classList.remove("hidden");
    ["gMsg", "cMsg", "oMsg", "dMsg"].forEach((id) => setMsg(id, "", ""));
    let s;
    try { s = await api("/api/settings"); } catch (e) { setMsg("gMsg", "bad", esc(e.message)); return; }
    $("setEnv").textContent = s.env_file;
    showCloud("gemini", s.gemini); showCloud("claude", s.claude); showOllama(s.ollama);
    $("dEditor").value = s.default_editor;
    loadLearning();
    // refresh the real status of what's saved (keys are checked with the provider, never shown)
    const checks = [["ollama", null]];
    if (s.gemini.configured && !s.gemini.verified) checks.push(["gemini", s.gemini]);
    if (s.claude.configured && !s.claude.verified) checks.push(["claude", s.claude]);
    for (const [name, sv] of checks) {
      api("/api/settings/verify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: name }) })
        .then((v) => { if (name === "ollama") showOllama(s.ollama, v); else showCloud(name, sv, v); })
        .catch(() => {});
    }
  }
  function closeSettings() { $("settingsModal").classList.add("hidden"); }
  $("setClose").onclick = closeSettings;
  $("settingsModal").addEventListener("click", (e) => { if (e.target.id === "settingsModal") closeSettings(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSettings(); });
  $("settingsModal").addEventListener("click", (e) => {
    const sh = e.target.closest("[data-show]");
    if (sh) { const inp = $(sh.dataset.show); inp.type = inp.type === "password" ? "text" : "password"; sh.textContent = inp.type === "password" ? "Show" : "Hide"; }
    const cp = e.target.closest("[data-copy]");
    if (cp) { navigator.clipboard && navigator.clipboard.writeText(cp.dataset.copy); cp.textContent = "Copied"; }
  });
  async function saveSettings(body, msgId, busyBtn) {
    const label = busyBtn.textContent;
    busyBtn.disabled = true; busyBtn.textContent = "Checking…";
    try {
      const r = await api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const details = Object.values(r.results || {}).map((x) => esc(x.detail || "Saved"));
      const warn = (r.warnings || []).map(esc).join("<br>");
      setMsg(msgId, warn ? "warn" : "ok", (details.length ? details.join("<br>") : "Saved") + (warn ? "<br>" + warn : ""));
      return r;
    } catch (e) {
      setMsg(msgId, "bad", esc(e.message));
      return null;
    } finally {
      busyBtn.disabled = false; busyBtn.textContent = label;
      loadConfig();
    }
  }
  for (const name of ["gemini", "claude"]) {
    const p = SET[name];
    $(p + "Save").onclick = async () => {
      const key = $(p + "Key").value.trim();
      const body = { [name]: { model: $(p + "Model").value } };
      if (key) body[name].key = key;
      const r = await saveSettings(body, p + "Msg", $(p + "Save"));
      if (r) showCloud(name, r.settings[name], r.results[name]);
    };
    $(p + "Test").onclick = async () => {
      $(p + "Test").disabled = true;
      try {
        const v = await api("/api/settings/verify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: name }) });
        setMsg(p + "Msg", v.ok ? "ok" : "bad", esc(v.detail));
        const s = await api("/api/settings");
        showCloud(name, s[name], v);
      } catch (e) { setMsg(p + "Msg", "bad", esc(e.message)); }
      finally { $(p + "Test").disabled = false; loadConfig(); }
    };
    $(p + "Remove").onclick = async () => {
      if (!confirm(`Remove the saved ${name === "gemini" ? "Gemini" : "Claude"} API key from this Mac?`)) return;
      const r = await saveSettings({ [name]: { remove: true } }, p + "Msg", $(p + "Remove"));
      if (r) showCloud(name, r.settings[name]);
    };
    $(p + "Key").addEventListener("keydown", (e) => { if (e.key === "Enter") $(p + "Save").click(); });
  }
  $("gModelTest").onclick = async () => {
    const btn = $("gModelTest"), model = $("gModel").value.trim(), key = $("gKey").value.trim();
    btn.disabled = true; btn.textContent = "Testing…";
    try {
      // tests the typed model with the typed (unsaved) key, or with the saved key
      const v = await api("/api/settings/verify", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: "gemini", model, ...(key ? { key } : {}) }) });
      setMsg("gMsg", v.ok ? "ok" : (v.ok === false ? "bad" : "warn"),
        esc(v.detail) + (v.ok && model ? " · click <b>Save &amp; verify</b> to use it" : ""));
    } catch (e) { setMsg("gMsg", "bad", esc(e.message)); }
    finally { btn.disabled = false; btn.textContent = "Test model"; }
  };
  $("gModel").addEventListener("keydown", (e) => { if (e.key === "Enter") $("gSave").click(); });
  async function loadLearning(stats) {
    let l = stats;
    try { l = l || await api("/api/learning"); } catch (e) { return; }
    $("learnOn").checked = l.enabled;
    statusChip("stlearn", l.enabled ? (l.edits ? "ok" : "warn") : "off",
      l.enabled ? `${Math.round(l.confidence * 100)}% sure` : "paused");
    const trend = l.avg_last != null ? ` · average rating went from ${l.avg_first}★ (first 10) to ${l.avg_last}★ (last 10)`
      : (l.avg_first != null ? ` · average rating so far ${l.avg_first}★` : "");
    $("learnStats").innerHTML = `<div class="bar2"><div style="width:${Math.round(l.confidence * 100)}%"></div></div>`
      + `${l.edits} edits · ${l.rated} rated · ${l.downloaded} downloaded · ${l.feedback} revision requests${esc(trend)}`;
    $("learnBullets").innerHTML = (l.house_style || []).length
      ? `<ul class="chg">${l.house_style.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>`
      : `<div class="note">Nothing learned yet. Rate a few reels (★), use Look Lab, or ask for revisions.</div>`;
    $("learnWeights").innerHTML = (l.learned || []).map((w) =>
      `<div class="wrow"><span>${esc(w.label)}</span><span class="${w.weight > 0 ? "pos" : "neg"}">${w.weight > 0 ? "likes" : "dislikes"} ${Math.abs(w.weight).toFixed(2)}</span></div>`).join("")
      || `<div class="note">No weights yet.</div>`;
  }
  $("learnOn").onchange = async () => {
    try { loadLearning(await api("/api/learning", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: $("learnOn").checked }) })); loadConfig(); } catch (e) { setMsg("learnMsg", "bad", esc(e.message)); }
  };
  $("learnReset").onclick = async () => {
    if (!confirm("Forget everything the taste model learned (ratings, preferences, examples)? Your reels are kept.")) return;
    try { loadLearning(await api("/api/learning/reset", { method: "POST" })); setMsg("learnMsg", "ok", "Learning reset."); loadConfig(); }
    catch (e) { setMsg("learnMsg", "bad", esc(e.message)); }
  };
  $("oSave").onclick = async () => {
    const r = await saveSettings({ ollama: { host: $("oHost").value.trim(), model: $("oModel").value,
      plan_model: $("oPlanModel").value.trim(), plan_depth: $("oDepth").value } }, "oMsg", $("oSave"));
    if (r) { showOllama(r.settings.ollama, r.results.ollama); }
  };
  $("oTest").onclick = async () => {
    $("oTest").disabled = true;
    try {
      const v = await api("/api/settings/verify", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: "ollama", host: $("oHost").value.trim(), model: $("oModel").value }) });
      const s = await api("/api/settings");
      showOllama({ ...s.ollama, host: $("oHost").value.trim(), model: $("oModel").value }, v);
    } catch (e) { setMsg("oMsg", "bad", esc(e.message)); }
    finally { $("oTest").disabled = false; }
  };
  $("dSave").onclick = async () => {
    const d = $("dEditor").value;
    const r = await saveSettings({ default_editor: d }, "dMsg", $("dSave"));
    if (r) {
      setMsg("dMsg", "ok", `New reels will use: ${esc(AI_NAMES[d] || d)}`);
      try { localStorage.setItem("reelsmith.planner", d); } catch (e) {}
      const btn = [...$("aiSeg").children].find((b) => b.dataset.v === d);
      if (btn) btn.click();
    }
  };

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
    if (i >= 0) { $("desc").value = EXAMPLES[i]; detectBrief(); }
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
    state.job = j;
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
    if (j.output_url && j.version !== state.lastVersion && j.status !== "running" && j.status !== "queued") {
      state.lastVersion = j.version;
      state.versions = j.versions || [];
      showVersion(j.version, j.report);
    }
  }

  // ---------- versions & "How this reel was made" ----------
  async function showVersion(v, report) {
    state.viewVersion = v;
    const id = state.jobId;
    const p = $("player");
    p.src = `/api/jobs/${id}/output?v=${v}`; p.controls = true; p.load();
    p.play().catch(() => {});
    $("emptyState").classList.add("hidden");
    $("dlBtn").href = `/api/jobs/${id}/output?download=1&v=${v}`;
    renderVersions();
    renderStars();
    renderLookLab();
    try { renderReport(report || await api(`/api/jobs/${id}/report?v=${v}`)); }
    catch (e) { $("reportCard").classList.add("hidden"); }
  }

  // ---------- ratings: every star teaches the taste model ----------
  function renderStars() {
    const cur = (state.versions || []).find((x) => x.version === state.viewVersion) || {};
    const r = cur.rating || 0;
    $("stars").innerHTML = [1, 2, 3, 4, 5].map((n) =>
      `<button data-n="${n}" class="${n <= r ? "on" : ""}" title="${n} star${n > 1 ? "s" : ""}">★</button>`).join("");
    $("rateMsg").textContent = r ? `You rated v${state.viewVersion}: ${r}★` : "";
  }
  $("stars").onclick = async (e) => {
    const b = e.target.closest("button[data-n]");
    if (!b || !state.jobId) return;
    const stars = +b.dataset.n;
    try {
      const r = await api(`/api/jobs/${state.jobId}/rate`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version: state.viewVersion, stars }) });
      const cur = (state.versions || []).find((x) => x.version === state.viewVersion);
      if (cur) cur.rating = stars;
      renderStars();
      $("rateMsg").textContent = `Thanks! Learned from it (taste model ${Math.round(r.stats.confidence * 100)}% sure, `
        + `${r.stats.rated} rated edits)`;
      loadConfig();
    } catch (err) { $("rateMsg").textContent = err.message; }
  };

  // ---------- Look Lab ----------
  let lookLabFor = null;
  async function renderLookLab() {
    const key = `${state.jobId}:${state.lastVersion}`;
    if (lookLabFor === key) return;
    lookLabFor = key;
    let L;
    try { L = await api(`/api/jobs/${state.jobId}/looks`); } catch (e) { $("lookCard").classList.add("hidden"); return; }
    $("lookCard").classList.remove("hidden");
    const j = state.job || {}, o = j.options || {};
    const graded = !(j.plan && j.plan.grading === false);
    $("llGrade").checked = graded;
    $("llStrength").value = Math.round((o.grade_strength ?? 1) * 100);
    $("llStrengthVal").textContent = $("llStrength").value + "%";
    $("llApply").classList.add("hidden");
    $("lookGrid").classList.toggle("off", !graded);
    $("llHint").textContent = graded ? "off = original colours" : "off: original colours. Turn on, then pick a look";
    lookLabData = L;
    drawLooks();
  }
  let lookLabData = null;
  function drawLooks() {
    const L = lookLabData;
    if (!L) return;
    const st = +$("llStrength").value / 100;
    const tile = (x) =>
      `<button class="lk${x.name === L.current ? " on" : ""}${x.liked ? " liked" : ""}" data-look="${esc(x.name)}"
        title="${esc(x.label)}: ${esc(x.description)}${x.liked ? " · one of your favourites" : ""}">
        <img loading="lazy" alt="" src="/api/jobs/${state.jobId}/look_preview?look=${encodeURIComponent(x.name)}&strength=${st}">
        <span>${esc(x.label)}</span></button>`;
    $("lookGrid").innerHTML = (L.categories || []).map((c) => {
      const items = L.looks.filter((x) => x.category === c);
      return items.length ? `<div class="lk-cat">${esc(c)}</div>${items.map(tile).join("")}` : "";
    }).join("");
  }
  $("lookGrid").onclick = async (e) => {
    const b = e.target.closest(".lk");
    if (!b || b.classList.contains("on")) return;
    try {
      await relook(b.dataset.look, { grading: true });
      [...$("lookGrid").children].forEach((x) => x.classList.toggle("on", x === b));
    } catch (err) { $("formErr").textContent = err.message; }
  };
  async function relook(look, extra) {
    await api(`/api/jobs/${state.jobId}/look`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ look, options: { quality: state.quality, grade_strength: +$("llStrength").value / 100, ...extra } }) });
    lookLabFor = null;
    openJob(state.jobId);
  }
  $("llGrade").onchange = async () => {
    const on = $("llGrade").checked;
    $("lookGrid").classList.toggle("off", !on);
    if (!on) {  // back to the original colours: re-render with no grading at all
      try { await relook("none", { grading: false }); } catch (err) { $("formErr").textContent = err.message; }
    } else {
      $("llHint").textContent = "pick a look below";
    }
  };
  let llTimer = null;
  $("llStrength").oninput = () => {
    $("llStrengthVal").textContent = $("llStrength").value + "%";
    $("llApply").classList.toggle("hidden", !$("llGrade").checked);
    clearTimeout(llTimer);
    llTimer = setTimeout(drawLooks, 300);  // previews follow the strength
  };
  $("llApply").onclick = async () => {
    const cur = state.job && state.job.plan && state.job.plan.look;
    if (!cur || cur === "none") return;
    try { await relook(cur, { grading: true }); } catch (err) { $("formErr").textContent = err.message; }
  };

  const TRIG = { new: "New", refine: "Revision", plan: "Manual edit", look: "Look Lab" };
  function renderVersions() {
    const vs = state.versions || [];
    $("versions").innerHTML = vs.length > 1 ? vs.map((x) =>
      `<button class="vchip${x.version === state.viewVersion ? " on" : ""}" data-v="${x.version}"
        title="${esc((x.feedback ? "“" + x.feedback + "” · " : "") + (x.editor || "") + (x.model ? " · " + x.model : ""))}">
        v${x.version} · ${esc(TRIG[x.trigger] || x.trigger)}</button>`).join("") : "";
  }
  $("versions").onclick = (e) => {
    const b = e.target.closest(".vchip");
    if (b && +b.dataset.v !== state.viewVersion) showVersion(+b.dataset.v);
  };

  const n2 = (x) => (x === null || x === undefined ? "–" : (+x).toLocaleString());
  const kv = (rows) => `<dl class="kv">${rows.filter((r) => r && r[1] !== null && r[1] !== undefined && r[1] !== "")
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;

  function renderReport(r) {
    if (!r) return;
    $("reportCard").classList.remove("hidden");
    const ed = r.editor || {}, s = r.settings || {}, f = r.finishing || {}, a = r.audio || {}, ch = r.changes || {};
    // your instructions (hard rules) and what had to be fixed to obey them
    const ins = r.instructions || {};
    $("repRulesSec").classList.toggle("hidden", !(ins.rules || []).length);
    if ((ins.rules || []).length) {
      const enf = ins.enforced || [];
      $("repRules").innerHTML = `<div class="rules">${ruleChips(ins, "").replace('<span class="lead"></span>', "")}</div>`
        + `<ul>${ins.rules.map((x) => `<li>${esc(x.label)} <span class="src">· ${esc(x.source)}</span></li>`).join("")}</ul>`
        + (enf.length ? `<div class="note"><b>The editor didn't fully follow them, so I fixed:</b></div><ul>${enf.map((c) => `<li>${esc(c.change)}</li>`).join("")}</ul>`
          : `<div class="note">The editor's plan already obeyed them: nothing had to be changed.</div>`);
    }
    // editor
    const u = ed.usage || {};
    let edHtml = `<div class="big">${esc(ed.label || "?")}</div>` + kv([
      ["Model", ed.model ? `<code>${esc(ed.model)}</code>${ed.model_version ? ` <span class="sub">(${esc(ed.model_version)})</span>` : ""}${ed.model_source ? ` <span class="sub">· ${esc(ed.model_source)}</span>` : ""}` : (ed.based_on ? "same edit as before, re-rendered" : "none (rules only)")],
      ed.output_format ? ["Output", esc(ed.output_format)] : null,
      ed.shot_log_model ? ["Shot log", `<code>${esc(ed.shot_log_model)}</code> · ${ed.shots_shown} shots (${ed.shots_logged} new, ${ed.shots_cached} cached${ed.shots_failed ? ", " + ed.shots_failed + " failed" : ""})`] : null,
      ed.runs_on ? ["Runs on", esc(ed.runs_on)] : null,
      (u.input_tokens || u.output_tokens) ? ["Tokens", `${n2(u.input_tokens)} in · ${n2(u.output_tokens)} out${u.thinking_tokens ? " · " + n2(u.thinking_tokens) + " thinking" : ""}`] : null,
      ed.images_sent ? ["Images sent", `${ed.images_sent} contact sheet(s)`] : null,
      ["Planning", ed.seconds !== undefined ? `${ed.seconds} s` : null],
      ["Requested", esc(ed.requested || s.editor_requested || "auto")],
      ed.based_on ? ["Based on", esc(ed.based_on)] : null,
    ]);
    if (ed.feedback || r.trigger?.feedback) edHtml += `<div class="quote">“${esc(ed.feedback || r.trigger.feedback)}”</div>`;
    (ed.attempts || []).forEach((x) => { edHtml += `<div class="note warn">Tried ${esc(x.label)} first: ${esc(x.error)}</div>`; });
    (ed.skipped || []).forEach((x) => { edHtml += `<div class="note">Skipped ${esc(x.provider)}: ${esc(x.reason)}</div>`; });
    $("repEditor").innerHTML = edHtml;
    // brief & settings
    $("repBrief").innerHTML = `<div class="quote">${esc(r.brief || "(no brief)")}</div>` + kv([
      ["Made", `${esc(r.trigger?.label || "")} · ${esc(r.created || "")}`],
      ["Target", `${s.target_duration_s} s → ${r.render?.duration_s} s`],
      ["Options", [s.auto_pick ? "auto-pick" : "clips in order", s.beat_sync ? "beat sync" : null, s.captions ? "captions" : null,
        s.smart_reframe ? "smart reframe" : "centre crop", s.quality].filter(Boolean).map(esc).join(" · ")],
      ["Footage", (r.footage || []).map((c) => `${esc(c.name)} <span class="sub">${c.duration_s.toFixed(1)} s · ${esc(c.resolution)}</span>`).join("<br>")],
      r.concept ? ["Concept", esc(r.concept)] : null,
    ]);
    // look & sound
    const m = a.music;
    $("repLook").innerHTML = kv([
      f.grading === false ? ["Colour", `<b>Original colours</b> <span class="sub">grading off: no look, no shot matching, untouched pixels</span>`]
        : ["Look", `<b>${esc((f.look || "").replace(/_/g, " "))}</b> <span class="sub">${esc(f.look_description || "")}${f.grade_strength !== undefined && f.grade_strength !== 1 ? " · strength " + Math.round(f.grade_strength * 100) + "%" : ""}</span>`],
      ["Finish", [`intensity ${(+f.look_intensity).toFixed(2)}`, `grain ${(+f.grain).toFixed(2)}`, `vignette ${(+f.vignette).toFixed(2)}`,
        f.letterbox ? "letterbox" : null, f.fade_in ? "fade in" : null, f.fade_out ? "fade out" : null].filter(Boolean).join(" · ")],
      ["Text", (f.texts || []).map((t) => `“${esc(t.content)}” <span class="sub">${esc(t.style)} ${t.from_s.toFixed(1)}–${t.to_s.toFixed(1)} s</span>`).join("<br>") || "none"],
      ["Captions", f.captions ? `${esc(f.captions.style)} · ${f.captions.words} words` : "off"],
      ["Music", m ? `${esc(m.name)} <span class="sub">${m.tempo_bpm} BPM · from ${(+m.starts_at_s).toFixed(1)} s · vol ${(+m.volume).toFixed(2)}${a.ducking ? " · ducked under speech" : ""}</span>` : "none"],
      ["Original sound", `${a.original_sound_shots} shot(s)`],
    ]);
    // EDL
    const rows = (r.segments || []).map((g) => {
      const extras = [];
      if (g.text) extras.push(`<span class="tag acc">text “${esc(g.text.content)}”</span>`);
      if (g.captions_words) extras.push(`<span class="tag">${g.captions_words} caption words</span>`);
      if (g.beat_adjust_s) extras.push(`<span class="tag">beat ${g.beat_adjust_s > 0 ? "+" : ""}${g.beat_adjust_s.toFixed(2)} s</span>`);
      const tr = g.transition_in === "start" ? "start" : g.transition_in.replace(/_/g, " ") + (g.transition_s ? ` <span class="sub">${g.transition_s.toFixed(2)} s</span>` : "");
      return `<tr class="edl-row" data-t="${g.starts_at}" title="${esc(g.reframe)}">
        <td>${g.n}</td>
        <td>${esc(g.clip)}<div class="sub">${esc(g.source)}${g.what ? " · " + esc(g.what) : ""}</div></td>
        <td>${g.starts_at.toFixed(1)} s<div class="sub">+${g.on_screen_s.toFixed(1)} s</div></td>
        <td>${g.speed}×</td><td>${esc(g.camera.replace(/_/g, " "))}</td><td>${tr}</td>
        <td>${esc(g.sound)}</td><td>${extras.join("")}</td></tr>`;
    }).join("");
    $("repEdl").innerHTML = `<thead><tr><th>#</th><th>Source (clip · in–out)</th><th>At</th><th>Speed</th><th>Camera</th>
      <th>Transition in</th><th>Sound</th><th>Extras</th></tr></thead><tbody>${rows}</tbody>`;
    // changes
    const items = [];
    (ch.validation || []).forEach((w) => items.push(`<li><span class="tag">validation</span>${esc(w)}</li>`));
    (ch.assistant_editor || []).forEach((c) => items.push(`<li><span class="tag acc">${esc(c.rule)}</span>${esc(c.change)} <span class="why">· ${esc(c.why)}</span></li>`));
    (ch.timeline || []).forEach((t) => items.push(`<li><span class="tag">timeline</span>${esc(t)}</li>`));
    if (ch.beat_sync) items.push(`<li><span class="tag">beat sync</span>${esc(ch.beat_sync)}</li>`);
    const lr = r.learning;
    $("repLearnSec").classList.toggle("hidden", !lr || !!lr.error);
    if (lr && !lr.error) {
      const ap = lr.applied || [];
      $("repLearn").innerHTML = `<div class="bar2" title="taste model confidence"><div style="width:${Math.round((lr.confidence || 0) * 100)}%"></div></div>`
        + `<div class="note">Taste model ${Math.round((lr.confidence || 0) * 100)}% sure · ${lr.candidates || 0} variants scored</div>`
        + (ap.length ? `<ul class="chg">${ap.map((a) => `<li><span class="tag acc">applied</span>${esc(a)}</li>`).join("")}</ul>`
          : `<div class="note">The editor's plan already matched your taste, so no changes were needed.</div>`)
        + (lr.explored ? `<div class="note">Exploring: tried ${esc(lr.explored)}. Rate it so I learn whether you like it.</div>` : "")
        + ((lr.house_style || []).length ? `<div class="note">House style used: ${lr.house_style.map(esc).join(" · ")}</div>` : "");
    }
    const nFix = (ch.validation || []).length + (ch.assistant_editor || []).length + (ch.timeline || []).length;
    $("repCount").textContent = nFix ? `${nFix} fix${nFix > 1 ? "es" : ""}` : "none";
    $("repChanges").innerHTML = items.length ? `<ul class="chg">${items.join("")}</ul>`
      : `<div class="note">None: the editor's plan was used exactly as written.</div>`;
    // footer
    const rd = r.render || {}, tm = r.timings_s || {};
    $("repRender").textContent = `${rd.resolution} · ${rd.fps} fps · ${rd.codec} · ${rd.frames} frames · ${rd.file_mb || "?"} MB · `
      + Object.entries(tm).map(([k, v]) => `${k} ${v}s`).join(", ");
    const base = `/api/jobs/${state.jobId}`;
    $("dlMd").href = `${base}/report.md?v=${r.version}`;
    $("dlJson").href = `${base}/report?v=${r.version}`;
    $("dlAi").href = `${base}/ai_plan?v=${r.version}`;
    $("dlAi").style.display = r.files && r.files.ai_plan ? "" : "none";
  }
  $("repEdl").onclick = (e) => {
    const tr = e.target.closest("tr.edl-row");
    if (!tr) return;
    const p = $("player");
    p.currentTime = +tr.dataset.t + 0.05;
    p.play().catch(() => {});
  };

  function renderPlan(j) {
    const p = j.plan, tl = j.timeline || {};
    $("planCard").classList.remove("hidden");
    $("refineCard").classList.remove("hidden");
    const model = p.editor && p.editor.model && !(p.planner || "").includes(p.editor.model) ? ` · ${p.editor.model}` : "";
    $("plannerName").textContent = `Edited by ${p.planner || "Reelsmith"}${model}${j.version ? " · latest v" + j.version : ""}`;
    $("planTitle").textContent = p.title || "Untitled";
    $("planReason").textContent = p.reasoning || "";
    const stats = [
      `${(tl.duration || p.estimated_duration || 0).toFixed(1)}s`,
      `${p.segments.length} shots`,
      p.grading === false ? "original colours" : `look: ${p.look.replace(/_/g, " ")}`,
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
      const fx = s.effect && s.effect !== "none" ? s.effect : "";
      const dropped = i > 0 && t && t.transition_in !== s.transition_in;
      return `<div class="seg-card" title="${esc(s.shot_id)} ${s.start.toFixed(1)}–${s.end.toFixed(1)}s${s.text ? " · text: " + esc(s.text.content) : ""}">
        <div class="th" ${bg}><span>${d}s${s.speed !== 1 ? " · " + s.speed + "×" : ""}</span>
          <button class="fxb${fx ? " on" : ""}" data-i="${i}" title="clip effect on this shot">${fx ? esc(fx.replace(/_/g, " ")) : "+ fx"}</button></div>
        ${i === 0 ? `<div class="tr">start</div>`
          : `<button class="tr pick" data-i="${i}" title="${dropped ? "planned " + esc(s.transition_in) + " - shots too short, became a cut. " : ""}change the transition into this shot">${esc(tr.replace(/_/g, " "))} ▾</button>`}
        <div class="mo">${esc(s.motion.replace(/_/g, " "))}${s.text ? " · T" : ""}</div></div>`;
    }).join("");
    const ta = $("planJson");
    if (document.activeElement !== ta) {
      const clean = { ...p };
      ["planner", "warnings", "estimated_duration", "editor", "craft_changes", "craft_notes"].forEach((k) => delete clean[k]);
      ta.value = JSON.stringify(clean, null, 2);
    }
  }

  // ---------- per-cut transition / per-shot effect picker (previews on your own shots) ----------
  let catalog = null, pick = null;
  async function loadCatalog() { if (!catalog) catalog = await api("/api/transitions"); return catalog; }
  function planForEdit() {
    const p = JSON.parse(JSON.stringify(state.job.plan));
    ["planner", "warnings", "estimated_duration", "editor", "craft_changes", "craft_notes"].forEach((k) => delete p[k]);
    return p;
  }
  const shotOf = (k) => { const s = state.job.plan.segments[k]; return s && /^c\d+s\d+$/.test(s.shot_id) ? s.shot_id : ""; };
  const tileHtml = (x, cur, src, poster) =>
    `<button class="pv${x.name === cur ? " on" : ""}" data-name="${esc(x.name)}" title="${esc(x.description)}${x.beat ? " (beat-synced)" : ""}">
      <video muted loop playsinline preload="none" data-src="${esc(src)}"${poster ? ` poster="${esc(poster)}"` : ""}></video>
      <span>${esc(x.name.replace(/_/g, " "))}</span></button>`;
  function drawPicker() {
    const j = state.jobId, i = pick.i, cur = pick.cur;
    if (pick.kind === "tr") {
      $("popTabs").innerHTML = catalog.categories.map((c) =>
        `<button data-c="${c.key}" class="${c.key === pick.cat ? "on" : ""}">${esc(c.label)}</button>`).join("");
      const cat = catalog.categories.find((c) => c.key === pick.cat) || catalog.categories[0];
      const a = shotOf(i - 1), b = shotOf(i);
      $("popGrid").innerHTML = cat.items.map((x) => tileHtml(x, cur,
        `/api/transition_preview?name=${x.name}&job=${j}&a=${a}&b=${b}`, a ? `/api/jobs/${j}/thumb/${a}` : "")).join("");
    } else {
      const s = shotOf(i);
      $("popGrid").innerHTML = catalog.effects.map((x) => tileHtml(x, cur,
        `/api/effect_preview?name=${x.name}&job=${j}&shot=${s}`, s ? `/api/jobs/${j}/thumb/${s}` : "")).join("");
    }
    [...$("popGrid").querySelectorAll(".pv")].forEach((b) => {
      const v = b.querySelector("video");
      b.addEventListener("mouseenter", () => { if (!v.src) v.src = v.dataset.src; v.play().catch(() => {}); });
      b.addEventListener("mouseleave", () => v.pause());
      b.addEventListener("focus", () => { if (!v.src) v.src = v.dataset.src; v.play().catch(() => {}); });
    });
  }
  async function openPicker(kind, i, anchor) {
    if (!state.job || !state.job.plan || state.job.status === "running") return;
    try { await loadCatalog(); } catch (e) { $("formErr").textContent = e.message; return; }
    const seg = state.job.plan.segments[i];
    const cur = kind === "tr" ? seg.transition_in : (seg.effect || "none");
    const c = catalog.categories.find((c) => c.items.some((x) => x.name === cur));
    pick = { kind, i, cur, cat: kind === "tr" ? (c && c.key !== "cut" ? c.key : catalog.categories[1].key) : null };
    $("popTitle").textContent = kind === "tr" ? `Transition into shot ${i + 1}` : `Effect on shot ${i + 1}`;
    $("popSub").textContent = `now: ${cur.replace(/_/g, " ")}`;
    $("popAllLabel").textContent = kind === "tr" ? "use it for every cut" : "use it on every shot";
    $("popAll").checked = false;
    $("popTabs").classList.toggle("hidden", kind !== "tr");
    drawPicker();
    const pop = $("pop");
    pop.classList.remove("hidden");
    const r = anchor.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight;
    const left = Math.min(window.innerWidth - pw - 12, Math.max(12, r.left + r.width / 2 - pw / 2));
    let top = r.bottom + 8;
    if (top + ph > window.innerHeight - 12) top = Math.max(12, r.top - ph - 8);
    pop.style.left = left + "px"; pop.style.top = top + "px";
  }
  function closePicker() {
    $("pop").classList.add("hidden");
    [...$("popGrid").querySelectorAll("video")].forEach((v) => { v.pause(); v.removeAttribute("src"); v.load(); });
    pick = null;
  }
  $("timeline").onclick = (e) => {
    const t = e.target.closest(".tr.pick"), f = e.target.closest(".fxb");
    if (t) openPicker("tr", +t.dataset.i, t);
    else if (f) openPicker("fx", +f.dataset.i, f);
  };
  $("popTabs").onclick = (e) => {
    const b = e.target.closest("button[data-c]");
    if (b && pick) { pick.cat = b.dataset.c; drawPicker(); }
  };
  $("popClose").onclick = closePicker;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && pick) closePicker(); });
  document.addEventListener("mousedown", (e) => {
    if (pick && !$("pop").contains(e.target) && !e.target.closest(".tr.pick, .fxb")) closePicker();
  });
  $("popGrid").onclick = async (e) => {
    const b = e.target.closest(".pv");
    if (!b || !pick) return;
    const name = b.dataset.name, all = $("popAll").checked, kind = pick.kind, i = pick.i;
    const plan = planForEdit();
    const dur = (catalog.categories.flatMap((c) => c.items).find((x) => x.name === name) || {}).duration || 0;
    plan.segments.forEach((s, k) => {
      if (kind === "tr") {
        if (k > 0 && (all || k === i)) { s.transition_in = name; s.transition_duration = dur; }
      } else if (all || k === i) s.effect = name;
    });
    closePicker();
    try {
      await api(`/api/jobs/${state.jobId}/render`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan, options: { quality: state.quality } }) });
      openJob(state.jobId);
    } catch (err) { $("formErr").textContent = err.message; }
  };

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
