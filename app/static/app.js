const view = document.getElementById("view");
const toastEl = document.getElementById("toast");
const zoom = document.getElementById("zoom");
let zoomRecord = null;
let compareA = null;
let compareB = null;
let picks = [];

function gpuConfirmMessage(host) {
  return (
    "Submit jobs to ComfyUI on Zach's PC?\n\n" +
    "This uses that PC's GPU. Nothing is sent until you confirm.\n" +
    "Host: " + host + "\n\n" +
    "Choose OK only if you intend to run these jobs."
  );
}

function toast(message) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  clearTimeout(toastEl._timer);
  toastEl._timer = setTimeout(() => toastEl.classList.remove("show"), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (_error) {
    data = { detail: text };
  }
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

async function upload(path, file) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(path, { method: "POST", body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "upload failed");
  return data;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function panel(title) {
  const box = el("section", "panel");
  if (title) box.append(el("h2", null, title));
  return box;
}

function setPill(settings) {
  const pill = document.getElementById("safety-pill");
  if (settings.comfyui_enabled) {
    pill.textContent = "ComfyUI on · still needs confirm";
    pill.classList.add("warn");
  } else {
    pill.textContent = "ComfyUI off";
    pill.classList.remove("warn");
  }
}

function tabFromLocation() {
  const params = new URLSearchParams(location.search);
  return params.get("tab") || "dashboard";
}

function showTab(name) {
  toastEl.classList.remove("show");
  document.querySelectorAll("#tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  const params = new URLSearchParams(location.search);
  params.set("tab", name);
  history.replaceState(null, "", "?" + params.toString());
  const renderers = {
    dashboard: renderDashboard,
    epochs: renderEpochs,
    variance: renderVariance,
    bleed: renderBleed,
    merge: renderMerge,
    provenance: renderProvenance,
    profiler: renderProfiler,
    index: renderIndex,
    shelf: renderShelf,
    settings: renderSettings,
  };
  Promise.resolve((renderers[name] || renderDashboard)()).catch((error) => {
    clear(view);
    view.append(el("p", null, error.message));
  });
}

document.getElementById("tabs").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-tab]");
  if (button) showTab(button.dataset.tab);
});

function stat(label, value) {
  const box = el("div", "card stat");
  box.append(el("strong", null, String(value)));
  box.append(el("span", "muted", label));
  return box;
}

async function renderDashboard() {
  const data = await api("/api/dashboard");
  setPill(data);
  clear(view);
  const stats = el("div", "stat-row");
  stats.append(
    stat("Sample images", data.sample_images),
    stat("Epochs", data.epochs),
    stat("Prompts", data.prompts),
    stat("Keepers", data.picks),
    stat("Sample LoRAs", data.sample_loras),
    stat("Indexed", data.indexed_loras),
  );
  view.append(stats);
  const flags = panel("Flags in the synthetic set");
  flags.append(el("p", null, "Unstable epochs: " + (data.unstable_epochs.join(", ") || "none")));
  flags.append(el("p", null, "Style-bleed flags: " + (data.bleed_flags.join(", ") || "none")));
  flags.append(el("p", "muted", "ComfyUI calls on this page: none. As of " + data.today + "."));
  view.append(flags);
  const cards = el("div", "cards");
  data.modules.forEach((module) => {
    const card = el("article", "card");
    card.append(el("h2", null, module.name));
    card.append(el("p", null, module.summary));
    const button = el("button", "ghost", "Open");
    button.type = "button";
    button.addEventListener("click", () => showTab(module.id));
    card.append(button);
    cards.append(card);
  });
  view.append(cards);
}

function openZoom(record) {
  zoomRecord = record;
  document.getElementById("zoom-title").textContent = record.epoch + " · " + record.prompt_id + " · seed " + record.seed;
  const image = document.getElementById("zoom-image");
  image.src = record.url;
  image.alt = record.prompt_id + " seed " + record.seed;
  document.getElementById("zoom-meta").textContent = record.text || "";
  const existing = picks.find((pick) => pick.epoch === record.epoch && pick.prompt_id === record.prompt_id && pick.seed === record.seed);
  document.getElementById("zoom-notes").value = existing ? existing.notes : "";
  zoom.showModal();
}

document.getElementById("zoom-save").addEventListener("click", async () => {
  if (!zoomRecord) return;
  try {
    const saved = await api("/api/epochs/picks", {
      method: "POST",
      body: JSON.stringify({
        epoch: zoomRecord.epoch,
        prompt_id: zoomRecord.prompt_id,
        seed: zoomRecord.seed,
        path: zoomRecord.path,
        notes: document.getElementById("zoom-notes").value,
      }),
    });
    picks = saved.picks;
    toast("Keeper saved locally in picks.json.");
    zoom.close();
    showTab("epochs");
  } catch (error) {
    toast(error.message);
  }
});

document.getElementById("zoom-a").addEventListener("click", () => {
  compareA = zoomRecord;
  toast("Compare A set.");
  zoom.close();
  showTab("epochs");
});

document.getElementById("zoom-b").addEventListener("click", () => {
  compareB = zoomRecord;
  toast("Compare B set.");
  zoom.close();
  showTab("epochs");
});

function renderCompare() {
  if (!compareA || !compareB) return null;
  const wrap = el("div", "compare");
  [compareA, compareB].forEach((record, index) => {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = record.url;
    image.alt = (index === 0 ? "Compare A " : "Compare B ") + record.prompt_id;
    const caption = document.createElement("figcaption");
    caption.textContent = (index === 0 ? "A · " : "B · ") + record.epoch + " · " + record.prompt_id + " · seed " + record.seed;
    figure.append(image, caption);
    wrap.append(figure);
  });
  return wrap;
}

async function renderEpochs() {
  const [grid, pickData] = await Promise.all([api("/api/epochs/grid"), api("/api/epochs/picks")]);
  picks = pickData.picks || [];
  clear(view);
  const intro = panel("Epoch Picker");
  intro.append(el("p", "muted", "Rows are prompts, columns are epochs. Click a frame to zoom. A and B sit side by side. Saving a keeper writes picks.json only."));
  const actions = el("div", "row");
  const queueButton = el("button", null, "Write queue file");
  queueButton.type = "button";
  queueButton.addEventListener("click", async () => {
    try {
      const result = await api("/api/epochs/queue", { method: "POST", body: "{}" });
      toast(result.message + " " + result.count + " workflows.");
    } catch (error) {
      toast(error.message);
    }
  });
  const submitButton = el("button", "ghost", "Submit to ComfyUI");
  submitButton.type = "button";
  submitButton.addEventListener("click", submitQueue);
  actions.append(queueButton, submitButton);
  intro.append(actions);
  view.append(intro);
  const compare = renderCompare();
  if (compare) view.append(compare);

  const sheetWrap = el("div", "sheet-wrap");
  const sheet = el("div", "sheet");
  sheet.style.gridTemplateColumns = "minmax(140px, 180px) repeat(" + grid.epochs.length + ", minmax(140px, 1fr))";
  sheet.append(el("div", "corner", "Prompt"));
  grid.epochs.forEach((epoch) => sheet.append(el("div", "colhead", epoch.replace("_", " "))));
  grid.prompts.forEach((prompt) => {
    const label = el("div", "rowhead", prompt.text);
    sheet.append(label);
    grid.epochs.forEach((epoch) => {
      const cellData = grid.cells.find((cell) => cell.epoch === epoch && cell.prompt_id === prompt.id);
      const cell = el("div", "cell");
      (cellData ? cellData.images : []).forEach((image) => {
        const button = document.createElement("button");
        button.type = "button";
        const kept = picks.some((pick) => pick.path === image.path);
        if (kept) button.classList.add("picked");
        const img = document.createElement("img");
        img.src = image.url;
        img.alt = prompt.id + " " + epoch + " seed " + image.seed;
        button.append(img);
        button.addEventListener("click", () => openZoom({
          ...image,
          epoch,
          prompt_id: prompt.id,
          text: prompt.text,
        }));
        cell.append(button);
      });
      sheet.append(cell);
    });
  });
  sheetWrap.append(sheet);
  view.append(sheetWrap);

  const list = panel("Keepers");
  if (!picks.length) list.append(el("p", "muted", "No keepers yet."));
  picks.forEach((pick) => {
    list.append(el("p", null, pick.epoch + " · " + pick.prompt_id + " · seed " + pick.seed + (pick.notes ? " — " + pick.notes : "")));
  });
  view.append(list);
}

async function submitQueue() {
  try {
    const settings = await api("/api/settings");
    setPill(settings);
    if (!settings.comfyui_enabled) {
      toast("ComfyUI is off. Nothing was sent. Enable it in Settings first.");
      return;
    }
    if (!settings.comfyui_host) {
      toast("ComfyUI host is blank. Nothing was sent.");
      return;
    }
    const ok = window.confirm(gpuConfirmMessage(settings.comfyui_host));
    if (!ok) {
      toast("Submit cancelled. ComfyUI was not contacted.");
      return;
    }
    const result = await api("/api/epochs/submit", {
      method: "POST",
      body: JSON.stringify({ confirm: true }),
    });
    toast("Submitted " + result.count + " workflows. " + result.warning);
  } catch (error) {
    toast(error.message);
  }
}

async function renderVariance() {
  const data = await api("/api/variance");
  clear(view);
  const box = panel("Seed variance band");
  box.append(el("p", "muted", "Method: " + data.method + ". An epoch is unstable when mean similarity drops below " + data.mean_floor + " or the spread rises above " + data.std_ceiling + "."));
  data.epochs.forEach((epoch) => {
    const row = el("div", "band");
    const name = el("div", null, epoch.epoch);
    const track = el("div", "track");
    const meanPct = Math.max(0, Math.min(1, epoch.mean)) * 100;
    const low = Math.max(0, Math.min(1, epoch.low)) * 100;
    const high = Math.max(0, Math.min(1, epoch.high)) * 100;
    const fill = el("div", "fill");
    fill.style.width = meanPct + "%";
    const range = el("div", "range");
    range.style.left = low + "%";
    range.style.width = Math.max(0, high - low) + "%";
    const mean = el("div", "mean");
    mean.style.left = meanPct + "%";
    track.append(fill, range, mean);
    const label = el("div", null, epoch.mean.toFixed(3) + (epoch.unstable ? " unstable" : " stable"));
    row.append(name, track, label);
    box.append(row);
  });
  view.append(box);
  const tableBox = panel("Per prompt");
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>Epoch</th><th>Prompt</th><th>Mean</th><th>Spread</th></tr></thead>";
  const body = document.createElement("tbody");
  data.prompts.forEach((row) => {
    const tr = document.createElement("tr");
    [row.epoch, row.prompt_id, row.mean.toFixed(3), row.std.toFixed(3)].forEach((value) => {
      tr.append(el("td", null, value));
    });
    body.append(tr);
  });
  table.append(body);
  tableBox.append(table);
  view.append(tableBox);
}

async function renderBleed() {
  const data = await api("/api/bleed");
  clear(view);
  const intro = panel("Style bleed");
  intro.append(el("p", "muted", "Leakage is 1 − similarity between the with-LoRA and without-LoRA pair. Flagged above " + data.leakage_ceiling + ". Method: " + data.method + "."));
  view.append(intro);
  data.epochs.forEach((epoch) => {
    const box = panel(epoch.epoch);
    const flag = el("span", "flag " + (epoch.flagged ? "bad" : "good"), epoch.flagged ? "leakage flagged" : "within threshold");
    box.append(flag);
    box.append(el("p", null, "Mean leakage " + epoch.leakage.toFixed(3)));
    epoch.pairs.forEach((pair) => {
      const wrap = el("div", "compare");
      [["with LoRA", pair.with_lora], ["without LoRA", pair.without_lora]].forEach(([label, path]) => {
        const figure = document.createElement("figure");
        const image = document.createElement("img");
        image.src = "/media/" + path;
        image.alt = label + " " + pair.prompt_id;
        const caption = document.createElement("figcaption");
        caption.textContent = pair.prompt_id + " · " + label + " · leakage " + pair.leakage.toFixed(3);
        figure.append(image, caption);
        wrap.append(figure);
      });
      box.append(wrap);
    });
    view.append(box);
  });
}

async function renderMerge() {
  clear(view);
  const form = panel("New recipe");
  form.append(el("p", "muted", "Saving and replaying write files only. Replay does not run the merge and does not call ComfyUI. CPU merge is a separate button and stays on this machine."));
  const name = field("Name", "text", "demo_blend");
  const method = document.createElement("select");
  ["weighted_sum", "add_difference", "dare_linear"].forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    method.append(option);
  });
  const methodLabel = el("label", null, "Method");
  methodLabel.append(method);
  const left = field("Source A path", "text", "sample_data/loras/style_amber.safetensors");
  const leftWeight = field("Weight A", "number", "0.6");
  const right = field("Source B path", "text", "sample_data/loras/style_teal.safetensors");
  const rightWeight = field("Weight B", "number", "0.4");
  const save = el("button", null, "Save recipe");
  save.type = "button";
  form.append(name.label, methodLabel, left.label, leftWeight.label, right.label, rightWeight.label, save);
  view.append(form);
  const list = panel("Saved recipes");
  view.append(list);

  async function refresh() {
    const data = await api("/api/recipes");
    clear(list);
    list.append(el("h2", null, "Saved recipes"));
    if (!data.recipes.length) list.append(el("p", "muted", "None yet."));
    data.recipes.forEach((recipe) => {
      const block = el("div", "card");
      block.append(el("h2", null, recipe.name));
      block.append(el("p", null, recipe.method + " · " + recipe.sources.map((source) => source.weight + " " + source.path.split("/").pop()).join(" + ")));
      const row = el("div", "row");
      const replay = el("button", null, "Write CPU script");
      replay.type = "button";
      replay.addEventListener("click", async () => {
        const result = await api("/api/recipes/" + encodeURIComponent(recipe.name) + "/replay", { method: "POST", body: "{}" });
        toast("Script written. executed=" + result.executed + ". Nothing was launched.");
      });
      const run = el("button", "ghost", "Run CPU merge");
      run.type = "button";
      run.addEventListener("click", async () => {
        const ok = window.confirm("Run a weighted sum on this computer's CPU? This does not contact ComfyUI and does not use a GPU.");
        if (!ok) return;
        try {
          const result = await api("/api/recipes/" + encodeURIComponent(recipe.name) + "/merge-cpu", { method: "POST", body: "{}" });
          toast("CPU merge wrote " + result.path.split("/").pop() + ".");
        } catch (error) {
          toast(error.message);
        }
      });
      row.append(replay, run);
      block.append(row);
      list.append(block);
    });
  }

  save.addEventListener("click", async () => {
    try {
      await api("/api/recipes", {
        method: "POST",
        body: JSON.stringify({
          name: name.input.value,
          method: method.value,
          sources: [
            { path: left.input.value, weight: Number(leftWeight.input.value) },
            { path: right.input.value, weight: Number(rightWeight.input.value) },
          ],
          notes: "Synthetic recipe. Replay writes a CPU script and does not run it.",
        }),
      });
      toast("Recipe saved. No merge was run.");
      refresh();
    } catch (error) {
      toast(error.message);
    }
  });
  refresh();
}

function field(labelText, type, value) {
  const label = el("label", null, labelText);
  const input = document.createElement("input");
  input.type = type;
  input.value = value;
  if (type === "number") input.step = "0.05";
  label.append(input);
  return { label, input };
}

function recordTable(records) {
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>Image</th><th>Seed</th><th>Sampler</th><th>LoRAs</th><th>Prompt</th><th>When</th></tr></thead>";
  const body = document.createElement("tbody");
  records.forEach((record) => {
    const tr = document.createElement("tr");
    const loras = (record.loras || []).map((lora) => lora.name + " @" + lora.strength_model).join(", ");
    [record.image || "", record.seed ?? "", record.sampler || "", loras, record.prompt_text || "", record.executed_at || record.source].forEach((value) => {
      tr.append(el("td", null, String(value)));
    });
    body.append(tr);
  });
  table.append(body);
  return table;
}

async function renderProvenance() {
  clear(view);
  const box = panel("Output provenance");
  box.append(el("p", "muted", "Parsed from PNG text chunks and the sample history file. Fetching a live history is a GET and still requires the Settings toggle plus confirm."));
  const search = field("Search", "search", "");
  const row = el("div", "row");
  const file = document.createElement("input");
  file.type = "file";
  file.accept = ".png,.json,application/json,image/png";
  const fetchButton = el("button", "ghost", "Fetch /history");
  fetchButton.type = "button";
  fetchButton.addEventListener("click", () => fetchRemote("provenance"));
  row.append(fetchButton);
  box.append(search.label, file, row);
  const results = el("div");
  box.append(results);
  view.append(box);

  let records = [];
  function draw() {
    const needle = search.input.value.trim().toLowerCase();
    const matched = records.filter((record) => JSON.stringify(record).toLowerCase().includes(needle));
    clear(results);
    results.append(el("p", "muted", matched.length + " rows"));
    results.append(recordTable(matched.slice(0, 80)));
  }
  search.input.addEventListener("input", draw);
  file.addEventListener("change", async () => {
    if (!file.files[0]) return;
    const data = await upload("/api/provenance/upload", file.files[0]);
    records = data.records.concat(records);
    draw();
  });
  records = (await api("/api/provenance")).records;
  draw();
}

async function fetchRemote(kind) {
  try {
    const settings = await api("/api/settings");
    setPill(settings);
    if (!settings.comfyui_enabled || !settings.comfyui_host) {
      toast("Nothing was sent. Enable ComfyUI and set a host in Settings first.");
      return;
    }
    const ok = window.confirm(
      "Read history from ComfyUI?\n\nThis sends one GET /history to " +
      settings.comfyui_host +
      ".\nIt does not submit a job. Choose OK to confirm this read."
    );
    if (!ok) {
      toast("Fetch cancelled. ComfyUI was not contacted.");
      return;
    }
    const path = kind === "profiler" ? "/api/profiler/fetch-history" : "/api/provenance/fetch-history";
    const result = await api(path, {
      method: "POST",
      body: JSON.stringify({ confirm: true, user_initiated: true }),
    });
    toast("GET finished. submitted=" + result.submitted + ".");
    if (kind === "profiler" && result.profile) drawProfile(result.profile);
  } catch (error) {
    toast(error.message);
  }
}

function drawProfile(data) {
  clear(view);
  const box = panel("Slowest node types");
  box.append(el("p", "muted", "Read-only. Slowest type: " + (data.slowest_type || "none") + "."));
  const max = Math.max(...data.by_type.map((row) => row.total_ms), 1);
  data.by_type.forEach((row) => {
    const line = el("div", "bar-row");
    const bar = el("div", "bar");
    bar.style.width = (100 * row.total_ms / max) + "%";
    const holder = el("div");
    holder.append(bar);
    line.append(el("div", null, row.class_type), holder, el("div", null, Math.round(row.total_ms) + " ms"));
    box.append(line);
  });
  view.append(box);
  const nodes = panel("By node id");
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>Node</th><th>Type</th><th>Total ms</th></tr></thead>";
  const body = document.createElement("tbody");
  data.by_node.forEach((row) => {
    const tr = document.createElement("tr");
    [row.node_id, row.class_type, String(Math.round(row.total_ms))].forEach((value) => tr.append(el("td", null, value)));
    body.append(tr);
  });
  table.append(body);
  nodes.append(table);
  view.append(nodes);
}

async function renderProfiler() {
  const data = await api("/api/profiler");
  drawProfile(data);
  const row = el("div", "row");
  const fetchButton = el("button", "ghost", "Fetch /history");
  fetchButton.type = "button";
  fetchButton.addEventListener("click", () => fetchRemote("profiler"));
  const file = document.createElement("input");
  file.type = "file";
  file.accept = ".json,application/json";
  file.addEventListener("change", async () => {
    if (!file.files[0]) return;
    drawProfile(await upload("/api/profiler/upload", file.files[0]));
  });
  row.append(fetchButton, file);
  view.prepend(row);
}

async function renderIndex() {
  clear(view);
  const box = panel("LoRA hash index");
  box.append(el("p", "muted", "Reads .safetensors headers and hashes. The catalog is SQLite. Model files are left where they are."));
  const folder = field("Folder", "text", "sample_data/loras");
  const scan = el("button", null, "Scan folder");
  scan.type = "button";
  const tableHost = el("div");
  box.append(folder.label, scan, tableHost);
  view.append(box);
  async function load(path) {
    const data = path
      ? await api("/api/index/scan", { method: "POST", body: JSON.stringify({ folder: path }) })
      : await api("/api/index");
    clear(tableHost);
    tableHost.append(el("p", "muted", (data.count ?? data.loras.length) + " files. deleted=" + (data.deleted || []).length));
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>File</th><th>AutoV2</th><th>SHA256</th><th>Base</th><th>Triggers</th><th>Meta</th></tr></thead>";
    const body = document.createElement("tbody");
    data.loras.forEach((lora) => {
      const tr = document.createElement("tr");
      const meta = lora.missing_metadata ? "missing" : "present";
      [lora.filename, lora.autov2, lora.sha256.slice(0, 12), lora.base_model || "", (lora.trigger_words || []).join(", "), meta].forEach((value) => {
        tr.append(el("td", null, value));
      });
      body.append(tr);
    });
    table.append(body);
    tableHost.append(table);
  }
  scan.addEventListener("click", () => load(folder.input.value).catch((error) => toast(error.message)));
  load("sample_data/loras").catch((error) => toast(error.message));
}

async function renderShelf() {
  clear(view);
  const box = panel("Shelf life");
  box.append(el("p", "muted", "Cross-references the hash index with dated provenance. Suggestions only — the report's deleted list stays empty. The sample history last uses style_amber on 2026-10-28 and style_teal on 2026-08-01."));
  const days = field("Unused after days", "number", "30");
  const today = field("As of", "date", "");
  const button = el("button", null, "Refresh suggestions");
  button.type = "button";
  const host = el("div");
  box.append(days.label, today.label, button, host);
  view.append(box);
  async function load() {
    const params = new URLSearchParams({ unused_days: days.input.value || "30", folder: "sample_data/loras" });
    if (today.input.value) params.set("today", today.input.value);
    const data = await api("/api/shelf?" + params.toString());
    if (!today.input.value) today.input.value = data.today;
    clear(host);
    host.append(el("p", null, data.suggestions.length + " suggestions. Files unchanged: " + data.files_unchanged + ". Deleted: " + data.deleted.length + "."));
    data.suggestions.forEach((item) => {
      const card = el("article", "card");
      card.append(el("h2", null, item.filename));
      card.append(el("p", null, item.reasons.join(", ") + " · " + item.action));
      card.append(el("p", "muted", item.detail));
      host.append(card);
    });
    if (data.duplicates.length) {
      const dup = panel("Duplicate groups");
      data.duplicates.forEach((group) => {
        dup.append(el("p", null, group.hash_type + " " + group.hash + " → " + group.paths.map((path) => path.split("/").pop()).join(", ")));
      });
      host.append(dup);
    }
  }
  button.addEventListener("click", () => load().catch((error) => toast(error.message)));
  load().catch((error) => toast(error.message));
}

async function renderSettings() {
  const settings = await api("/api/settings");
  setPill(settings);
  clear(view);
  const box = panel("ComfyUI");
  box.append(el("p", null, "Submission is off by default. The host box starts empty. The documented example, Zach's PC at 192.168.4.47:8188, is a placeholder, not a saved default."));
  box.append(el("p", "muted", "Turning the toggle on does not send anything. Every submit still opens a confirm dialog that names Zach's PC and the GPU. History reads are GET /history and need the same toggle plus confirm. They never POST."));
  const toggle = document.createElement("input");
  toggle.type = "checkbox";
  toggle.checked = settings.comfyui_enabled === true;
  const toggleLabel = el("label", "check");
  toggleLabel.append(toggle, document.createTextNode("Enable ComfyUI connections"));
  const host = field("Host", "text", settings.comfyui_host || "");
  host.input.placeholder = "example: 192.168.4.47:8188";
  const save = el("button", null, "Save settings");
  save.type = "button";
  save.addEventListener("click", async () => {
    const saved = await api("/api/settings", {
      method: "POST",
      body: JSON.stringify({ comfyui_enabled: toggle.checked, comfyui_host: host.input.value.trim() }),
    });
    setPill(saved);
    toast(saved.comfyui_enabled ? "Toggle on. Nothing has been sent." : "ComfyUI off. Nothing has been sent.");
  });
  box.append(toggleLabel, host.label, save);
  view.append(box);
}

showTab(tabFromLocation());
