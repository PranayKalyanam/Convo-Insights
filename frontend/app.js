// Conversation Insights frontend. Plain JavaScript, no build step.
// Every piece of text from the server is escaped before it is inserted.

const state = { labels: null, conv: { offset: 0, limit: 25, selected: null } };

// ------------------------------------------------------------------ helpers
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nice = (s) => { const t = String(s ?? "").replace(/_/g, " "); return t.charAt(0).toUpperCase() + t.slice(1); };
const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
const tag = (value, cls) => `<span class="tag ${cls || "t-" + esc(value)}">${esc(nice(value))}</span>`;

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (res.status === 204) return null;
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error */ }
  if (!res.ok) {
    let detail = body?.detail ?? res.statusText;
    if (Array.isArray(detail)) detail = detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ");
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return body;
}

function query(params) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== "" && v != null) q.append(k, v);
  const s = q.toString();
  return s ? `?${s}` : "";
}

// ------------------------------------------------------------------ navigation
function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll(".nav-item").forEach((b) => {
    const on = b.dataset.view === name;
    b.classList.toggle("active", on);
    if (on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  if (name === "dashboard") loadDashboard();
  if (name === "conversations") loadConversations();
  if (name === "evaluation") loadEvaluation();
  history.replaceState(null, "", `#${name}`);
}
document.querySelectorAll(".nav-item").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));

// ------------------------------------------------------------------ status
async function loadHealth() {
  const el = $("#provider-status");
  try {
    const h = await api("/api/health");
    el.innerHTML = `Model: <strong>${esc(h.provider)} / ${esc(h.model)}</strong><br>` +
      `${h.conversations_stored} conversations stored` +
      (h.is_mock ? `<span class="warn">Offline mock in use, not a real LLM. Set LLM_PROVIDER in .env and restart the server.</span>` : "") +
      (!h.llm_ready ? `<span class="warn">${esc(h.llm_setup_error)}</span>` : "");
  } catch (e) {
    el.textContent = `Server not reachable: ${e.message}`;
  }
}

$("#btn-test-llm").addEventListener("click", async () => {
  const btn = $("#btn-test-llm"), out = $("#llm-test-result");
  btn.disabled = true; out.textContent = "Sending a test conversation…";
  try {
    const r = await api("/api/llm/test", { method: "POST" });
    out.innerHTML = r.ok
      ? `<span class="ok">Working: ${esc(r.provider)} answered in ${Math.round(r.latency_ms)} ms (intent: ${esc(nice(r.insight.intent))}).</span>` +
        (r.is_mock ? `<span class="warn">This is the mock, not a real LLM.</span>` : "")
      : `<span class="bad">Not working.</span> ${esc(r.advice || r.error || "")}`;
  } catch (e) { out.innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; }
});

// ------------------------------------------------------------------ shared renderers
function renderThread(text) {
  const lines = String(text || "").split("\n").filter((l) => l.trim());
  const turns = lines.map((line) => {
    const m = line.match(/^\s*(agent|customer|speaker)\s*:\s*(.*)$/i);
    const who = m ? m[1].toLowerCase() : "other";
    const body = esc(m ? m[2] : line).replace(/\[(PHONE|EMAIL|UPI|CARD|AADHAAR|PAN)\]/g,
      (_, t) => `<span class="pii" title="Masked before sending to the model">${t.toLowerCase()} hidden</span>`);
    const cls = who === "agent" ? "agent" : who === "customer" ? "customer" : "other";
    const label = m ? nice(who) : "";
    return `<div class="turn ${cls}">${label ? `<span class="who">${label}</span>` : ""}<div class="bubble">${body}</div></div>`;
  });
  return `<div class="thread" aria-label="Conversation">${turns.join("")}</div>`;
}

function renderQA(qa, score) {
  const checks = state.labels?.qa_checks || {};
  const items = Object.entries(checks).map(([k, desc]) => {
    const v = qa?.[k];
    const mark = v ? `<span class="mark yes" aria-label="yes">✓</span>` : `<span class="mark no" aria-label="no">✗</span>`;
    return `<li>${mark}<span>${esc(desc)}</span></li>`;
  }).join("");
  return `<div class="prose-block"><h3>Agent checklist: ${score != null ? Math.round(score) : "–"} / 100</h3><ul class="qa">${items}</ul></div>`;
}

// r: {transcript, summary, intent, sentiment, resolution_status, priority, next_action,
//     agent_qa, qa_score, valid, error, raw_output, provider, model, attempts, latency_ms,
//     pii, conversation_id, saved}
function renderInsight(r, { deletable = false } = {}) {
  const pii = Object.entries(r.pii || {}).map(([k, n]) => `${n} ${k.toLowerCase()}`).join(", ");
  const meta = [
    r.conversation_id ? `Id ${esc(r.conversation_id)}${r.saved === false ? " (not saved)" : ""}` : "Not saved",
    `${esc(r.provider)} / ${esc(r.model)}`,
    `${Math.round(r.latency_ms ?? 0)} ms`,
    `${r.attempts} attempt${r.attempts === 1 ? "" : "s"}`,
    pii ? `masked: ${esc(pii)}` : "no personal data detected",
  ].join(", ");

  if (!r.valid) {
    return `<div class="result">
      <div class="error-box"><strong>The model's answer failed validation.</strong> ${esc(r.error || "")}</div>
      ${renderThread(r.transcript)}
      ${r.raw_output ? `<details open><summary>Raw model output</summary><pre>${esc(r.raw_output)}</pre></details>` : ""}
      <p class="meta">${meta}</p></div>`;
  }
  return `<div class="result">
    <div class="labels">
      <div class="label-cell"><div class="k">Intent</div>${tag(r.intent, "t-intent")}</div>
      <div class="label-cell"><div class="k">Sentiment</div>${tag(r.sentiment)}</div>
      <div class="label-cell"><div class="k">Resolution</div>${tag(r.resolution_status)}</div>
      <div class="label-cell"><div class="k">Priority</div>${tag(r.priority)}</div>
    </div>
    <div class="prose-block"><h3>Summary</h3><p>${esc(r.summary)}</p></div>
    <div class="prose-block next-action"><h3>Next action</h3><p>${esc(r.next_action)}</p></div>
    ${renderQA(r.agent_qa, r.qa_score)}
    ${renderThread(r.transcript)}
    <p class="meta">${meta}</p>
    ${deletable ? `<div><button class="btn ghost small" data-delete="${esc(r.conversation_id)}">Delete this conversation</button></div>` : ""}
  </div>`;
}

// ------------------------------------------------------------------ analyze
$("#btn-example").addEventListener("click", async () => {
  try {
    const ex = await api("/api/examples/random");
    $("#transcript").value = ex.transcript;
    $("#analyze-msg").textContent = `Loaded ${ex.conversation_id} (${ex.language || "unknown language"}).`;
  } catch (e) { $("#analyze-msg").textContent = e.message; }
});

$("#btn-analyze").addEventListener("click", async () => {
  const transcript = $("#transcript").value.trim();
  const msg = $("#analyze-msg");
  if (transcript.length < 10) { msg.textContent = "Paste a transcript of at least 10 characters."; return; }
  const btn = $("#btn-analyze");
  btn.disabled = true; btn.textContent = "Analyzing…"; msg.textContent = "";
  try {
    const d = await api("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, save: $("#save-result").checked }),
    });
    const ins = d.insight || {};
    $("#analyze-result").innerHTML = renderInsight({
      ...ins, transcript: d.masked_transcript, qa_score: d.qa_score, valid: d.valid, error: d.error,
      raw_output: d.raw_output, provider: d.provider, model: d.model, attempts: d.attempts,
      latency_ms: d.latency_ms, pii: d.pii_masked, conversation_id: d.conversation_id, saved: d.saved,
    });
    if (d.saved) loadHealth();
  } catch (e) {
    $("#analyze-result").innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Analyze";
  }
});

// ------------------------------------------------------------------ filters
const FILTERS = [
  ["intent", "All intents", () => Object.keys(state.labels.intents)],
  ["sentiment", "All sentiments", () => state.labels.sentiments],
  ["resolution_status", "All outcomes", () => state.labels.resolution_statuses],
  ["priority", "All priorities", () => state.labels.priorities],
  ["language", "All languages", () => ["english", "hinglish"]],
];

function buildFilters(container, prefix, onChange) {
  for (const [key, allLabel, values] of FILTERS) {
    const sel = document.createElement("select");
    sel.id = `${prefix}-${key}`;
    sel.setAttribute("aria-label", allLabel.replace("All ", "Filter by "));
    sel.innerHTML = `<option value="">${allLabel}</option>` +
      values().map((v) => `<option value="${esc(v)}">${esc(nice(v))}</option>`).join("");
    sel.addEventListener("change", onChange);
    container.appendChild(sel);
  }
}
const filterValues = (prefix) => Object.fromEntries(FILTERS.map(([k]) => [k, $(`#${prefix}-${k}`)?.value || ""]));

// ------------------------------------------------------------------ dashboard
function stackedBar(parts, total) {
  if (!total) return `<div class="bar-track"></div>`;
  return `<div class="bar-track">${parts.map(([cls, n, label]) =>
    n ? `<span class="seg-${cls}" style="width:${(100 * n) / total}%" title="${esc(label)}: ${n}"></span>` : "").join("")}</div>`;
}
const legend = (items) => `<div class="legend">${items.map(([cls, label]) => `<span><i class="seg-${cls}"></i>${label}</span>`).join("")}</div>`;

async function loadDashboard() {
  const body = $("#dash-body");
  let s;
  try { s = await api(`/api/stats${query(filterValues("dash"))}`); }
  catch (e) { body.innerHTML = `<div class="error-box">${esc(e.message)}</div>`; return; }
  if (!s.total) {
    body.innerHTML = `<div class="empty">No conversations match. If the database is empty, process the sample data on the Batch processing page.</div>`;
    return;
  }
  const max = Math.max(...s.by_intent.map((r) => r.total));
  const res = state.labels.resolution_statuses.slice().reverse(); // unresolved first
  const intentRows = s.by_intent.map((r) => `<div class="bar-row"><span class="name" title="${esc(nice(r.intent))}">${esc(nice(r.intent))}</span>
      <div style="width:${(100 * r.total) / max}%">${stackedBar(res.map((k) => [k, r.resolution[k], nice(k)]), r.total)}</div>
      <span class="n">${r.total}</span></div>`).join("");
  const sentRows = s.by_intent.map((r) => `<div class="bar-row"><span class="name">${esc(nice(r.intent))}</span>
      ${stackedBar(["negative", "neutral", "positive"].map((k) => [k, r.sentiment[k], nice(k)]), r.total)}
      <span class="n">${pct(r.sentiment.negative / r.total)}</span></div>`).join("");
  const qaRows = Object.entries(s.qa_pass_rates).map(([k, v]) => `<div class="bar-row"><span class="name" title="${esc(state.labels.qa_checks[k])}">${esc(nice(k))}</span>
      ${stackedBar([["solid", v, "pass rate"]], 1)}<span class="n">${pct(v)}</span></div>`).join("");
  const queue = s.follow_up_queue.map((r) => `<tr data-open="${esc(r.conversation_id)}">
      <td>${esc(r.conversation_id)}</td><td>${tag(r.intent, "t-intent")}</td>
      <td class="summary">${esc(r.summary)}</td><td class="summary">${esc(r.next_action)}</td></tr>`).join("");

  body.innerHTML = `
    <div class="kpis">
      <div class="kpi"><div class="v">${s.total.toLocaleString()}</div><div class="k">Conversations</div></div>
      <div class="kpi"><div class="v">${pct(s.unresolved_rate)}</div><div class="k">Unresolved</div></div>
      <div class="kpi"><div class="v">${pct(s.negative_rate)}</div><div class="k">Negative customers</div></div>
      <div class="kpi"><div class="v">${s.high_priority}</div><div class="k">High priority</div></div>
      <div class="kpi"><div class="v">${Math.round(s.avg_qa_score)}</div><div class="k">Average agent score / 100</div></div>
    </div>
    <div class="charts">
      <div class="panel"><h2>What customers call about, and how it ends</h2>
        <div class="bar-chart">${intentRows}</div>
        ${legend([["unresolved", "Unresolved"], ["partially_resolved", "Partially resolved"], ["resolved", "Resolved"]])}</div>
      <div class="panel"><h2>Sentiment at the end of the call</h2>
        <div class="bar-chart">${sentRows}</div>
        ${legend([["negative", "Negative (share shown)"], ["neutral", "Neutral"], ["positive", "Positive"]])}</div>
      <div class="panel"><h2>Agent checklist pass rate</h2><div class="bar-chart">${qaRows}</div></div>
      <div class="panel"><h2>Languages</h2><div class="bar-chart">${Object.entries(s.languages).map(([k, v]) =>
        `<div class="bar-row"><span class="name">${esc(nice(k))}</span>${stackedBar([["solid", v, k]], s.total)}<span class="n">${v}</span></div>`).join("")}</div></div>
    </div>
    <div class="panel table-panel"><div style="padding:16px 16px 4px"><h2>Follow-up queue</h2>
      <p class="hint">High priority and not resolved. Select a row to open it.</p></div>
      <div class="table-wrap" style="max-height:420px">
      <table class="queue"><thead><tr><th>Id</th><th>Intent</th><th>Summary</th><th>Next action</th></tr></thead>
      <tbody>${queue || `<tr><td colspan="4">Nothing in the queue.</td></tr>`}</tbody></table></div></div>`;

  body.querySelectorAll("tr[data-open]").forEach((tr) => tr.addEventListener("click", () => {
    showView("conversations");
    openConversation(tr.dataset.open);
  }));
}

// ------------------------------------------------------------------ conversations
let searchTimer;
$("#conv-search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.conv.offset = 0; loadConversations(); }, 300);
});
$("#conv-prev").addEventListener("click", () => { state.conv.offset = Math.max(0, state.conv.offset - state.conv.limit); loadConversations(); });
$("#conv-next").addEventListener("click", () => { state.conv.offset += state.conv.limit; loadConversations(); });

async function loadConversations() {
  const { offset, limit } = state.conv;
  const params = { ...filterValues("conv"), q: $("#conv-search").value.trim(), offset, limit };
  let d;
  try { d = await api(`/api/conversations${query(params)}`); }
  catch (e) { $("#conv-table").innerHTML = `<tr><td>${esc(e.message)}</td></tr>`; return; }
  const rows = d.items.map((r) => `<tr data-id="${esc(r.conversation_id)}" class="${r.conversation_id === state.conv.selected ? "selected" : ""}" tabindex="0">
      <td>${esc(r.conversation_id)}</td><td>${tag(r.intent, "t-intent")}</td><td>${tag(r.sentiment)}</td>
      <td>${tag(r.resolution_status)}</td><td>${tag(r.priority)}</td><td class="summary">${esc(r.summary)}</td></tr>`).join("");
  $("#conv-table").innerHTML = `<thead><tr><th>Id</th><th>Intent</th><th>Sentiment</th><th>Resolution</th><th>Priority</th><th>Summary</th></tr></thead>
    <tbody>${rows || `<tr><td colspan="6">No conversations match.</td></tr>`}</tbody>`;
  const end = Math.min(offset + limit, d.total);
  $("#conv-count").textContent = d.total ? `${offset + 1}–${end} of ${d.total}` : "0 results";
  $("#conv-prev").disabled = offset === 0;
  $("#conv-next").disabled = end >= d.total;
  $("#conv-table").querySelectorAll("tbody tr[data-id]").forEach((tr) => {
    const open = () => openConversation(tr.dataset.id);
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  });
}

async function openConversation(id) {
  state.conv.selected = id;
  document.querySelectorAll("#conv-table tbody tr").forEach((tr) => tr.classList.toggle("selected", tr.dataset.id === id));
  const el = $("#conv-detail");
  try {
    const r = await api(`/api/conversations/${encodeURIComponent(id)}`);
    el.innerHTML = renderInsight({ ...r, pii: r.pii_counts }, { deletable: true });
    el.querySelector("[data-delete]")?.addEventListener("click", async () => {
      if (!confirm(`Delete ${id} from the database?`)) return;
      await api(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
      el.innerHTML = `<div class="empty">Deleted ${esc(id)}.</div>`;
      state.conv.selected = null;
      loadConversations(); loadHealth();
    });
  } catch (e) { el.innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
}

// ------------------------------------------------------------------ jobs
async function pollJob(jobId, el, onDone) {
  const render = (j) => {
    const p = j.total ? Math.round((100 * j.done) / j.total) : (j.status === "done" ? 100 : 0);
    const label = { queued: "Queued", running: "Running", done: "Finished", failed: "Failed" }[j.status];
    const counts = j.total ? `${j.done} of ${j.total} processed, ${j.valid} valid, ${j.invalid} failed validation` : "";
    el.hidden = false;
    el.innerHTML = `<h2>${label}${j.kind === "evaluate" ? ": evaluation" : ": processing"}</h2>
      <div class="progress" role="progressbar" aria-valuenow="${p}" aria-valuemin="0" aria-valuemax="100"><span style="width:${p}%"></span></div>
      <p class="hint">${counts}${j.result?.seconds != null ? `, ${j.result.seconds}s` : ""}</p>
      ${j.message ? `<p class="${j.status === "failed" ? "error-box" : "hint"}">${esc(j.message)}</p>` : ""}`;
  };
  for (;;) {
    const j = await api(`/api/jobs/${jobId}`);
    render(j);
    if (j.status === "done" || j.status === "failed") { onDone?.(j); return j; }
    await new Promise((r) => setTimeout(r, 700));
  }
}

$("#btn-upload").addEventListener("click", async () => {
  const file = $("#csv-file").files[0];
  const el = $("#batch-job");
  if (!file) { el.hidden = false; el.innerHTML = `<div class="error-box">Choose a CSV file first.</div>`; return; }
  const fd = new FormData();
  fd.append("file", file);
  fd.append("reprocess", $("#csv-reprocess").checked);
  if ($("#csv-limit").value) fd.append("limit", $("#csv-limit").value);
  try {
    const job = await api("/api/jobs/upload", { method: "POST", body: fd });
    pollJob(job.job_id, el, loadHealth);
  } catch (e) { el.hidden = false; el.innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
});

$("#btn-sample").addEventListener("click", async () => {
  const el = $("#batch-job");
  const params = query({ limit: $("#sample-limit").value, reprocess: $("#sample-reprocess").checked || "" });
  try {
    const job = await api(`/api/jobs/process-sample${params}`, { method: "POST" });
    pollJob(job.job_id, el, loadHealth);
  } catch (e) { el.hidden = false; el.innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
});

// ------------------------------------------------------------------ evaluation
$("#btn-eval").addEventListener("click", async () => {
  const btn = $("#btn-eval");
  btn.disabled = true;
  $("#eval-msg").textContent = "Any test conversations not yet processed will be sent to the model first.";
  try {
    const job = await api("/api/jobs/evaluate", { method: "POST" });
    await pollJob(job.job_id, $("#eval-job"), (j) => { if (j.status === "done") { loadEvaluation(); loadHealth(); } });
  } catch (e) {
    $("#eval-job").hidden = false;
    $("#eval-job").innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
  } finally { btn.disabled = false; $("#eval-msg").textContent = ""; }
});

const f3 = (x) => (x == null ? "–" : Number(x).toFixed(3));

async function loadEvaluation() {
  const body = $("#eval-body");
  let m;
  try { m = await api("/api/evaluation"); }
  catch (e) {
    body.innerHTML = e.status === 404
      ? `<div class="empty" style="margin-top:16px">No evaluation yet. Select Run evaluation to score the model on the test set.</div>`
      : `<div class="error-box">${esc(e.message)}</div>`;
    return;
  }
  const rows = Object.entries(m.llm).map(([k, s]) => {
    const b = m.baseline[k];
    const win = b && s.macro_f1 > b.macro_f1;
    return `<tr><td>${esc(nice(k))}</td><td>${f3(s.accuracy)}</td><td class="${win ? "better" : ""}">${f3(s.macro_f1)}</td>
      <td>${b ? f3(b.accuracy) : "–"}</td><td class="${b && !win && b.macro_f1 > s.macro_f1 ? "better" : ""}">${b ? f3(b.macro_f1) : "–"}</td></tr>`;
  }).join("");
  const qa = m.agent_qa_accuracy ? Object.entries(m.agent_qa_accuracy).map(([k, v]) =>
    `<tr><td>${esc(state.labels.qa_checks[k] || k)}</td><td>${f3(v)}</td></tr>`).join("") : "";
  const lang = m.intent_by_language ? Object.entries(m.intent_by_language).map(([k, v]) =>
    `<tr><td>${esc(nice(k))}</td><td>${v.n}</td><td>${f3(v.accuracy)}</td><td>${f3(v.macro_f1)}</td></tr>`).join("") : "";
  const o = m.operational;

  let cm = "";
  if (m.confusion) {
    const { labels_gold: g, labels_pred: p, matrix } = m.confusion;
    const max = Math.max(1, ...matrix.flat());
    cm = `<div class="panel wide"><h2>Intent confusion matrix</h2><p class="hint">Rows are the correct intent, columns the predicted one. Off-diagonal cells are mistakes.</p>
      <div class="table-wrap"><table class="cm"><thead><tr><th></th>${p.map((l) => `<th title="${esc(l)}">${esc(nice(l)).slice(0, 12)}</th>`).join("")}</tr></thead><tbody>
      ${g.map((row, i) => `<tr><th class="rowh">${esc(nice(row))}</th>${matrix[i].map((v, j) => {
        const a = v / max; const diag = row === p[j];
        const bg = v === 0 ? "#F4F6F9" : diag ? `rgba(47,125,79,${0.15 + 0.7 * a})` : `rgba(184,58,50,${0.15 + 0.7 * a})`;
        return `<td style="background:${bg};color:${a > 0.55 ? "#fff" : "inherit"}">${v || ""}</td>`; }).join("")}</tr>`).join("")}
      </tbody></table></div></div>`;
  }

  body.innerHTML = `
    ${m.provider.includes("mock") ? `<div class="notice">These scores come from the offline keyword mock, not an LLM. Switch LLM_PROVIDER in .env, restart the server and run the evaluation again.</div>` : ""}
    <p class="hint" style="margin-top:16px">${esc(m.provider)} / ${esc(m.model)} on ${m.n_test} test conversations, ${esc(m.generated_at)}.</p>
    <div class="eval-grid">
      <div class="panel wide"><h2>Model vs TF-IDF baseline</h2>
        <div class="table-wrap"><table><thead><tr><th>Label</th><th>Model accuracy</th><th>Model macro-F1</th><th>Baseline accuracy</th><th>Baseline macro-F1</th></tr></thead><tbody>${rows}</tbody></table></div>
        <p class="hint" style="margin-top:10px">Invalid model outputs count as wrong. The baseline has no priority model. On template data the baseline is unrealistically strong; see the README on building a hand-labelled test set.</p></div>
      <div class="panel"><h2>Reliability and speed</h2><table><tbody>
        <tr><td>Invalid output after retries</td><td>${pct(o.invalid_output_rate)}</td></tr>
        <tr><td>Needed a retry</td><td>${o.needed_retry_rate == null ? "–" : pct(o.needed_retry_rate)}</td></tr>
        <tr><td>Model latency, mean</td><td>${o.latency_ms_mean ?? "–"} ms</td></tr>
        <tr><td>Model latency, 95th percentile</td><td>${o.latency_ms_p95 ?? "–"} ms</td></tr>
        <tr><td>Baseline latency, mean</td><td>${m.baseline.latency_ms_mean} ms</td></tr>
        <tr><td>Rows with any wrong label</td><td>${m.n_rows_with_errors}</td></tr></tbody></table></div>
      <div class="panel"><h2>Intent by language</h2><table><thead><tr><th>Language</th><th>n</th><th>Accuracy</th><th>Macro-F1</th></tr></thead><tbody>${lang}</tbody></table></div>
      ${qa ? `<div class="panel"><h2>Agent checklist agreement</h2><table><tbody>${qa}</tbody></table></div>` : ""}
      ${cm}
    </div>`;
}

// ------------------------------------------------------------------ init
(async function init() {
  loadHealth();
  try {
    state.labels = await api("/api/labels");
  } catch (e) {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<div class="error-box">Could not reach the API: ${esc(e.message)}. Is the server running?</div>`);
    return;
  }
  buildFilters($("#dash-filters"), "dash", loadDashboard);
  buildFilters($("#conv-filters"), "conv", () => { state.conv.offset = 0; loadConversations(); });
  const start = location.hash.slice(1);
  showView(["analyze", "dashboard", "conversations", "batch", "evaluation"].includes(start) ? start : "analyze");
})();
