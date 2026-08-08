const state = { snapshot: null, activeSession: null, mode: "voice", oauthTimer: null, lastMessageCount: -1 };
const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, cache: "no-store", ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  setTimeout(() => element.classList.remove("show"), 2800);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderSessions() {
  const list = $("#session-list");
  list.replaceChildren();
  for (const session of state.snapshot.sessions) {
    const button = el("button", `session ${session.id === state.activeSession ? "active" : ""}`, session.title);
    button.onclick = () => selectSession(session.id);
    list.append(button);
  }
  const active = state.snapshot.sessions.find((item) => item.id === state.activeSession);
  $("#session-title").textContent = active?.title || "Session";
}

async function fetchMessages(id) {
  const payload = await api(`/api/sessions/${encodeURIComponent(id)}/messages`);
  return payload.messages.filter((item) => item.kind === "message");
}

function renderMessages(messages) {
  const list = $("#message-list");
  list.replaceChildren();
  $("#empty-state").classList.toggle("hidden", messages.length > 0);
  for (const message of messages) {
    const card = el("article", `message ${message.role}`);
    card.append(el("div", "", message.content));
    card.append(el("div", "message-meta", `${message.source || "system"}${message.model ? ` · ${message.model}` : ""}`));
    list.append(card);
  }
  state.lastMessageCount = messages.length;
  list.lastElementChild?.scrollIntoView({ behavior: "smooth" });
}

async function selectSession(id) {
  state.activeSession = id;
  renderSessions();
  renderMessages(await fetchMessages(id));
}

// Called on every poll tick so messages appended in the background (approvals
// resolving, a delegated GPT-OSS run finishing, a voice turn coming in) show
// up without the user having to click the session again.
async function syncActiveMessages() {
  if (!state.activeSession) return;
  try {
    const messages = await fetchMessages(state.activeSession);
    if (messages.length !== state.lastMessageCount) renderMessages(messages);
  } catch (error) {
    // Transient poll failure -- next tick will retry. Don't toast every 1.5s.
  }
}

function renderApprovals() {
  const pending = state.snapshot.approvals.filter((item) => item.status === "pending");
  $("#approval-count").textContent = pending.length;
  const list = $("#approval-list");
  list.replaceChildren();
  if (!pending.length) return list.append(el("div", "muted-card", "No actions waiting"));
  for (const approval of pending) {
    const card = el("div", "approval-card");
    card.append(el("strong", "", approval.tool.replaceAll("_", " ")));
    card.append(el("div", "", JSON.stringify(approval.arguments, null, 2)));
    const actions = el("div", "approval-actions");
    const approve = el("button", "approve", "Approve once");
    const reject = el("button", "reject", "Reject");
    approve.onclick = () => resolveApproval(approval.id, true);
    reject.onclick = () => resolveApproval(approval.id, false);
    actions.append(approve, reject); card.append(actions); list.append(card);
  }
}

async function resolveApproval(id, approve) {
  try {
    const result = await api(`/api/approvals/${id}`, { method: "POST", body: JSON.stringify({ approve }) });
    toast(approve ? `Action ${result.status}` : "Action rejected");
    await refresh(); await selectSession(state.activeSession);
  } catch (error) { toast(error.message); }
}

function renderUsage() {
  const list = $("#usage-list"); list.replaceChildren();
  const rows = Object.entries(state.snapshot.usage);
  if (!rows.length) return list.append(el("div", "muted-card", "No model requests in this session yet."));
  const max = Math.max(...rows.map(([, value]) => value.input_tokens + value.output_tokens), 1);
  for (const [model, usage] of rows) {
    const row = el("div", "usage-row");
    const head = el("div", "usage-head"); head.append(el("span", "", model), el("span", "", `${usage.requests} req · ${usage.input_tokens + usage.output_tokens} tok`));
    const bar = el("div", "usage-bar"); const fill = el("i"); fill.style.width = `${Math.max(4, (usage.input_tokens + usage.output_tokens) / max * 100)}%`; bar.append(fill);
    row.append(head, bar); list.append(row);
  }
}

function renderToggles(items, target, endpoint) {
  const list = $(target); list.replaceChildren();
  for (const item of items) {
    const card = el("div", "data-card toggle-row");
    const copy = el("div"); copy.append(el("strong", "", item.name.replaceAll("_", " ")), el("span", "", item.url || `${item.mode || "skill"} · ${item.model || "configured"} · ${(item.triggers || []).join(", ")}`));
    const toggle = el("input", "toggle"); toggle.type = "checkbox"; toggle.checked = item.enabled;
    toggle.onchange = async () => { try { await api(`${endpoint}/${encodeURIComponent(item.name)}/toggle`, { method: "POST", body: JSON.stringify({ enabled: toggle.checked }) }); toast(`${item.name} ${toggle.checked ? "enabled" : "disabled"}`); } catch (error) { toggle.checked = !toggle.checked; toast(error.message); } };
    card.append(copy, toggle); list.append(card);
  }
}

function renderSnapshot() {
  const snap = state.snapshot;
  state.activeSession ||= snap.active_session_id;
  renderSessions(); renderApprovals(); renderUsage();
  renderToggles(snap.mcp_servers, "#mcp-list", "/api/mcp");
  renderToggles(snap.skills, "#skill-list", "/api/skills");
  $("#github-dot").classList.toggle("on", snap.github.connected);
  $("#github-label").textContent = snap.github.connected ? `GitHub connected · ${snap.github.source}` : "GitHub disconnected";
  $("#oauth-status").textContent = snap.github.connected ? "Connected" : "Disconnected";
  $("#oauth-status").classList.toggle("on", snap.github.connected);
  $("#github-key-status").textContent = snap.github.connected ? "Configured" : "Missing";
  $("#github-key-status").classList.toggle("on", snap.github.connected);
  $("#groq-key-status").textContent = snap.models.groq_key_configured ? "Configured" : "Missing";
  $("#groq-key-status").classList.toggle("on", snap.models.groq_key_configured);
  $("#worker-pulse").className = "pulse blue";
  const voice = snap.voice || {}; const voiceState = voice.mic_state === "speech" ? "Listening" : (voice.conv_state || "Ready");
  $("#voice-state").textContent = voiceState[0].toUpperCase() + voiceState.slice(1);
  $("#voice-orb").className = `voice-orb ${voice.conv_state || ""}`;
}

async function refresh() {
  try { state.snapshot = await api("/api/snapshot"); renderSnapshot(); await syncActiveMessages(); } catch (error) { toast(error.message); }
}

async function refreshGitHub() {
  const target = $("#github-overview"); target.replaceChildren(el("div", "muted-card", "Loading GitHub activity…"));
  try {
    const payload = await api("/api/github/overview"); target.replaceChildren();
    for (const run of (payload.workflow_runs || []).slice(0, 5)) target.append(el("div", "data-card", `${run.display_title || run.name || run.event} · ${run.status}${run.conclusion ? ` / ${run.conclusion}` : ""}`));
    for (const runner of (payload.runners || [])) target.append(el("div", "data-card", `Runner ${runner.name} · ${runner.status}${runner.busy ? " · busy" : ""}`));
    for (const repo of payload.repositories.slice(0, 5)) target.append(el("div", "data-card", `${repo.full_name} · ${repo.private ? "private" : "public"}`));
  } catch (error) { target.replaceChildren(el("div", "muted-card", error.message)); }
}

$("#command-form").onsubmit = async (event) => {
  event.preventDefault(); const input = $("#command-input"); const text = input.value.trim(); if (!text) return;
  $("#send-button").disabled = true; input.value = "";
  try { await api("/api/commands", { method: "POST", body: JSON.stringify({ session_id: state.activeSession, text }) }); await refresh(); await selectSession(state.activeSession); }
  catch (error) { toast(error.message); } finally { $("#send-button").disabled = false; input.focus(); }
};

$("#new-session").onclick = async () => { const title = prompt("Session name", "New session"); if (title === null) return; const session = await api("/api/sessions", { method: "POST", body: JSON.stringify({ title }) }); state.activeSession = session.id; await refresh(); await selectSession(session.id); };
$("#voice-mode").onclick = async () => { state.mode = "voice"; $("#voice-mode").classList.add("active"); $("#text-mode").classList.remove("active"); $("#command-input").placeholder = "Voice is active — you can also type here…"; await api("/api/input-mode", { method: "POST", body: JSON.stringify({ mode: "voice" }) }); };
$("#text-mode").onclick = async () => { state.mode = "text"; $("#text-mode").classList.add("active"); $("#voice-mode").classList.remove("active"); $("#command-input").placeholder = "Type a command…"; await api("/api/input-mode", { method: "POST", body: JSON.stringify({ mode: "text" }) }); $("#command-input").focus(); };
$("#ops-toggle").onclick = () => $("#ops-panel").classList.add("open"); $("#close-ops").onclick = () => $("#ops-panel").classList.remove("open");
$("#open-settings").onclick = () => $("#settings-dialog").showModal(); $("#refresh-github").onclick = refreshGitHub;
$("#oauth-start").onclick = async () => { try { const flow = await api("/api/github/oauth/start", { method: "POST", body: JSON.stringify({ client_id: $("#oauth-client-id").value }) }); const box = $("#device-code"); box.classList.remove("hidden"); box.textContent = `Open ${flow.verification_uri} and enter ${flow.user_code}`; window.open(flow.verification_uri, "_blank", "noopener"); clearInterval(state.oauthTimer); state.oauthTimer = setInterval(pollOAuth, Math.max(5, flow.interval) * 1000); } catch (error) { toast(error.message); } };
async function pollOAuth() { try { const result = await api("/api/github/oauth/poll", { method: "POST", body: "{}" }); if (result.status === "connected") { clearInterval(state.oauthTimer); $("#device-code").classList.add("hidden"); toast("GitHub connected"); refresh(); } } catch (error) { clearInterval(state.oauthTimer); toast(error.message); } }

refresh().then(() => selectSession(state.activeSession));
setInterval(refresh, 1500);
setInterval(() => { if (state.snapshot?.github?.connected) refreshGitHub(); }, 15000);
