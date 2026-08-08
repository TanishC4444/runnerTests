const state = { snapshot: null, activeSession: null, mode: "voice", oauthTimer: null, lastMessageCount: -1, editingSkill: null, editingMcp: null };
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

function timeAgo(seconds) {
  const diff = Math.round(Date.now() / 1000 - seconds);
  if (diff < 5) return "just now";
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  return `${Math.round(diff / 3600)}h ago`;
}

function timeUntil(seconds) {
  const diff = Math.round(seconds - Date.now() / 1000);
  if (diff <= 0) return "expired";
  if (diff < 60) return `${diff}s left`;
  return `${Math.round(diff / 60)}m left`;
}

/* ---------------------------- sessions + chat ---------------------------- */

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

/* -------------------------------- approvals ------------------------------- */

function formatArgValue(value) {
  if (Array.isArray(value)) return value.length ? value.join("; ") : "—";
  if (value && typeof value === "object") return JSON.stringify(value);
  if (value === "" || value === null || value === undefined) return "—";
  return String(value);
}

function renderApprovals() {
  const pending = state.snapshot.approvals.filter((item) => item.status === "pending");
  const countEl = $("#approval-count");
  countEl.textContent = pending.length;
  countEl.classList.toggle("hidden", pending.length === 0);
  const badge = $("#ops-badge");
  badge.textContent = pending.length;
  badge.classList.toggle("hidden", pending.length === 0);
  const list = $("#approval-list");
  list.replaceChildren();
  if (!pending.length) return list.append(el("div", "muted-card", "No actions waiting"));
  for (const approval of pending) {
    const card = el("div", "approval-card");
    const head = el("div", "approval-head");
    head.append(el("strong", "", approval.tool.replaceAll("_", " ")));
    head.append(el("span", "approval-time", timeUntil(approval.expires_at)));
    card.append(head);
    const args = el("div", "approval-args");
    for (const [key, value] of Object.entries(approval.arguments || {})) {
      const row = el("div", "approval-arg");
      row.append(el("span", "k", `${key}:`), el("span", "v", formatArgValue(value)));
      args.append(row);
    }
    card.append(args);
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

/* ---------------------------------- usage --------------------------------- */

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

/* ------------------------------- github tracker ---------------------------- */

function statusClass(run) {
  const key = (run.conclusion || run.status || "").toLowerCase();
  if (["success", "completed"].includes(key)) return "success";
  if (["failure", "cancelled", "timed_out", "action_required"].includes(key)) return "failure";
  if (["in_progress", "queued", "waiting", "pending"].includes(key)) return "in_progress";
  return "";
}

async function refreshGitHub() {
  const target = $("#github-overview"); target.replaceChildren(el("div", "muted-card", "Loading GitHub activity…"));
  try {
    const payload = await api("/api/github/overview"); target.replaceChildren();
    if (payload.repository) {
      const r = payload.repository;
      const card = el("div", "repo-card");
      card.append(el("div", "repo-name", r.full_name));
      if (r.description) card.append(el("div", "repo-desc", r.description));
      const stats = el("div", "repo-stats");
      stats.append(el("span", "", r.language || "no primary language"));
      stats.append(el("span", "", `★ ${r.stargazers_count ?? 0}`));
      stats.append(el("span", "", `⑂ ${r.forks_count ?? 0}`));
      stats.append(el("span", "", `${r.open_issues_count ?? 0} open issues`));
      stats.append(el("span", "", r.private ? "private" : "public"));
      card.append(stats);
      target.append(card);
    }
    const runs = (payload.workflow_runs || []).slice(0, 5);
    if (runs.length) {
      target.append(el("div", "github-group-label", "Recent workflow runs"));
      for (const run of runs) {
        const row = el("div", "data-card status-row");
        row.append(el("span", `status-dot ${statusClass(run)}`));
        row.append(el("span", "", `${run.display_title || run.name || run.event} · ${run.status}${run.conclusion ? ` / ${run.conclusion}` : ""}`));
        target.append(row);
      }
    }
    const runners = payload.runners || [];
    if (runners.length) {
      target.append(el("div", "github-group-label", "Self-hosted runners"));
      for (const runner of runners) {
        const row = el("div", "data-card status-row");
        row.append(el("span", `status-dot ${runner.busy ? "in_progress" : "success"}`));
        row.append(el("span", "", `${runner.name} · ${runner.status}${runner.busy ? " · busy" : ""}`));
        target.append(row);
      }
    }
    const others = (payload.repositories || []).filter((repo) => repo.full_name !== payload.repository?.full_name).slice(0, 5);
    if (others.length) {
      target.append(el("div", "github-group-label", "Other repositories"));
      for (const repo of others) target.append(el("div", "data-card", `${repo.full_name} · ${repo.private ? "private" : "public"}`));
    }
    if (!target.children.length) target.append(el("div", "muted-card", "No GitHub activity yet."));
  } catch (error) { target.replaceChildren(el("div", "muted-card", error.message)); }
}

/* --------------------------------- skills ---------------------------------- */

function renderSkills(items) {
  const list = $("#skill-list"); list.replaceChildren();
  for (const item of items) {
    const card = el("div", "data-card toggle-row");
    const copy = el("div", "row-copy");
    copy.append(el("strong", "", item.title || item.name));
    const triggers = (item.triggers || []).length ? (item.triggers || []).join(", ") : "always active";
    copy.append(el("span", "", `${item.mode || "chat"} · ${item.model || "configured"} · ${triggers}`));
    const edit = el("button", "edit-button", "Edit");
    edit.onclick = () => openSkillDialog(item);
    const toggle = el("input", "toggle"); toggle.type = "checkbox"; toggle.checked = item.enabled;
    toggle.onchange = async () => { try { await api(`/api/skills/${encodeURIComponent(item.name)}/toggle`, { method: "POST", body: JSON.stringify({ enabled: toggle.checked }) }); toast(`${item.name} ${toggle.checked ? "enabled" : "disabled"}`); } catch (error) { toggle.checked = !toggle.checked; toast(error.message); } };
    card.append(copy, edit, toggle); list.append(card);
  }
}

function openSkillDialog(skill) {
  state.editingSkill = skill?.name || null;
  $("#skill-dialog-title").textContent = skill ? "Edit skill" : "New skill";
  $("#skill-name").value = skill?.name || "";
  $("#skill-name").disabled = !!skill;
  $("#skill-title").value = skill?.title || "";
  $("#skill-mode").value = skill?.mode || "chat";
  $("#skill-model").value = skill?.model || "qwen/qwen3.6-27b";
  $("#skill-triggers").value = (skill?.triggers || []).join(", ");
  $("#skill-tools").value = (skill?.allowed_tools || []).join(", ");
  $("#skill-required").value = (skill?.required_context || []).join(", ");
  $("#skill-instructions").value = skill?.instructions || "";
  $("#skill-delete").classList.toggle("hidden", !skill);
  $("#skill-dialog").showModal();
}

function splitList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

$("#add-skill").onclick = () => openSkillDialog(null);
$("#skill-cancel").onclick = () => $("#skill-dialog").close();
$("#skill-form").onsubmit = async (event) => {
  event.preventDefault();
  const skill = {
    name: $("#skill-name").value.trim(),
    title: $("#skill-title").value.trim(),
    mode: $("#skill-mode").value,
    model: $("#skill-model").value.trim(),
    triggers: splitList($("#skill-triggers").value),
    allowed_tools: splitList($("#skill-tools").value),
    required_context: splitList($("#skill-required").value),
    instructions: $("#skill-instructions").value.trim(),
  };
  try {
    if (state.editingSkill) await api(`/api/skills/${encodeURIComponent(state.editingSkill)}/save`, { method: "POST", body: JSON.stringify(skill) });
    else await api("/api/skills", { method: "POST", body: JSON.stringify(skill) });
    $("#skill-dialog").close();
    toast("Skill saved");
    await refresh();
  } catch (error) { toast(error.message); }
};
$("#skill-delete").onclick = async () => {
  if (!state.editingSkill) return;
  try {
    await api(`/api/skills/${encodeURIComponent(state.editingSkill)}/delete`, { method: "POST", body: "{}" });
    $("#skill-dialog").close();
    toast("Skill deleted");
    await refresh();
  } catch (error) { toast(error.message); }
};

/* ---------------------------------- mcp ------------------------------------ */

function renderMcpServers(items) {
  const list = $("#mcp-list"); list.replaceChildren();
  for (const item of items) {
    const card = el("div", "data-card toggle-row");
    const copy = el("div", "row-copy");
    copy.append(el("strong", "", item.name.replaceAll("_", " ")));
    copy.append(el("span", "", `${item.url || "no url"}${item.read_only ? " · read-only" : ""}`));
    const edit = el("button", "edit-button", "Edit");
    edit.onclick = () => openMcpDialog(item);
    const toggle = el("input", "toggle"); toggle.type = "checkbox"; toggle.checked = item.enabled;
    toggle.onchange = async () => { try { await api(`/api/mcp/${encodeURIComponent(item.name)}/toggle`, { method: "POST", body: JSON.stringify({ enabled: toggle.checked }) }); toast(`${item.name} ${toggle.checked ? "enabled" : "disabled"}`); } catch (error) { toggle.checked = !toggle.checked; toast(error.message); } };
    card.append(copy, edit, toggle); list.append(card);
  }
}

function openMcpDialog(server) {
  state.editingMcp = server?.name || null;
  $("#mcp-dialog-title").textContent = server ? "Edit MCP server" : "New MCP server";
  $("#mcp-name").value = server?.name || "";
  $("#mcp-name").disabled = !!server;
  $("#mcp-url").value = server?.url || "";
  $("#mcp-auth").value = server?.auth || "";
  $("#mcp-token-env").value = server?.token_env || "";
  $("#mcp-token-env-wrap").classList.toggle("hidden", $("#mcp-auth").value !== "static_token");
  $("#mcp-readonly").checked = !!server?.read_only;
  $("#mcp-toolsets").value = (server?.toolsets || []).join(", ");
  $("#mcp-delete").classList.toggle("hidden", !server);
  $("#mcp-dialog").showModal();
}

$("#add-mcp").onclick = () => openMcpDialog(null);
$("#mcp-cancel").onclick = () => $("#mcp-dialog").close();
$("#mcp-auth").onchange = () => $("#mcp-token-env-wrap").classList.toggle("hidden", $("#mcp-auth").value !== "static_token");
$("#mcp-form").onsubmit = async (event) => {
  event.preventDefault();
  const server = {
    name: $("#mcp-name").value.trim(),
    url: $("#mcp-url").value.trim(),
    transport: "streamable_http",
    auth: $("#mcp-auth").value || null,
    token_env: $("#mcp-token-env").value.trim(),
    read_only: $("#mcp-readonly").checked,
    toolsets: splitList($("#mcp-toolsets").value),
    enabled: true,
  };
  try {
    if (state.editingMcp) await api(`/api/mcp/${encodeURIComponent(state.editingMcp)}/save`, { method: "POST", body: JSON.stringify(server) });
    else await api("/api/mcp", { method: "POST", body: JSON.stringify(server) });
    $("#mcp-dialog").close();
    toast("MCP server saved");
    await refresh();
  } catch (error) { toast(error.message); }
};
$("#mcp-delete").onclick = async () => {
  if (!state.editingMcp) return;
  try {
    await api(`/api/mcp/${encodeURIComponent(state.editingMcp)}/delete`, { method: "POST", body: "{}" });
    $("#mcp-dialog").close();
    toast("MCP server removed");
    await refresh();
  } catch (error) { toast(error.message); }
};

/* -------------------------------- ops tabs --------------------------------- */

for (const tab of document.querySelectorAll(".ops-tab")) {
  tab.onclick = () => {
    for (const other of document.querySelectorAll(".ops-tab")) other.classList.remove("active");
    tab.classList.add("active");
    const target = tab.dataset.tab;
    for (const view of document.querySelectorAll(".ops-view")) view.classList.toggle("hidden", view.dataset.view !== target);
    if (target === "github") refreshGitHub();
  };
}

/* -------------------------------- snapshot ---------------------------------- */

function renderSnapshot() {
  const snap = state.snapshot;
  state.activeSession ||= snap.active_session_id;
  renderSessions(); renderApprovals(); renderUsage();
  renderMcpServers(snap.mcp_servers);
  renderSkills(snap.skills);
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
$("#open-settings").onclick = () => $("#settings-dialog").showModal();
$("#refresh-github").onclick = refreshGitHub;
$("#oauth-start").onclick = async () => { try { const flow = await api("/api/github/oauth/start", { method: "POST", body: JSON.stringify({ client_id: $("#oauth-client-id").value }) }); const box = $("#device-code"); box.classList.remove("hidden"); box.textContent = `Open ${flow.verification_uri} and enter ${flow.user_code}`; window.open(flow.verification_uri, "_blank", "noopener"); clearInterval(state.oauthTimer); state.oauthTimer = setInterval(pollOAuth, Math.max(5, flow.interval) * 1000); } catch (error) { toast(error.message); } };
async function pollOAuth() { try { const result = await api("/api/github/oauth/poll", { method: "POST", body: "{}" }); if (result.status === "connected") { clearInterval(state.oauthTimer); $("#device-code").classList.add("hidden"); toast("GitHub connected"); refresh(); } } catch (error) { clearInterval(state.oauthTimer); toast(error.message); } }

refresh().then(() => selectSession(state.activeSession));
setInterval(refresh, 1500);
setInterval(() => { if (state.snapshot?.github?.connected && !$("[data-view='github']").classList.contains("hidden")) refreshGitHub(); }, 15000);
