# Runner Control setup

## Start locally

```bash
python3 -m pip install -r requirements-local.txt
python3 run_all.py
```

Open `http://127.0.0.1:8765`. No local AI model or Ollama installation is
required on your machine -- GPT-OSS 20B runs entirely on a GitHub Actions
runner, reached over `ollama_relay.py`. The local machine runs the microphone,
dashboard, approval boundary, and speech playback only.

`run_all.py` automatically loads `.env`. Keep `GH_TOKEN` there -- it's the only
credential needed now (there is no more Groq/hosted API key). The repository
Actions secret `RUNNER_PAT` is required for both `gptoss_watcher.yml` (the
live coordinator) and `agent_worker.yml` (background engineering jobs).

Every session, `run_all.py` triggers `gptoss_watcher.yml` and waits for it to
report ready before starting the microphone. Cold start (installing Ollama,
pulling the ~14GB model if the cache missed, loading it) is typically
1-3+ minutes, and CPU-only generation on the runner is meaningfully slower
per turn than a hosted API would be -- this is the tradeoff of running the
model on Actions infrastructure instead of a local GPU/API, and it's expected
behavior, not a bug.

## How commands route

- GPT-OSS 20B, running on the `gptoss_watcher.yml` Actions runner, is the only
  conversational coordinator now -- both tool-calling routing decisions and
  plain chat replies go through it, relayed via `chat/Log 1/completions.json`
  (control_plane.py/ollama_relay.py write requests, watch_and_respond.py
  answers them).
- Token ceilings adapt by intent: short follow-ups stay small, reads get a
  moderate ceiling, and structured engineering plans receive more room.
- Voice and typed commands both pass through the same session-aware router.
- A safe read tool can be chained automatically inside one router turn (up to
  `ModelRouter.MAX_TOOL_STEPS`); every other tool call stops the loop and
  returns to the caller for approval.
- Read-only GitHub and enabled read-only MCP tools may run immediately.
- Every other GitHub write, MCP write, and agent dispatch becomes a one-time
  dashboard approval -- nothing executes until you approve it.
- Once repository, objective, constraints, and success criteria are clear,
  GPT-OSS creates ordered steps and queues a `delegate_to_gptoss` dispatch
  (optionally fanned out across `subtasks`, each its own parallel runner) for
  approval.
- After approval, GPT-OSS 20B runs the actual engineering work asynchronously
  on `agent_worker.yml`; the live coordinator and voice assistant remain
  available while it runs, and the dashboard/voice announce completion when
  it's done.

## GitHub-hosted GPT-OSS worker

The `GPT-OSS Agent Worker` workflow:

1. restores `~/.ollama/models` with `actions/cache`;
2. installs/starts Ollama on the GitHub-hosted runner;
3. pulls `gpt-oss:20b` when the cache is missing;
4. runs a bounded file/read/write/check loop inside the checked-out repository;
5. blocks `.git`, `.env`, `agent_output`, and workflow-file edits;
6. pushes changes to `agent/gpt-oss-<run id>`; and
7. opens a pull request for human review.

The model cache is best-effort: GitHub cache quota or artifact-size policies
may prevent saving a model this large, in which case a later job pulls it again.
No model is hosted on the user's computer.

The workflow must exist on the branch used for dispatch. Merge these files into
`main` before delegating tasks against `main`.

In repository **Settings → Actions → General**, enable **Allow GitHub Actions to
create and approve pull requests**. Without it, the worker can upload its result
artifact but the final pull-request step will be rejected.

## GitHub authentication

`GH_TOKEN` works for direct tools and workflow dispatch. The dashboard also
supports GitHub OAuth Device Flow:

1. register a GitHub OAuth App and enable Device Flow;
2. open **Connections & routing**;
3. enter the OAuth App client ID; and
4. complete the displayed device code.

OAuth tokens remain in process memory. They are not returned to browser code,
written into sessions, or persisted after restart. `GH_TOKEN` is still
required regardless, because the microphone process and the GPT-OSS relay
(both separate from the dashboard's own OAuth flow) cannot inherit an OAuth
token obtained after they started.

## Skills infrastructure

Skills are JSON packages in `skills/`. Each package defines:

- name and title;
- enabled state;
- trigger phrases;
- mode (`read`, `delegate`, or `chat`);
- model;
- allowed tools;
- required context; and
- instructions injected into the coordinator.

Add another `*.json` file to create a future skill. The registry discovers it
at runtime, the dashboard lists it, and its enabled state can be toggled there.
Current skills are:

- `github_reader` — evidence-based read-only GitHub/MCP answers;
- `github_engineering` — clarification, planning, and GPT-OSS delegation;
- `conversation` — ordinary chat without invented actions.
- `repository_bootstrap` — resolves visibility, README, license, and relevant
  gitignore choices before proposing repository creation.

## MCP infrastructure

MCP servers live in `config/control_plane.json`. The included GitHub remote MCP
uses its read-only endpoint so voice commands can safely call its tools. It is
disabled initially; authenticate GitHub and enable it in the dashboard.

Only Streamable HTTP MCP is implemented. Every non-read-only MCP call is
approval-gated, and tools with destructive names remain blocked.

## Safeguards

- dashboard binds to localhost only;
- GitHub owner/repository allow-lists;
- repository writes and agent dispatch require one-time approval;
- 15-minute approval expiry;
- no direct delete/force/secret/transfer tools;
- file/path/size restrictions;
- background worker has a 24-step cap and command allow-list;
- background changes go to a new branch and pull request, never directly to
  the base branch; and
- per-session logs exclude credentials and are ignored by Git.
