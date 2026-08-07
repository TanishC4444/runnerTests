# Runner Control setup

## Start locally

```bash
python3 -m pip install -r requirements-local.txt
python3 run_all.py
```

Open `http://127.0.0.1:8765`. No local AI model or Ollama installation is
required. The local machine runs the microphone, dashboard, approval boundary,
and speech playback only.

`run_all.py` automatically loads `.env`. Keep `GROQ_API_KEY` and `GH_TOKEN`
there. The repository Actions secret `RUNNER_PAT` is still required by the
emergency Qwen 2.5 voice watcher.

## How commands route

- Qwen 3.6 27B remains the live conversational coordinator.
- Qwen token ceilings adapt by intent: short follow-ups stay small, reads get a
  moderate ceiling, and structured engineering plans receive more room.
- Voice and typed commands both pass through the same session-aware router.
- Read-only GitHub and enabled read-only MCP tools may run immediately.
- Simple GitHub writes become one-time dashboard approvals.
- Engineering tasks trigger clarification when material context is missing.
- Once repository, objective, constraints, and success criteria are clear,
  Qwen creates ordered steps and queues a GPT-OSS delegation for approval.
- After approval, GPT-OSS 20B runs asynchronously on `ubuntu-latest`; Qwen and
  the voice assistant remain available during the job.

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
written into sessions, or persisted after restart. The microphone fallback
still needs `GH_TOKEN` because a child process cannot inherit an OAuth token
obtained after that process started.

## Skills infrastructure

Skills are JSON packages in `skills/`. Each package defines:

- name and title;
- enabled state;
- trigger phrases;
- mode (`read`, `delegate`, or `chat`);
- model;
- allowed tools;
- required context; and
- instructions injected into Qwen.

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
