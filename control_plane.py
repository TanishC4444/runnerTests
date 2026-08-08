"""Local control plane for sessions, model routing, approvals, GitHub, and MCP.

Secrets are accepted from the process environment or held in memory after an
OAuth device flow. They are never returned by dashboard APIs or written to the
session log. All repository writes are approval-gated by default; destructive
GitHub operations are intentionally not exposed as tools.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "control_plane.json"
DATA_DIR = ROOT / "control_data"
SESSIONS_DIR = DATA_DIR / "sessions"
GITHUB_API = "https://api.github.com"


class ControlPlaneError(RuntimeError):
    pass


def _now() -> float:
    return time.time()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    return cleaned[:80] or uuid.uuid4().hex


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _coerce_bool(value: Any, field: str) -> bool:
    """Some Groq-hosted tool-calling models (e.g. qwen3.6-27b via Groq's
    XML-style function-call parsing) emit Python-style boolean literals
    ("True"/"False") as strings instead of JSON booleans. The tool schema
    accepts either type so Groq's own strict validation doesn't reject the
    call outright; this normalizes whatever comes through into a real bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no", ""):
            return False
    raise ControlPlaneError(f"Could not interpret {field!r} value {value!r} as a boolean")


class SessionStore:
    def __init__(self, root: Path | None = None):
        self.root = root or SESSIONS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.lock = threading.RLock()
        self._index = _load_json(self.index_path, {"sessions": []})

    def _save_index(self):
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    def create(self, title: str = "New session", source: str = "mixed") -> dict:
        with self.lock:
            session = {
                "id": f"s-{int(_now())}-{uuid.uuid4().hex[:8]}",
                "title": title.strip()[:100] or "New session",
                "source": source,
                "created_at": _now(),
                "updated_at": _now(),
            }
            self._index["sessions"].insert(0, session)
            self._save_index()
            return dict(session)

    def list(self) -> list[dict]:
        with self.lock:
            return [dict(item) for item in self._index["sessions"]]

    def get(self, session_id: str) -> dict | None:
        return next((s for s in self.list() if s["id"] == session_id), None)

    def append(self, session_id: str, event: dict):
        if not self.get(session_id):
            raise ControlPlaneError("Unknown session")
        record = {"timestamp": _now(), **event}
        path = self.root / f"{_safe_id(session_id)}.jsonl"
        with self.lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            for item in self._index["sessions"]:
                if item["id"] == session_id:
                    item["updated_at"] = record["timestamp"]
                    break
            self._index["sessions"].sort(key=lambda s: s["updated_at"], reverse=True)
            self._save_index()

    def messages(self, session_id: str, limit: int = 250) -> list[dict]:
        if not self.get(session_id):
            raise ControlPlaneError("Unknown session")
        path = self.root / f"{_safe_id(session_id)}.jsonl"
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records[-max(1, min(limit, 1000)):]


class UsageMeter:
    def __init__(self):
        self.lock = threading.Lock()
        self.by_model: dict[str, dict[str, int]] = {}

    def add(self, model: str, usage: dict | None):
        usage = usage or {}
        with self.lock:
            row = self.by_model.setdefault(model, {"requests": 0, "input_tokens": 0, "output_tokens": 0})
            row["requests"] += 1
            row["input_tokens"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            row["output_tokens"] += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.by_model))


class GitHubAuth:
    def __init__(self):
        self._oauth_token: str | None = None
        self._device: dict | None = None
        self.lock = threading.Lock()

    def token(self) -> str | None:
        return self._oauth_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    def source(self) -> str:
        if self._oauth_token:
            return "oauth-memory"
        if os.environ.get("GITHUB_TOKEN"):
            return "environment"
        if os.environ.get("GH_TOKEN"):
            return "environment"
        return "none"

    def status(self) -> dict:
        return {"connected": bool(self.token()), "source": self.source(), "token_exposed": False}

    def start_device_flow(self, client_id: str, scope: str = "repo workflow read:org") -> dict:
        if not client_id.strip():
            raise ControlPlaneError("A GitHub OAuth App client ID is required")
        response = requests.post(
            "https://github.com/login/device/code",
            headers={"Accept": "application/json"},
            data={"client_id": client_id.strip(), "scope": scope},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        with self.lock:
            self._device = {
                "client_id": client_id.strip(),
                "device_code": payload["device_code"],
                "interval": int(payload.get("interval", 5)),
                "expires_at": _now() + int(payload.get("expires_in", 900)),
                "last_poll": 0.0,
            }
        return {
            "user_code": payload["user_code"],
            "verification_uri": payload["verification_uri"],
            "expires_in": payload.get("expires_in", 900),
            "interval": payload.get("interval", 5),
        }

    def poll_device_flow(self) -> dict:
        with self.lock:
            device = dict(self._device) if self._device else None
        if not device:
            raise ControlPlaneError("No GitHub device authorization is pending")
        if _now() >= device["expires_at"]:
            raise ControlPlaneError("The GitHub device code expired")
        wait_for = device["interval"] - (_now() - device["last_poll"])
        if wait_for > 0:
            return {"status": "pending", "retry_after": round(wait_for, 1)}
        with self.lock:
            if self._device:
                self._device["last_poll"] = _now()
        response = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": device["client_id"],
                "device_code": device["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("access_token"):
            with self.lock:
                self._oauth_token = payload["access_token"]
                self._device = None
            return {"status": "connected", "scope": payload.get("scope", "")}
        error = payload.get("error", "authorization_pending")
        if error == "slow_down":
            with self.lock:
                if self._device:
                    self._device["interval"] += 5
        if error in ("authorization_pending", "slow_down"):
            return {"status": "pending", "error": error}
        raise ControlPlaneError(payload.get("error_description", error))

    def disconnect(self):
        with self.lock:
            self._oauth_token = None
            self._device = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    write: bool = False


class GitHubTools:
    def __init__(self, config: dict, auth: GitHubAuth):
        self.config = config
        self.auth = auth

    @property
    def specs(self) -> list[ToolSpec]:
        repo_target = {
            "owner": {"type": "string"},
            "repo": {"type": "string"},
        }
        return [
            ToolSpec("github_list_repositories", "List repositories available to the authenticated user.", {"type": "object", "properties": {}, "additionalProperties": False}),
            ToolSpec("github_get_repository", "Get basic info about one repository: description, language, stars, forks, open issues, default branch, license, and topics.", {"type": "object", "properties": repo_target, "required": ["owner", "repo"], "additionalProperties": False}),
            ToolSpec("github_list_repo_files", "List the files and folders at a path inside a repository (non-recursive, one directory level).", {"type": "object", "properties": {**repo_target, "path": {"type": "string", "description": "Directory path; empty string for the repository root"}, "ref": {"type": "string", "description": "Branch, tag, or commit SHA; defaults to the default branch"}}, "required": ["owner", "repo"], "additionalProperties": False}),
            ToolSpec("github_get_file", "Read the text contents of one file in a repository.", {"type": "object", "properties": {**repo_target, "path": {"type": "string"}, "ref": {"type": "string", "description": "Branch, tag, or commit SHA; defaults to the default branch"}}, "required": ["owner", "repo", "path"], "additionalProperties": False}),
            ToolSpec("github_list_workflow_runs", "List recent GitHub Actions workflow runs for a repository.", {"type": "object", "properties": repo_target, "required": ["owner", "repo"], "additionalProperties": False}),
            ToolSpec("github_list_runners", "List self-hosted runners configured for a repository.", {"type": "object", "properties": repo_target, "required": ["owner", "repo"], "additionalProperties": False}),
            ToolSpec("github_dispatch_agent", "Dispatch an approved GPT-OSS 20B task to a GitHub-hosted Actions runner. It works in the background and may open a reviewable pull request.", {"type": "object", "properties": {**repo_target, "task": {"type": "string"}, "session_id": {"type": "string"}, "ref": {"type": "string"}}, "required": ["owner", "repo", "task"], "additionalProperties": False}, True),
            ToolSpec("github_create_repository", "Create a new repository after the user has explicitly chosen visibility and whether to initialize a README. A license and gitignore template may also be selected.", {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "private": {"type": ["boolean", "string"], "description": "true or false"}, "auto_init": {"type": ["boolean", "string"], "description": "Create an initial README commit; true or false"}, "license_template": {"type": "string", "description": "GitHub license keyword such as mit, apache-2.0, gpl-3.0, or empty for none"}, "gitignore_template": {"type": "string", "description": "GitHub gitignore template such as Python or Node, or empty for none"}}, "required": ["name", "private", "auto_init"], "additionalProperties": False}, True),
            ToolSpec("github_create_folder", "Create a folder in a repository by adding a .gitkeep file.", {"type": "object", "properties": {**repo_target, "path": {"type": "string"}, "branch": {"type": "string"}, "message": {"type": "string"}}, "required": ["owner", "repo", "path"], "additionalProperties": False}, True),
            ToolSpec("github_put_file", "Create or update one UTF-8 text file in a repository.", {"type": "object", "properties": {**repo_target, "path": {"type": "string"}, "content": {"type": "string"}, "branch": {"type": "string"}, "message": {"type": "string"}}, "required": ["owner", "repo", "path", "content"], "additionalProperties": False}, True),
        ]

    def openai_tools(self) -> list[dict]:
        return [{"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.parameters}} for s in self.specs]

    def spec(self, name: str) -> ToolSpec:
        match = next((item for item in self.specs if item.name == name), None)
        if not match:
            raise ControlPlaneError("Unknown or prohibited GitHub tool")
        return match

    def _headers(self) -> dict:
        token = self.auth.token()
        if not token:
            raise ControlPlaneError("GitHub is not connected")
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        }

    def _request(self, method: str, path: str, **kwargs) -> Any:
        response = requests.request(method, GITHUB_API + path, headers=self._headers(), timeout=30, **kwargs)
        if not response.ok:
            detail = response.json().get("message", response.text[:500]) if response.text else "empty response"
            raise ControlPlaneError(f"GitHub returned HTTP {response.status_code}: {detail}")
        return response.json() if response.content else {}

    def _validate_repo(self, owner: str, repo: str):
        if owner not in self.config["allowed_owners"]:
            raise ControlPlaneError(f"Owner {owner!r} is outside the configured allow-list")
        allowed = self.config.get("allowed_repositories", ["*"])
        if "*" not in allowed and repo not in allowed:
            raise ControlPlaneError(f"Repository {repo!r} is outside the configured allow-list")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
            raise ControlPlaneError("Invalid repository name")

    def _validate_path(self, path: str):
        normalized = path.strip("/")
        if not normalized or ".." in normalized.split("/") or normalized.startswith(".git/"):
            raise ControlPlaneError("Unsafe repository path")
        if normalized.startswith(".github/workflows/") and not self.config.get("allow_workflow_file_edits"):
            raise ControlPlaneError("Workflow-file edits are disabled by policy")
        return normalized

    def execute(self, name: str, arguments: dict) -> Any:
        self.spec(name)
        if name == "github_list_repositories":
            rows = self._request("GET", "/user/repos?per_page=50&sort=updated")
            return [{"full_name": r["full_name"], "private": r["private"], "updated_at": r["updated_at"], "default_branch": r["default_branch"]} for r in rows]
        if name == "github_get_repository":
            owner, repo = arguments["owner"], arguments["repo"]
            self._validate_repo(owner, repo)
            r = self._request("GET", f"/repos/{quote(owner)}/{quote(repo)}")
            return {
                "full_name": r.get("full_name"),
                "description": r.get("description"),
                "private": r.get("private"),
                "language": r.get("language"),
                "default_branch": r.get("default_branch"),
                "stargazers_count": r.get("stargazers_count"),
                "forks_count": r.get("forks_count"),
                "open_issues_count": r.get("open_issues_count"),
                "license": (r.get("license") or {}).get("spdx_id"),
                "topics": r.get("topics", []),
                "html_url": r.get("html_url"),
                "updated_at": r.get("updated_at"),
            }
        if name == "github_list_repo_files":
            owner, repo = arguments["owner"], arguments["repo"]
            self._validate_repo(owner, repo)
            path = (arguments.get("path") or "").strip("/")
            query = f"?ref={quote(arguments['ref'])}" if arguments.get("ref") else ""
            result = self._request("GET", f"/repos/{quote(owner)}/{quote(repo)}/contents/{quote(path, safe='/')}{query}")
            rows = result if isinstance(result, list) else [result]
            return [{"name": row.get("name"), "path": row.get("path"), "type": row.get("type"), "size": row.get("size")} for row in rows]
        if name == "github_get_file":
            owner, repo = arguments["owner"], arguments["repo"]
            self._validate_repo(owner, repo)
            path = self._validate_path(arguments["path"])
            query = f"?ref={quote(arguments['ref'])}" if arguments.get("ref") else ""
            result = self._request("GET", f"/repos/{quote(owner)}/{quote(repo)}/contents/{quote(path, safe='/')}{query}")
            if isinstance(result, list) or result.get("type") != "file":
                raise ControlPlaneError(f"{path!r} is a directory, not a file")
            content = base64.b64decode(result.get("content", "")).decode("utf-8", errors="replace")
            truncated = len(content) > 20000
            return {"path": path, "size": result.get("size"), "truncated": truncated, "content": content[:20000]}
        if name in ("github_list_workflow_runs", "github_list_runners"):
            owner, repo = arguments["owner"], arguments["repo"]
            self._validate_repo(owner, repo)
            suffix = "actions/runs?per_page=25" if name.endswith("workflow_runs") else "actions/runners?per_page=50"
            result = self._request("GET", f"/repos/{quote(owner)}/{quote(repo)}/{suffix}")
            key = "workflow_runs" if name.endswith("workflow_runs") else "runners"
            keep = ("id", "name", "display_title", "status", "conclusion", "event", "html_url", "created_at", "updated_at", "busy", "os")
            return [{field: row.get(field) for field in keep if field in row} for row in result.get(key, [])]
        if name == "github_create_repository":
            if not self.config.get("allow_create_repository"):
                raise ControlPlaneError("Repository creation is disabled by policy")
            repo_name = arguments["name"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo_name):
                raise ControlPlaneError("Invalid repository name")
            authenticated_user = self._request("GET", "/user").get("login")
            if authenticated_user not in self.config["allowed_owners"]:
                raise ControlPlaneError("The authenticated GitHub user is outside the configured owner allow-list")
            is_private = _coerce_bool(arguments["private"], "private")
            auto_init = _coerce_bool(arguments["auto_init"], "auto_init")
            if (arguments.get("license_template") or arguments.get("gitignore_template")) and not auto_init:
                raise ControlPlaneError("A license or gitignore template requires repository initialization")
            body = {"name": repo_name, "description": arguments.get("description", ""), "private": is_private, "auto_init": auto_init}
            if arguments.get("license_template"):
                body["license_template"] = arguments["license_template"]
            if arguments.get("gitignore_template"):
                body["gitignore_template"] = arguments["gitignore_template"]
            result = self._request("POST", "/user/repos", json=body)
            created_owner = result.get("owner", {}).get("login")
            if created_owner not in self.config["allowed_owners"]:
                raise ControlPlaneError("GitHub created the repository outside the configured owner allow-list")
            return {"full_name": result["full_name"], "html_url": result["html_url"], "private": result["private"]}
        if name == "github_dispatch_agent":
            owner, repo = arguments["owner"], arguments["repo"]
            self._validate_repo(owner, repo)
            task = arguments["task"].strip()
            if not task or len(task) > 12000:
                raise ControlPlaneError("Agent task must contain 1 to 12,000 characters")
            self._request(
                "POST",
                f"/repos/{quote(owner)}/{quote(repo)}/actions/workflows/agent_worker.yml/dispatches",
                json={"ref": arguments.get("ref") or "main", "inputs": {"task": task, "session_id": arguments.get("session_id") or "dashboard"}},
            )
            return {"dispatched": True, "workflow": "agent_worker.yml", "model": "gpt-oss:20b"}
        owner, repo = arguments["owner"], arguments["repo"]
        self._validate_repo(owner, repo)
        if name == "github_create_folder":
            path = self._validate_path(arguments["path"]) + "/.gitkeep"
            content = ""
        else:
            path = self._validate_path(arguments["path"])
            content = arguments["content"]
        encoded = content.encode("utf-8")
        if len(encoded) > int(self.config.get("max_file_bytes", 262144)):
            raise ControlPlaneError("File exceeds the configured size limit")
        branch = arguments.get("branch") or "main"
        api_path = f"/repos/{quote(owner)}/{quote(repo)}/contents/{quote(path, safe='/')}"
        existing_sha = None
        if name == "github_put_file":
            response = requests.get(GITHUB_API + api_path, headers=self._headers(), params={"ref": branch}, timeout=30)
            if response.ok:
                existing_sha = response.json().get("sha")
            elif response.status_code != 404:
                raise ControlPlaneError(f"GitHub returned HTTP {response.status_code}: {response.text[:500]}")
        body = {"message": arguments.get("message") or f"Create {path}", "content": base64.b64encode(encoded).decode("ascii"), "branch": branch}
        if existing_sha:
            body["sha"] = existing_sha
        result = self._request("PUT", api_path, json=body)
        return {"path": path, "commit": result.get("commit", {}).get("html_url"), "sha": result.get("content", {}).get("sha")}


class MCPRegistry:
    """Configurable Streamable HTTP MCP registry.

    MCP tools are discoverable for monitoring now. Calls remain subject to the
    same approval boundary in ControlPlane; an enabled remote server receives
    the in-memory GitHub OAuth token only in its Authorization header.
    """

    def __init__(self, config: dict, auth: GitHubAuth):
        self.config = config
        self.auth = auth
        self._tool_cache: dict[str, dict] = {}

    def snapshot(self) -> list[dict]:
        return [{"name": name, "enabled": bool(cfg.get("enabled")), "transport": cfg.get("transport"), "url": cfg.get("url"), "read_only": bool(cfg.get("read_only")), "auth": cfg.get("auth"), "toolsets": cfg.get("toolsets", [])} for name, cfg in self.config.items()]

    @staticmethod
    def _decode_response(response: requests.Response) -> dict:
        if not response.content:
            return {}
        if "text/event-stream" in response.headers.get("Content-Type", ""):
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:].strip())
            return {}
        return response.json()

    def _session(self, name: str) -> tuple[dict, requests.Session, dict]:
        cfg = self.config.get(name)
        if not cfg or not cfg.get("enabled"):
            raise ControlPlaneError(f"MCP server {name!r} is not enabled")
        if cfg.get("transport") != "streamable_http":
            raise ControlPlaneError("Only Streamable HTTP MCP servers are supported by this control plane")
        auth_mode = cfg.get("auth")
        token = None
        if auth_mode == "github_oauth":
            token = self.auth.token()
        elif auth_mode == "static_token":
            # A bearer token read from an environment variable at request time --
            # lets you add API-key-based MCP servers (Notion, Linear, a private
            # server, etc.) from the dashboard without ever putting the secret
            # itself in config/control_plane.json. Set the env var in .env.
            env_var = cfg.get("token_env")
            token = os.environ.get(env_var) if env_var else None
        if auth_mode and not token:
            raise ControlPlaneError(f"MCP server {name!r} requires an authenticated connection")
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": "2025-03-26",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        session = requests.Session()
        init = session.post(
            cfg["url"],
            headers=headers,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "runner-control", "version": "1.0"}}},
            timeout=30,
        )
        if not init.ok:
            raise ControlPlaneError(f"MCP initialize failed with HTTP {init.status_code}: {init.text[:500]}")
        session_id = init.headers.get("Mcp-Session-Id")
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        session.post(cfg["url"], headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout=15)
        return cfg, session, headers

    def list_tools(self, name: str, refresh: bool = False) -> list[dict]:
        if name in self._tool_cache and not refresh:
            return list(self._tool_cache[name].values())
        cfg, session, headers = self._session(name)
        response = session.post(cfg["url"], headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, timeout=30)
        if not response.ok:
            raise ControlPlaneError(f"MCP tools/list failed with HTTP {response.status_code}: {response.text[:500]}")
        payload = self._decode_response(response)
        if payload.get("error"):
            raise ControlPlaneError(f"MCP tools/list failed: {payload['error']}")
        tools = payload.get("result", {}).get("tools", [])
        self._tool_cache[name] = {tool["name"]: tool for tool in tools}
        return tools

    def openai_tools(self) -> list[dict]:
        result = []
        for server, cfg in self.config.items():
            if not cfg.get("enabled"):
                continue
            try:
                tools = self.list_tools(server)
            except Exception:
                continue
            for tool in tools:
                result.append({"type": "function", "function": {"name": f"mcp__{server}__{tool['name']}", "description": tool.get("description", f"MCP tool {tool['name']}"), "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})}})
        return result

    def execute(self, qualified_name: str, arguments: dict) -> Any:
        parts = qualified_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            raise ControlPlaneError("Invalid MCP tool name")
        _, server, tool_name = parts
        denied = ("delete", "remove", "destroy", "force", "secret", "token", "transfer", "archive")
        if any(word in tool_name.lower() for word in denied):
            raise ControlPlaneError("This MCP tool is blocked by the destructive-action policy")
        known = {tool["name"] for tool in self.list_tools(server)}
        if tool_name not in known:
            raise ControlPlaneError("The MCP tool is not in the server's advertised tool list")
        cfg, session, headers = self._session(server)
        response = session.post(cfg["url"], headers=headers, json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}, timeout=120)
        if not response.ok:
            raise ControlPlaneError(f"MCP tools/call failed with HTTP {response.status_code}: {response.text[:500]}")
        payload = self._decode_response(response)
        if payload.get("error"):
            raise ControlPlaneError(f"MCP tool failed: {payload['error']}")
        return payload.get("result", {})

    def is_read_only(self, qualified_name: str) -> bool:
        parts = qualified_name.split("__", 2)
        if len(parts) != 3:
            return False
        _, server, tool_name = parts
        cfg = self.config.get(server, {})
        if cfg.get("read_only"):
            return True
        tool = next((item for item in self.list_tools(server) if item.get("name") == tool_name), {})
        return bool(tool.get("annotations", {}).get("readOnlyHint"))


class SkillRegistry:
    """File-backed skill packages used as Qwen routing instructions."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict]:
        skills = []
        for path in sorted(self.directory.glob("*.json")):
            skill = _load_json(path, None)
            if isinstance(skill, dict) and skill.get("name"):
                skill["file"] = path.name
                skills.append(skill)
        return skills

    def select(self, text: str) -> list[dict]:
        lowered = text.lower()
        selected = []
        for skill in self.list():
            if not skill.get("enabled", True):
                continue
            triggers = skill.get("triggers", [])
            if not triggers or any(trigger.lower() in lowered for trigger in triggers):
                selected.append(skill)
        return selected

    def set_enabled(self, name: str, enabled: bool) -> dict:
        skill = next((item for item in self.list() if item["name"] == name), None)
        if not skill:
            raise ControlPlaneError("Unknown skill")
        path = self.directory / skill.pop("file")
        skill["enabled"] = bool(enabled)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(skill, indent=2), encoding="utf-8")
        tmp.replace(path)
        return skill

    def save(self, skill: dict, original_name: str | None = None) -> dict:
        """Create a new skill, or overwrite one in place when `original_name`
        matches an existing file. Renaming (original_name != new name)
        removes the old file so there's never a stale duplicate on disk."""
        name = str(skill.get("name", "")).strip().lower().replace(" ", "_")
        if not re.fullmatch(r"[a-z0-9_]+", name or ""):
            raise ControlPlaneError("Skill name must use lowercase letters, numbers, and underscores only")
        if not str(skill.get("title", "")).strip():
            raise ControlPlaneError("Skill needs a title")
        cleaned = {
            "name": name,
            "title": skill["title"].strip(),
            "enabled": bool(skill.get("enabled", True)),
            "triggers": [str(t).strip() for t in skill.get("triggers", []) if str(t).strip()],
            "mode": skill.get("mode") or "chat",
            "model": skill.get("model") or "qwen/qwen3.6-27b",
            "allowed_tools": [str(t).strip() for t in skill.get("allowed_tools", []) if str(t).strip()],
            "instructions": str(skill.get("instructions", "")).strip(),
        }
        if skill.get("required_context"):
            cleaned["required_context"] = [str(t).strip() for t in skill["required_context"] if str(t).strip()]
        path = self.directory / f"{name}.json"
        if original_name and original_name != name:
            existing = next((item for item in self.list() if item["name"] == original_name), None)
            if existing:
                (self.directory / existing["file"]).unlink(missing_ok=True)
        elif not original_name and path.exists():
            raise ControlPlaneError(f"A skill named {name!r} already exists")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
        tmp.replace(path)
        return {**cleaned, "file": path.name}

    def delete(self, name: str) -> None:
        skill = next((item for item in self.list() if item["name"] == name), None)
        if not skill:
            raise ControlPlaneError("Unknown skill")
        (self.directory / skill["file"]).unlink()


class ModelRouter:
    def __init__(self, config: dict, skills: SkillRegistry, github: GitHubTools, mcp: MCPRegistry, usage: UsageMeter):
        self.config = config
        self.skills = skills
        self.github = github
        self.mcp = mcp
        self.usage = usage

    @staticmethod
    def _delegation_tool() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "delegate_to_gptoss",
                "description": (
                    "Submit a complete, scoped engineering plan to the background GPT-OSS 20B GitHub Actions worker. "
                    "Call only after material ambiguities are resolved. If the work has genuinely independent pieces "
                    "(e.g. separate modules/files that don't depend on each other's output), list each as its own "
                    "entry in `subtasks` instead of one long sequential plan -- each one gets dispatched to its own "
                    "parallel GitHub Actions runner at approval time instead of running one after another."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "base_branch": {"type": "string"},
                        "objective": {"type": "string", "description": "Overall goal. If subtasks are used, this is the umbrella description."},
                        "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Ordered steps for a single sequential run. Omit/ignore in favor of `subtasks` when the work parallelizes."},
                        "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                        "subtasks": {
                            "type": "array",
                            "description": "Optional. Independent chunks of work, each dispatched to its own parallel runner instead of one sequential worker. Only use when the pieces truly don't depend on each other's output.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "objective": {"type": "string"},
                                    "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                    "constraints": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["objective", "steps", "acceptance_criteria"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["owner", "repo", "base_branch", "objective", "steps", "acceptance_criteria"],
                    "additionalProperties": False,
                },
            },
        }

    def _messages(self, text: str, history: list[dict], skills: list[dict]) -> list[dict]:
        skill_text = "\n".join(f"- {skill['title']}: {skill.get('instructions', '')}" for skill in skills)
        # Brevity comes first and is unconditional -- every extra word here is
        # extra prompt tokens on every single turn, and short replies cost
        # fewer completion tokens too. Both count against Groq's per-minute
        # token limit for this model, so this line is the main lever against
        # rate-limit errors, not a style preference.
        system = (
            "Reply in as few words as possible -- one or two short sentences unless the user "
            "explicitly asks for more. No filler, no restating the question, no hedging.\n\n"
            "You are the always-available Qwen coordinator and conversational assistant. "
            "Use read tools immediately when they can answer or discover missing facts. "
            "For engineering work delegated to GPT-OSS, ask only high-impact questions whose answers "
            "materially affect implementation; ask no more than three concise questions at once. "
            "Do not ask the user for facts a repository read can discover. Do not delegate until the "
            "repository, objective, constraints that matter, and measurable success criteria are clear. "
            "When clear, call delegate_to_gptoss with an ordered plan. Never claim work completed before "
            "a tool result says so. Use direct write tools only for simple administrative actions or when "
            "the user supplied the exact content. Delegate any task requiring repository inspection, code "
            "generation, multiple edits, debugging, or tests. For new repositories, never choose visibility, "
            "README initialization, license, or gitignore on the user's behalf; ask for missing choices, combining "
            "them into at most three concise questions.\n\nActive skills:\n"
            + (skill_text or "- Conversation only")
        )
        messages = [{"role": "system", "content": system}]
        # Trimmed from 16 messages / 6000 chars each: that ceiling alone could
        # add ~24k prompt tokens before the tool schemas or this turn's text
        # are even counted -- easily enough on its own to blow an 8k/minute cap.
        for item in history[-8:]:
            if item.get("kind") == "message" and item.get("role") in ("user", "assistant"):
                messages.append({"role": item["role"], "content": str(item.get("content", ""))[:1500]})
        if not messages or messages[-1].get("content") != text:
            messages.append({"role": "user", "content": text})
        return messages

    @staticmethod
    def token_budget(text: str, history: list[dict]) -> int:
        """Give short turns small ceilings and reserve space for real plans.
        Lowered across the board -- these are hard caps on completion tokens,
        the other lever (besides prompt size) against the per-minute limit."""
        lowered = text.lower()
        planning = any(word in lowered for word in ("implement", "build", "refactor", "debug", "fix", "plan", "script"))
        reading = any(word in lowered for word in ("list", "show", "read", "status", "latest", "what", "which"))
        followup = len(text.split()) <= 20 and len(history) >= 2
        if planning:
            return 500
        if reading:
            return 220
        if followup:
            return 120
        return 160

    def _tools_for_skills(self, selected_skills: list[dict]) -> list[dict]:
        """Every enabled tool, every turn -- by request, trusting each tool's
        name/description to tell the model when to use it rather than
        gating the list per-skill. `allowed_tools` in skill files is no
        longer read for filtering (skills still contribute their
        `instructions` text to the system prompt, just not a tool allow-list).
        `github_dispatch_agent` stays excluded regardless -- it's an internal
        step resolve_approval takes after a delegation plan is approved, not
        something the router should ever pick directly.

        Trade-off worth knowing: this maximizes what any turn can reach for,
        but it also maximizes prompt size (full GitHub + every enabled MCP
        toolset) on every single request, which pushes harder against Groq's
        per-minute token limit than the filtered version did.
        """
        catalog = [t for t in self.github.openai_tools() + self.mcp.openai_tools() if t["function"]["name"] != "github_dispatch_agent"]
        return catalog + [self._delegation_tool()]

    # Caps how many tool calls the model can chain in a single turn before it
    # must produce an answer. Each step is a full Groq round trip, so this
    # bounds worst-case token/latency cost, not just infinite-loop risk.
    MAX_TOOL_STEPS = 4

    def _is_safe_to_chain(self, name: str) -> bool:
        """Only tools that need no human approval can execute mid-loop.
        Writes, delegation, and non-read-only MCP tools always stop the loop
        and fall through to the existing approval flow -- chaining never
        bypasses approval, it only chains together things that already ran
        without it."""
        if name == "delegate_to_gptoss":
            return False
        if name.startswith("mcp__"):
            return self.mcp.is_read_only(name)
        try:
            return not self.github.spec(name).write
        except ControlPlaneError:
            return False

    def route(self, text: str, history: list[dict], voice: bool = False) -> dict:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise ControlPlaneError("GROQ_API_KEY is not configured")
        model = self.config["tool_router"]
        skill_context = " ".join(
            [str(item.get("content", "")) for item in history[-12:] if item.get("role") == "user"]
            + [text]
        )
        selected_skills = self.skills.select(skill_context)
        tools = self._tools_for_skills(selected_skills)
        messages = self._messages(text, history, selected_skills)

        for _ in range(self.MAX_TOOL_STEPS):
            request_body = {
                "model": model,
                "messages": messages,
                "reasoning_effort": "none",
                "temperature": 0.2,
                "max_tokens": self.token_budget(text, history),
            }
            if tools:
                request_body["tools"] = tools
                request_body["tool_choice"] = "auto"
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=request_body,
                timeout=30,
            )
            if not response.ok:
                raise ControlPlaneError(f"Groq planner failed with HTTP {response.status_code}: {response.text[:500]}")
            payload = response.json()
            self.usage.add(model, payload.get("usage"))
            message = payload["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                return {"type": "message", "text": message.get("content") or "I need more detail before choosing a GitHub tool.", "model": model}
            call = calls[0]
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError as e:
                raise ControlPlaneError(f"The model produced invalid tool arguments: {e}")
            name = call["function"]["name"]
            # Voice can propose exactly the same tools as typed, including
            # MCP writes -- it never executes them itself. `_is_safe_to_chain`
            # is the only real gate here, and anything that fails it (any
            # write, delegate_to_gptoss, any non-read-only MCP tool) falls
            # through to `_submit`'s approval flow whether it came from voice
            # or the dashboard.
            if not self._is_safe_to_chain(name):
                return {"type": "tool", "name": name, "arguments": arguments, "model": model}

            # Safe read: run it now and hand the result back to the same
            # model call instead of stopping the turn, so "check X then read
            # Y" resolves in one pass -- and so this replaces the separate
            # summarize() round trip for the common single-tool case too.
            try:
                result = self.mcp.execute(name, arguments) if name.startswith("mcp__") else self.github.execute(name, arguments)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, default=str)[:4000]})

        return {"type": "message", "text": "I gathered some information but hit my per-turn tool-call limit -- ask again more specifically.", "model": model}

    def summarize(self, request: str, tool_name: str, result: Any) -> str:
        key = os.environ.get("GROQ_API_KEY")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": self.config["tool_router"], "messages": [{"role": "system", "content": "Summarize this read-only tool result as briefly as possible while fully answering the request. Use only supplied facts and no markdown."}, {"role": "user", "content": json.dumps({"request": request, "tool": tool_name, "result": result}, default=str)[:20000]}], "reasoning_effort": "none", "temperature": 0.2, "max_tokens": 180 if len(request.split()) < 20 else 280},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        self.usage.add(self.config["tool_router"], payload.get("usage"))
        return payload["choices"][0]["message"]["content"].strip()


class ControlPlane:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = _load_json(config_path, {})
        self.sessions = SessionStore()
        self.auth = GitHubAuth()
        self.usage = UsageMeter()
        self.github = GitHubTools(self.config["github"], self.auth)
        self.mcp = MCPRegistry(self.config.get("mcp_servers", {}), self.auth)
        skills_dir = ROOT / self.config.get("skills_directory", "skills")
        self.skills = SkillRegistry(skills_dir)
        self.router = ModelRouter(self.config["models"], self.skills, self.github, self.mcp, self.usage)
        self.approvals: dict[str, dict] = {}
        self.lock = threading.RLock()
        sessions = self.sessions.list()
        self.active_session_id = sessions[0]["id"] if sessions else self.sessions.create("Voice session")["id"]
        self.started_at = _now()

    def set_active_session(self, session_id: str):
        if not self.sessions.get(session_id):
            raise ControlPlaneError("Unknown session")
        self.active_session_id = session_id

    def create_session(self, title: str = "New session", source: str = "mixed") -> dict:
        session = self.sessions.create(title, source)
        self.active_session_id = session["id"]
        return session

    def set_mcp_enabled(self, name: str, enabled: bool) -> dict:
        server = self.config.get("mcp_servers", {}).get(name)
        if not server:
            raise ControlPlaneError("Unknown MCP server")
        server["enabled"] = bool(enabled)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        tmp.replace(self.config_path)
        return next(item for item in self.mcp.snapshot() if item["name"] == name)

    def set_skill_enabled(self, name: str, enabled: bool) -> dict:
        return self.skills.set_enabled(name, enabled)

    def save_skill(self, skill: dict, original_name: str | None = None) -> dict:
        return self.skills.save(skill, original_name)

    def delete_skill(self, name: str) -> None:
        self.skills.delete(name)

    def save_mcp_server(self, name: str, cfg: dict, original_name: str | None = None) -> dict:
        clean_name = str(name or "").strip().lower().replace(" ", "_")
        if not re.fullmatch(r"[a-z0-9_]+", clean_name or ""):
            raise ControlPlaneError("MCP server name must use lowercase letters, numbers, and underscores only")
        if not str(cfg.get("url", "")).strip():
            raise ControlPlaneError("MCP server needs a url")
        servers = self.config.setdefault("mcp_servers", {})
        if not original_name and clean_name in servers:
            raise ControlPlaneError(f"An MCP server named {clean_name!r} already exists")
        if original_name and original_name != clean_name and original_name in servers:
            del servers[original_name]
            self.mcp._tool_cache.pop(original_name, None)
        entry = {
            "enabled": bool(cfg.get("enabled", True)),
            "transport": cfg.get("transport") or "streamable_http",
            "url": cfg["url"].strip(),
            "read_only": bool(cfg.get("read_only", False)),
            "auth": cfg.get("auth") or None,
            "toolsets": [str(t).strip() for t in cfg.get("toolsets", []) if str(t).strip()],
        }
        if entry["auth"] == "static_token" and cfg.get("token_env"):
            entry["token_env"] = str(cfg["token_env"]).strip()
        servers[clean_name] = entry
        self.mcp._tool_cache.pop(clean_name, None)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        tmp.replace(self.config_path)
        return next(item for item in self.mcp.snapshot() if item["name"] == clean_name)

    def delete_mcp_server(self, name: str) -> None:
        servers = self.config.get("mcp_servers", {})
        if name not in servers:
            raise ControlPlaneError("Unknown MCP server")
        del servers[name]
        self.mcp._tool_cache.pop(name, None)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        tmp.replace(self.config_path)

    def github_overview(self, owner: str | None = None, repo: str | None = None) -> dict:
        owner = owner or self.config["github"].get("default_owner")
        repo = repo or self.config["github"].get("default_repository")
        result = {"repositories": self.github.execute("github_list_repositories", {})}
        if owner and repo:
            try:
                result["repository"] = self.github.execute("github_get_repository", {"owner": owner, "repo": repo})
            except Exception as e:
                result["repository"] = None
                result["repository_error"] = str(e)
            result["workflow_runs"] = self.github.execute("github_list_workflow_runs", {"owner": owner, "repo": repo})
            try:
                result["runners"] = self.github.execute("github_list_runners", {"owner": owner, "repo": repo})
            except Exception as e:
                result["runners"] = []
                result["runners_error"] = str(e)
        return result

    def record_voice_turn(self, user_text: str, assistant_text: str, model: str = "qwen/qwen3.6-27b"):
        self.sessions.append(self.active_session_id, {"kind": "message", "role": "user", "source": "voice", "content": user_text})
        self.sessions.append(self.active_session_id, {"kind": "message", "role": "assistant", "source": "voice", "model": model, "content": assistant_text})

    def _submit(self, session_id: str, text: str, source: str, voice: bool) -> dict:
        text = text.strip()
        if not text:
            raise ControlPlaneError("Command cannot be empty")
        self.set_active_session(session_id)
        self.sessions.append(session_id, {"kind": "message", "role": "user", "source": source, "content": text})
        history = self.sessions.messages(session_id, limit=40)
        try:
            routed = self.router.route(text, history, voice=voice)
        except Exception as e:
            routed = {"type": "message", "text": f"Routing failed: {e}", "model": "control-plane"}
        if routed["type"] == "message":
            self.sessions.append(session_id, {"kind": "message", "role": "assistant", "source": source, "model": routed.get("model"), "content": routed["text"]})
            return routed
        if routed["name"] == "delegate_to_gptoss":
            approval_id = f"a-{uuid.uuid4().hex[:12]}"
            approval = {"id": approval_id, "session_id": session_id, "tool": routed["name"], "arguments": routed["arguments"], "model": routed.get("model"), "status": "pending", "created_at": _now(), "expires_at": _now() + int(self.config["safeguards"].get("approval_ttl_seconds", 900))}
            with self.lock:
                self.approvals[approval_id] = approval
            self.sessions.append(session_id, {"kind": "approval", **approval})
            reply = "I have a scoped plan ready for GPT-OSS 20B. Review and approve it in the dashboard, and I can keep talking with you while it runs in the background."
            self.sessions.append(session_id, {"kind": "message", "role": "assistant", "source": source, "model": routed.get("model"), "content": reply})
            return {"type": "approval", "approval": approval, "text": reply, "model": routed.get("model")}
        is_mcp = routed["name"].startswith("mcp__")
        spec = None if is_mcp else self.github.spec(routed["name"])
        needs_approval = (is_mcp and not self.mcp.is_read_only(routed["name"])) or (spec and spec.write and self.config["safeguards"].get("require_approval_for_writes", True))
        if needs_approval:
            approval_id = f"a-{uuid.uuid4().hex[:12]}"
            approval = {"id": approval_id, "session_id": session_id, "tool": routed["name"], "arguments": routed["arguments"], "model": routed.get("model"), "status": "pending", "created_at": _now(), "expires_at": _now() + int(self.config["safeguards"].get("approval_ttl_seconds", 900))}
            with self.lock:
                self.approvals[approval_id] = approval
            self.sessions.append(session_id, {"kind": "approval", **approval})
            reply = "That action can change GitHub, so I placed it in the dashboard for your one-time approval."
            self.sessions.append(session_id, {"kind": "message", "role": "assistant", "source": source, "model": routed.get("model"), "content": reply})
            return {"type": "approval", "approval": approval, "text": reply, "model": routed.get("model")}
        result = self.mcp.execute(routed["name"], routed["arguments"]) if is_mcp else self.github.execute(spec.name, routed["arguments"])
        message = self.router.summarize(text, routed["name"], result)
        self.sessions.append(session_id, {"kind": "tool_result", "tool": routed["name"], "arguments": routed["arguments"], "result": result})
        self.sessions.append(session_id, {"kind": "message", "role": "assistant", "source": source, "model": routed.get("model"), "content": message})
        return {"type": "tool_result", "message": message, "result": result}

    def submit(self, session_id: str, text: str) -> dict:
        return self._submit(session_id, text, source="typed", voice=False)

    def submit_voice(self, text: str) -> dict:
        return self._submit(self.active_session_id, text, source="voice", voice=True)

    def resolve_approval(self, approval_id: str, approve: bool) -> dict:
        with self.lock:
            item = self.approvals.get(approval_id)
            if not item or item["status"] != "pending":
                raise ControlPlaneError("Approval is not pending")
            if _now() > item["expires_at"]:
                item["status"] = "expired"
                raise ControlPlaneError("Approval expired")
            item["status"] = "approved" if approve else "rejected"
        if not approve:
            self.sessions.append(item["session_id"], {"kind": "approval_result", "approval_id": approval_id, "status": "rejected"})
            return {"status": "rejected"}
        try:
            if item["tool"] == "delegate_to_gptoss":
                plan = item["arguments"]
                subtasks = plan.get("subtasks") or [{"objective": plan["objective"], "steps": plan["steps"], "acceptance_criteria": plan["acceptance_criteria"], "constraints": plan.get("constraints", [])}]
                runs = []
                for index, sub in enumerate(subtasks):
                    task = json.dumps({"objective": sub["objective"], "steps": sub["steps"], "acceptance_criteria": sub["acceptance_criteria"], "constraints": sub.get("constraints", [])}, indent=2)
                    # Each dispatch gets its own session id, which is also
                    # agent_worker.yml's concurrency-group key. Reusing the
                    # dashboard session id here (the old behavior) meant every
                    # approval from the same session queued behind the last
                    # one -- one worker at a time no matter how many plans you
                    # approved. A unique id per dispatch (and per subtask,
                    # when fanned out) puts each in its own concurrency group
                    # so they run as independent, parallel runners instead.
                    dispatch_session_id = f"{item['session_id']}-{approval_id}" + (f"-{index}" if len(subtasks) > 1 else "")
                    dispatch = self.github.execute("github_dispatch_agent", {"owner": plan["owner"], "repo": plan["repo"], "task": task, "session_id": dispatch_session_id, "ref": plan.get("base_branch") or "main"})
                    runs.append({**dispatch, "dispatch_session_id": dispatch_session_id, "objective": sub["objective"]})
                item["dispatch_owner"] = plan["owner"]
                item["dispatch_repo"] = plan["repo"]
                item["dispatch_runs"] = [{"dispatch_session_id": r["dispatch_session_id"], "objective": r["objective"], "seen_run_id": None, "reported": False} for r in runs]
                result = {"dispatched": len(runs), "runs": runs}
            else:
                result = self.mcp.execute(item["tool"], item["arguments"]) if item["tool"].startswith("mcp__") else self.github.execute(item["tool"], item["arguments"])
        except Exception as e:
            item["status"] = "failed"
            self.sessions.append(item["session_id"], {"kind": "tool_error", "approval_id": approval_id, "tool": item["tool"], "error": str(e)})
            raise
        item["status"] = "completed"
        self.sessions.append(item["session_id"], {"kind": "tool_result", "approval_id": approval_id, "tool": item["tool"], "arguments": item["arguments"], "result": result})
        if item["tool"] == "delegate_to_gptoss":
            n = result["dispatched"]
            message = (
                "GPT-OSS 20B is now working on the approved plan in GitHub Actions. I'll let you know here (and say something out loud) when it finishes."
                if n == 1 else
                f"GPT-OSS 20B is now working on {n} independent plans in parallel, each on its own GitHub Actions runner. I'll report back as each one finishes."
            )
        else:
            message = f"Completed the approved {item['tool']} action."
        self.sessions.append(item["session_id"], {"kind": "message", "role": "assistant", "source": "tool", "content": message})
        return {"status": "completed", "result": result}

    def snapshot(self, voice_status: dict | None = None) -> dict:
        now = _now()
        approvals = []
        with self.lock:
            for item in self.approvals.values():
                if item["status"] == "pending" and now > item["expires_at"]:
                    item["status"] = "expired"
                approvals.append({**item})
        return {
            "uptime_seconds": round(now - self.started_at),
            "active_session_id": self.active_session_id,
            "sessions": self.sessions.list(),
            "approvals": sorted(approvals, key=lambda a: a["created_at"], reverse=True),
            "github": self.auth.status(),
            "mcp_servers": self.mcp.snapshot(),
            "skills": self.skills.list(),
            "models": {**self.config["models"], "groq_key_configured": bool(os.environ.get("GROQ_API_KEY")), "worker_location": "github-actions"},
            "usage": self.usage.snapshot(),
            "voice": voice_status or {},
            "safeguards": self.config["safeguards"],
        }
