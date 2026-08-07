import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import control_plane as module


class ControlPlaneSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sessions_patch = patch.object(module, "SESSIONS_DIR", Path(self.temp.name) / "sessions")
        self.sessions_patch.start()
        self.control = module.ControlPlane()

    def tearDown(self):
        self.sessions_patch.stop()
        self.temp.cleanup()

    def test_write_is_approval_gated(self):
        self.control.router.route = Mock(
            return_value={
                "type": "tool",
                "name": "github_create_repository",
                "arguments": {"name": "safe-test", "private": True},
                "model": "qwen/qwen3.6-27b",
            }
        )
        result = self.control.submit(
            self.control.active_session_id,
            "Create a GitHub repository named safe-test",
        )
        self.assertEqual(result["type"], "approval")
        self.assertEqual(result["approval"]["status"], "pending")

    def test_unsafe_repository_paths_are_rejected(self):
        for path in ("../secrets", ".git/config", ".github/workflows/unsafe.yml"):
            with self.subTest(path=path), self.assertRaises(module.ControlPlaneError):
                self.control.github._validate_path(path)

    def test_destructive_mcp_names_are_blocked_before_connection(self):
        with self.assertRaisesRegex(module.ControlPlaneError, "blocked"):
            self.control.mcp.execute("mcp__github__delete_repository", {})

    def test_rejected_approval_never_executes(self):
        self.control.router.route = Mock(
            return_value={
                "type": "tool",
                "name": "github_create_folder",
                "arguments": {"owner": "TanishC4444", "repo": "runnerTests", "path": "demo"},
                "model": "qwen/qwen3.6-27b",
            }
        )
        proposed = self.control.submit(self.control.active_session_id, "Create demo folder")
        self.control.github.execute = Mock()
        result = self.control.resolve_approval(proposed["approval"]["id"], False)
        self.assertEqual(result["status"], "rejected")
        self.control.github.execute.assert_not_called()

    def test_skills_are_file_backed(self):
        names = {skill["name"] for skill in self.control.skills.list()}
        self.assertTrue({"github_reader", "github_engineering", "conversation"} <= names)

    def test_voice_read_executes_without_approval(self):
        self.control.router.route = Mock(
            return_value={
                "type": "tool",
                "name": "github_list_workflow_runs",
                "arguments": {"owner": "TanishC4444", "repo": "runnerTests"},
                "model": "qwen/qwen3.6-27b",
            }
        )
        self.control.github.execute = Mock(return_value=[{"status": "completed"}])
        self.control.router.summarize = Mock(return_value="The latest workflow completed.")
        result = self.control.submit_voice("Did the latest GitHub workflow finish?")
        self.assertEqual(result["type"], "tool_result")
        self.assertEqual(result["message"], "The latest workflow completed.")
        self.assertEqual(self.control.approvals, {})

    def test_delegation_is_approved_before_dispatch(self):
        plan = {
            "owner": "TanishC4444",
            "repo": "runnerTests",
            "base_branch": "main",
            "objective": "Add a health check",
            "steps": ["Inspect the service", "Implement the endpoint", "Run tests"],
            "acceptance_criteria": ["Health endpoint returns success"],
            "constraints": ["Do not edit workflows"],
        }
        self.control.router.route = Mock(
            return_value={"type": "tool", "name": "delegate_to_gptoss", "arguments": plan, "model": "qwen/qwen3.6-27b"}
        )
        proposed = self.control.submit(self.control.active_session_id, "Add a health check")
        self.assertEqual(proposed["type"], "approval")
        self.control.github.execute = Mock(return_value={"dispatched": True})
        result = self.control.resolve_approval(proposed["approval"]["id"], True)
        self.assertEqual(result["status"], "completed")
        call = self.control.github.execute.call_args
        self.assertEqual(call.args[0], "github_dispatch_agent")
        self.assertIn("acceptance_criteria", call.args[1]["task"])

    def test_qwen_coordinator_receives_history_skills_and_delegation_schema(self):
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "choices": [{"message": {"content": "Which repository should I use, and what result should the test verify?"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }
        history = [
            {"kind": "message", "role": "user", "content": "Build a health check"},
            {"kind": "message", "role": "assistant", "content": "What environment is this for?"},
            {"kind": "message", "role": "user", "content": "Production"},
        ]
        with patch.dict(module.os.environ, {"GROQ_API_KEY": "test-key"}), patch.object(module.requests, "post", return_value=response) as post:
            result = self.control.router.route("Production", history, voice=True)
        self.assertEqual(result["type"], "message")
        request = post.call_args.kwargs["json"]
        self.assertTrue(any(tool["function"]["name"] == "delegate_to_gptoss" for tool in request["tools"]))
        self.assertTrue(any(message.get("content") == "What environment is this for?" for message in request["messages"]))
        self.assertIn("no more than three", request["messages"][0]["content"])

    def test_token_budgets_expand_only_for_more_complex_intents(self):
        history = [{"kind": "message", "role": "user", "content": "Earlier context"}]
        short = self.control.router.token_budget("yes", history)
        read = self.control.router.token_budget("show the latest workflow status", history)
        plan = self.control.router.token_budget("implement and test a new API endpoint", history)
        self.assertLess(short, read)
        self.assertLess(read, plan)

    def test_repository_creation_passes_explicit_bootstrap_choices(self):
        responses = [
            {"login": "TanishC4444"},
            {"owner": {"login": "TanishC4444"}, "full_name": "TanishC4444/CompanyTest", "html_url": "https://github.com/TanishC4444/CompanyTest", "private": True},
        ]
        self.control.github._request = Mock(side_effect=responses)
        result = self.control.github.execute(
            "github_create_repository",
            {"name": "CompanyTest", "private": True, "auto_init": True, "license_template": "mit", "gitignore_template": "Python"},
        )
        self.assertEqual(result["full_name"], "TanishC4444/CompanyTest")
        create_call = self.control.github._request.call_args_list[1]
        self.assertEqual(create_call.kwargs["json"]["license_template"], "mit")
        self.assertTrue(create_call.kwargs["json"]["auto_init"])


if __name__ == "__main__":
    unittest.main()
