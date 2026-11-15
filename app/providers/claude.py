"""Claude (Anthropic API) editor: vision on contact sheets + forced tool call for the plan."""
from __future__ import annotations

import importlib.util

from .. import config as C
from ..playbook import PLAN_SCHEMA, SYSTEM_PROMPT_FULL, build_context_text, refine_suffix, sheet_images_b64
from .base import Provider, ProviderError, rejected

PLAN_TOOL = {
    "name": "submit_edit_plan",
    "description": "Submit the complete edit decision list for the vertical reel.",
    "input_schema": PLAN_SCHEMA,
}


class ClaudeProvider(Provider):
    name = "claude"
    polish_level = "light"

    @property
    def label(self):
        return f"Claude ({C.CLAUDE_MODEL})"

    def status(self):
        if not C.has_api_key():
            return {"available": False, "detail": "no ANTHROPIC_API_KEY"}
        if rejected("claude", C._env("ANTHROPIC_API_KEY") or C._env("ANTHROPIC_AUTH_TOKEN")):
            return {"available": False, "detail": "API key was rejected by Anthropic - update it in Settings"}
        if importlib.util.find_spec("anthropic") is None:
            return {"available": False, "detail": "anthropic package missing"}
        return {"available": True, "detail": C.CLAUDE_MODEL}

    def _call(self, text: str, analysis: dict) -> dict:
        import anthropic

        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": d}}
                   for d in sheet_images_b64(analysis)]
        content.append({"type": "text", "text": text + "\n\nReturn it by calling submit_edit_plan."})
        client = anthropic.Anthropic(timeout=240, max_retries=2)
        resp = client.messages.create(
            model=C.CLAUDE_MODEL, max_tokens=12000, system=SYSTEM_PROMPT_FULL, tools=[PLAN_TOOL],
            tool_choice={"type": "tool", "name": "submit_edit_plan"},
            messages=[{"role": "user", "content": content}],
        )
        u = getattr(resp, "usage", None)
        self.last_meta = {
            "model": getattr(resp, "model", None) or C.CLAUDE_MODEL,
            "usage": {"input_tokens": getattr(u, "input_tokens", None),
                      "output_tokens": getattr(u, "output_tokens", None)} if u is not None else None,
            "images_sent": len(content) - 1,
        }
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "submit_edit_plan":
                return dict(block.input)
        raise ProviderError("Claude did not return an edit plan")

    def plan(self, analysis, music, options, description, log, progress):
        progress(0.2, "Claude is watching the footage and planning the edit")
        return self._call(build_context_text(analysis, music, options, description), analysis)

    def refine(self, prev_plan, feedback, analysis, music, options, description, log, progress):
        progress(0.2, "Claude is revising the edit")
        text = build_context_text(analysis, music, options, description) + refine_suffix(prev_plan, feedback)
        return self._call(text, analysis)
