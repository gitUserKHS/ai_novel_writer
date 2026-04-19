from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict, List


def render_memory_context(bundle: Dict[str, Any]) -> str:
    story = bundle["story"]
    bible = bundle["bible"]
    state = bundle["state"]
    recent = bundle["recent_scenes"]
    outline = bundle.get("outline", [])
    parts: List[str] = []
    parts.append(f"Title: {story['title']}")
    parts.append(f"Genre: {story['genre']}")
    parts.append(f"Premise: {story['premise']}")
    parts.append(f"Tone: {story['tone']}")
    if story.get("themes"):
        parts.append("Themes: " + ", ".join(story["themes"]))
    if story.get("characters"):
        parts.append("Characters: " + ", ".join(story["characters"]))
    if story.get("forbidden_facts"):
        parts.append("Forbidden facts / rules: " + "; ".join(story["forbidden_facts"]))
    if bible.get("static_facts"):
        parts.append("Static facts: " + "; ".join(bible["static_facts"]))
    if bible.get("rules"):
        parts.append("Rules: " + "; ".join(bible["rules"]))
    if bible.get("motifs"):
        parts.append("Motifs: " + "; ".join(bible["motifs"]))
    if bible.get("voice_notes"):
        parts.append("Voice notes: " + "; ".join(bible["voice_notes"]))
    if state.get("current_time_label"):
        parts.append(f"Current time label: {state['current_time_label']}")
    if state.get("current_location"):
        parts.append(f"Current location: {state['current_location']}")
    if state.get("active_threads"):
        parts.append("Active threads: " + "; ".join(state["active_threads"]))
    if state.get("resolved_threads"):
        parts.append("Resolved threads: " + "; ".join(state["resolved_threads"]))
    if state.get("character_knowledge"):
        parts.append("Character knowledge: " + str(state["character_knowledge"]))
    if state.get("inventory"):
        parts.append("Inventory: " + str(state["inventory"]))
    if state.get("emotional_state"):
        parts.append("Emotional state: " + str(state["emotional_state"]))
    if state.get("summary_memory"):
        parts.append("Recent memory summaries: " + " || ".join(state["summary_memory"][-3:]))
    if recent:
        parts.append("Recent scenes:")
        for scene in recent:
            parts.append(
                f"- Scene {scene['scene_index']} [{scene['time_label']} @ {scene['location']}] {scene['title']}: {scene['summary']}"
            )
    if outline:
        parts.append("Outline cards:")
        for card in outline:
            parts.append(
                f"- {card['title']} ({card['status']}) POV={card['pov']} Goal={card['goal']} Time={card['time_label']} Location={card['location']}"
            )
    return "\n".join(parts)


def outline_system_prompt() -> str:
    return dedent(
        """
        You are a story architect for long-form fiction.
        Produce a practical scene-by-scene outline that is coherent and causally sound.
        All content fields must be written in natural Korean unless the user explicitly requested another language.
        Obey the user premise and continuity constraints closely.
        Return JSON only.
        """
    ).strip()


def outline_user_prompt(bundle: Dict[str, Any], scene_count: int) -> str:
    memory = render_memory_context(bundle)
    return dedent(
        f"""
        Build {scene_count} outline cards for the following project.

        {memory}

        Write the semantic content of every field in Korean.

        Output JSON with this exact shape:
        {{
          "cards": [
            {{
              "title": "",
              "pov": "",
              "goal": "",
              "location": "",
              "time_label": "",
              "summary_request": "",
              "beats": [""],
              "must_include": [""],
              "must_avoid": [""],
              "status": "planned"
            }}
          ]
        }}
        """
    ).strip()


def planner_system_prompt() -> str:
    return dedent(
        """
        You are the plot planner for a scene-based fiction system.
        Produce compact, high-signal plans that strictly respect continuity.
        All generated content must be in Korean unless the user explicitly requested another language.
        Avoid global rewrites and stay tightly aligned to the request.
        Return JSON only.
        """
    ).strip()


def planner_user_prompt(bundle: Dict[str, Any], request: Dict[str, Any]) -> str:
    memory = render_memory_context(bundle)
    return dedent(
        f"""
        Plan the next scene using the project memory below.

        MEMORY
        {memory}

        USER SCENE REQUEST
        {request}

        Write every string field in Korean.

        Output JSON with shape:
        {{
          "scene_title": "",
          "synopsis": "",
          "beats": [{{"label": "", "purpose": ""}}],
          "expected_reveals": [""],
          "expected_new_threads": [""],
          "expected_resolved_threads": [""],
          "expected_state_delta": {{}}
        }}
        """
    ).strip()


def writer_system_prompt() -> str:
    return dedent(
        """
        You are the prose writer in a scene-based fiction engine.
        Write vivid, coherent, literary scene prose in natural Korean unless the user explicitly requested another language.
        Avoid meta commentary, headings, and placeholders.
        Follow the plan, user request, and continuity constraints closely.
        Do not switch to English for narration unless a proper noun or explicit quote requires it.
        """
    ).strip()


def writer_user_prompt(bundle: Dict[str, Any], request: Dict[str, Any], plan: Dict[str, Any], variant_hint: str) -> str:
    memory = render_memory_context(bundle)
    return dedent(
        f"""
        Write one complete scene.

        MEMORY
        {memory}

        SCENE REQUEST
        {request}

        PLAN
        {plan}

        VARIANT STRATEGY
        {variant_hint}

        Requirements:
        - Keep the requested POV, location, and time label coherent.
        - Include all must_include items unless they violate continuity.
        - Avoid all must_avoid items.
        - Target the requested emotional arc and approximate length.
        - Write only the scene prose in Korean.
        """
    ).strip()


def consistency_system_prompt() -> str:
    return dedent(
        """
        You are a continuity and logic critic for long-form fiction.
        Focus on timeline, causality, location continuity, knowledge leakage, rule breaks, and user constraints.
        Be precise and evidence-based.
        Write issue messages in Korean unless the user explicitly requested another language.
        Return JSON only.
        """
    ).strip()


def consistency_user_prompt(bundle: Dict[str, Any], request: Dict[str, Any], plan: Dict[str, Any], scene_text: str) -> str:
    memory = render_memory_context(bundle)
    return dedent(
        f"""
        Audit this scene.

        MEMORY
        {memory}

        REQUEST
        {request}

        PLAN
        {plan}

        SCENE
        {scene_text}

        Output JSON with shape:
        {{
          "score": 0.0,
          "checks_run": ["timeline", "causality", "coverage"],
          "verdict": "accept" | "needs_revision",
          "issues": [
            {{
              "issue_type": "timeline|location|causality|knowledge_leak|inventory|character|rule|style|coverage|other",
              "severity": "low|medium|high",
              "message": "",
              "evidence": "",
              "suggested_fix": ""
            }}
          ]
        }}
        """
    ).strip()


def creativity_system_prompt() -> str:
    return dedent(
        """
        You are a creativity critic for literary fiction scenes.
        Write the evaluation content in Korean unless the user explicitly requested another language.
        Rate novelty, hook strength, emotional depth, and language.
        Return JSON only.
        """
    ).strip()


def creativity_user_prompt(bundle: Dict[str, Any], request: Dict[str, Any], plan: Dict[str, Any], scene_text: str) -> str:
    memory = render_memory_context(bundle)
    return dedent(
        f"""
        Evaluate the creative quality of this scene.

        MEMORY
        {memory}

        REQUEST
        {request}

        PLAN
        {plan}

        SCENE
        {scene_text}

        Output JSON with shape:
        {{
          "novelty_score": 0.0,
          "hook_score": 0.0,
          "emotional_depth_score": 0.0,
          "language_score": 0.0,
          "summary": "",
          "strengths": [""],
          "opportunities": ["" ]
        }}
        """
    ).strip()


def revision_system_prompt() -> str:
    return dedent(
        """
        You are a surgical fiction reviser.
        Keep the scene's best language when possible, but fix the listed issues.
        Do not produce notes; return only the revised scene prose.
        """
    ).strip()


def revision_user_prompt(bundle: Dict[str, Any], request: Dict[str, Any], plan: Dict[str, Any], scene_text: str, issues: List[Dict[str, Any]]) -> str:
    memory = render_memory_context(bundle)
    return dedent(
        f"""
        Revise the scene below.

        MEMORY
        {memory}

        REQUEST
        {request}

        PLAN
        {plan}

        ISSUES TO FIX
        {issues}

        ORIGINAL SCENE
        {scene_text}
        """
    ).strip()


def extraction_system_prompt() -> str:
    return dedent(
        """
        You extract machine-readable memory updates from scenes.
        Separate static facts from dynamic state and event graph edges.
        Return JSON only.
        """
    ).strip()


def extraction_user_prompt(bundle: Dict[str, Any], request: Dict[str, Any], plan: Dict[str, Any], scene_text: str) -> str:
    memory = render_memory_context(bundle)
    return dedent(
        f"""
        Extract the memory update for this scene.

        MEMORY
        {memory}

        REQUEST
        {request}

        PLAN
        {plan}

        SCENE
        {scene_text}

        Output JSON with shape:
        {{
          "summary": "",
          "new_static_facts": [""],
          "state_updates": {{"current_time_label": "", "current_location": "", "active_threads_add": [""], "resolved_threads_add": [""]}},
          "new_threads": [""],
          "resolved_threads": [""],
          "knowledge_updates": {{"Character": ["fact"]}},
          "inventory_updates": {{"Character": ["item"]}},
          "emotional_updates": {{"Character": "emotion"}},
          "kg_edges": [{{"source": "", "relation": "", "target": "", "edge_type": "event", "metadata": {{}}}}],
          "tags": [""]
        }}
        """
    ).strip()
