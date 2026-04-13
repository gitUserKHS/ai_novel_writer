from __future__ import annotations

import re
from typing import Iterable


def slugify_hf_fragment(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "artifact"


def infer_base_model_slug(model_name_or_path: str) -> str:
    text = str(model_name_or_path or "").strip()
    if not text:
        return "base"
    tail = text.split("/")[-1]
    return slugify_hf_fragment(tail)


def suggest_repo_id(
    namespace: str,
    *,
    repo_type: str = "model",
    project: str = "conarrative",
    role: str = "",
    base_model: str = "",
    stage: str = "",
) -> str:
    owner = slugify_hf_fragment(namespace)
    name_parts = [slugify_hf_fragment(project)]
    if repo_type == "dataset":
        if role:
            name_parts.append(slugify_hf_fragment(role))
        name_parts.append("corpus")
    else:
        if role:
            name_parts.append(slugify_hf_fragment(role))
        if base_model:
            name_parts.append(infer_base_model_slug(base_model))
        if stage:
            name_parts.append(slugify_hf_fragment(stage))
        name_parts.append("lora")
    return f"{owner}/{'-'.join(part for part in name_parts if part)}"


def next_release_tag(existing_tags: Iterable[str], *, prefix: str = "v", bump: str = "patch") -> str:
    semver_pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.(\d+)\.(\d+)$")
    versions: list[tuple[int, int, int]] = []
    for tag in existing_tags:
        match = semver_pattern.match(str(tag or "").strip())
        if match:
            versions.append(tuple(int(group) for group in match.groups()))
    if not versions:
        return f"{prefix}0.1.0"
    major, minor, patch = max(versions)
    if bump == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{prefix}{major}.{minor}.{patch}"
