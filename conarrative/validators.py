from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .models import ConsistencyIssue, IssueType, PlanOutput, SceneRequest, Severity


def rule_based_consistency(
    memory_bundle: Dict[str, Any],
    request: SceneRequest,
    plan: PlanOutput,
    scene_text: str,
) -> List[ConsistencyIssue]:
    """Deterministic guardrails that complement the LLM critic."""

    issues: List[ConsistencyIssue] = []
    text = scene_text or ""
    lowered = text.lower()

    for item in _clean_items(request.must_include):
        if item and item not in text:
            issues.append(
                ConsistencyIssue(
                    issue_type=IssueType.COVERAGE,
                    severity=Severity.HIGH,
                    message=f"필수 포함 요소가 장면에 없습니다: {item}",
                    evidence=item,
                    suggested_fix=f"장면 안에 '{item}'를 행동, 이미지, 대사 중 하나로 자연스럽게 포함하세요.",
                )
            )

    forbidden_items = list(_clean_items(request.must_avoid))
    story = memory_bundle.get("story", {})
    bible = memory_bundle.get("bible", {})
    forbidden_items.extend(_clean_items(story.get("forbidden_facts", [])))
    forbidden_items.extend(_clean_items(bible.get("rules", [])))
    for item in forbidden_items:
        if item and item.lower() in lowered:
            issues.append(
                ConsistencyIssue(
                    issue_type=IssueType.RULE,
                    severity=Severity.HIGH,
                    message=f"금지/규칙 위반 표현이 장면에 직접 등장합니다: {item}",
                    evidence=item,
                    suggested_fix="금지 사실을 직접 서술하지 말고 우회하거나 해당 문장을 제거하세요.",
                )
            )

    for marker in ("TODO", "TBD", "[", "]", "<", ">", "여기에", "작성 필요"):
        if marker.lower() in lowered:
            issues.append(
                ConsistencyIssue(
                    issue_type=IssueType.OTHER,
                    severity=Severity.MEDIUM,
                    message=f"초안/플레이스홀더 흔적이 남아 있습니다: {marker}",
                    evidence=marker,
                    suggested_fix="플레이스홀더를 실제 장면 문장으로 치환하세요.",
                )
            )
            break

    for risk in plan.contradiction_risks[:6]:
        risk_text = risk.strip()
        if risk_text and risk_text in text:
            issues.append(
                ConsistencyIssue(
                    issue_type=IssueType.CAUSALITY,
                    severity=Severity.MEDIUM,
                    message="계획 단계의 모순 위험 문장이 본문에 그대로 노출되었습니다.",
                    evidence=risk_text,
                    suggested_fix="위험 설명을 제거하고 장면 내부 사건으로만 표현하세요.",
                )
            )

    min_words = max(80, int(request.desired_length_words * 0.35))
    rough_words = _rough_word_count(text)
    if rough_words < min_words:
        issues.append(
            ConsistencyIssue(
                issue_type=IssueType.COVERAGE,
                severity=Severity.MEDIUM,
                message="장면 분량이 요청 대비 지나치게 짧습니다.",
                evidence=f"rough_words={rough_words}, requested={request.desired_length_words}",
                suggested_fix="핵심 비트를 더 구체적인 행동, 감각, 대사로 확장하세요.",
            )
        )

    return issues


def merge_rule_issues(
    report_issues: Sequence[ConsistencyIssue],
    rule_issues: Sequence[ConsistencyIssue],
) -> List[ConsistencyIssue]:
    seen = set()
    merged: List[ConsistencyIssue] = []
    for issue in [*report_issues, *rule_issues]:
        key = (issue.issue_type.value, issue.severity.value, issue.message, issue.evidence)
        if key in seen:
            continue
        seen.add(key)
        merged.append(issue)
    return merged


def _clean_items(items: Any) -> List[str]:
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    return [str(item).strip() for item in items if str(item).strip()]


def _rough_word_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(len(stripped.split()), len(stripped) // 8)
