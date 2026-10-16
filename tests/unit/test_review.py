"""Reviewer verdict parsing — the trust boundary between a reply and the graph.

Design §8 pins the JSON shape; ``prompts.VERDICT_JSON_SPEC`` tells the model to emit
it. The reply is untrusted text: it may be fenced, wrapped in prose, or malformed, and
the graph acts on the result, so anything ambiguous must raise rather than guess.
Extra keys are the single tolerated deviation (rationale in ``orch.review``).
"""

import pytest

from orch.review import Finding, Verdict, extract_json_object, parse_verdict

APPROVE = '{"verdict": "approve", "risky": false, "findings": [], "summary": "looks good"}'
REJECT = (
    '{"verdict": "reject", "risky": true, '
    '"findings": [{"severity": "blocker", "file": "src/a.py", "note": "wrong"}], '
    '"summary": "not yet"}'
)


# --- extract_json_object ------------------------------------------------------


def test_fenced_json_block_is_extracted():
    assert extract_json_object(f"```json\n{APPROVE}\n```") == {
        "verdict": "approve",
        "risky": False,
        "findings": [],
        "summary": "looks good",
    }


def test_bare_object_is_found_with_chatter_before_and_after():
    text = f"Here is my review.\n{REJECT}\nHope that helps."
    assert extract_json_object(text)["verdict"] == "reject"


def test_bare_scan_skips_braces_in_prose():
    # `{the criteria}` is not a JSON object; the scan must keep looking.
    text = f"I weighed {{the criteria}} and concluded:\n{APPROVE}\nDone."
    assert extract_json_object(text)["verdict"] == "approve"


def test_fenced_block_wins_over_a_bare_object():
    assert extract_json_object(f"{REJECT}\n```json\n{APPROVE}\n```")["verdict"] == "approve"


def test_last_parseable_fenced_block_wins():
    text = (
        f"```json\n{APPROVE}\n```\n"
        '```json\n{"verdict": "reject", broken\n```\n'
        f"```json\n{REJECT}\n```\n"
    )
    assert extract_json_object(text)["summary"] == "not yet"


def test_a_broken_final_fenced_block_does_not_hide_the_earlier_one():
    text = f"```json\n{APPROVE}\n```\n```json\nnonsense\n```\n"
    assert extract_json_object(text)["verdict"] == "approve"


def test_a_fenced_non_object_falls_back_to_the_bare_object():
    text = f"```json\n[1, 2, 3]\n```\n{APPROVE}"
    assert extract_json_object(text)["verdict"] == "approve"


@pytest.mark.parametrize("text", ["", "no json here", "{broken", '{"a": }', "[1, 2, 3]"])
def test_no_json_object_raises(text):
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object(text)


# --- parse_verdict: happy paths -----------------------------------------------


def test_parse_fenced_verdict():
    assert parse_verdict(f"```json\n{REJECT}\n```") == Verdict(
        verdict="reject",
        risky=True,
        findings=[Finding(severity="blocker", file="src/a.py", note="wrong")],
        summary="not yet",
    )


def test_parse_bare_verdict_surrounded_by_chatter():
    v = parse_verdict(f"Verdict follows.\n{APPROVE}\nEnd.")
    assert v.verdict == "approve" and v.risky is False and v.findings == []


def test_parse_multiple_fenced_blocks_takes_the_last_parseable():
    text = f"```json\n{APPROVE}\n```\n```json\n{REJECT}\n```\n"
    assert parse_verdict(text).verdict == "reject"


def test_surplus_keys_are_ignored_at_every_level():
    text = (
        '{"verdict": "approve", "risky": false, "findings": '
        '[{"severity": "minor", "note": "nit", "line": 3}], "summary": "ok", "model": "x"}'
    )
    v = parse_verdict(text)
    assert v.findings == [Finding(severity="minor", file="", note="nit")]


# --- parse_verdict: strict failures -------------------------------------------


def test_wrong_verdict_value_raises():
    with pytest.raises(ValueError, match="verdict"):
        parse_verdict(APPROVE.replace("approve", "maybe"))


def test_missing_summary_raises():
    with pytest.raises(ValueError, match="summary"):
        parse_verdict('{"verdict": "approve", "risky": false, "findings": []}')


def test_missing_risky_raises():
    with pytest.raises(ValueError, match="risky"):
        parse_verdict('{"verdict": "approve", "findings": [], "summary": "ok"}')


def test_missing_findings_raises():
    with pytest.raises(ValueError, match="findings"):
        parse_verdict('{"verdict": "approve", "risky": false, "summary": "ok"}')


def test_wrong_typed_required_field_raises():
    # Lax coercion would read "false" as False; a wrong-typed field is an error.
    text = '{"verdict": "approve", "risky": "false", "findings": [], "summary": "ok"}'
    with pytest.raises(ValueError, match="risky"):
        parse_verdict(text)


def test_bad_finding_severity_raises():
    text = (
        '{"verdict": "reject", "risky": false, "findings": '
        '[{"severity": "critical", "file": "a.py", "note": "boom"}], "summary": "no"}'
    )
    with pytest.raises(ValueError, match="severity"):
        parse_verdict(text)


def test_findings_must_be_a_list_of_objects():
    text = '{"verdict": "reject", "risky": false, "findings": ["a string"], "summary": "no"}'
    with pytest.raises(ValueError, match="findings"):
        parse_verdict(text)


@pytest.mark.parametrize("text", ["", "I approve this change.", "{}"])
def test_unparseable_or_empty_text_raises(text):
    with pytest.raises(ValueError):
        parse_verdict(text)
