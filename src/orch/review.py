"""Strict parsing of a Reviewer session's reply into a verdict.

The reviewer prompt (``orch.prompts.VERDICT_JSON_SPEC``) pins the JSON shape, but a
reply is untrusted text: a model may fence the object, wrap it in prose, or emit
fields that do not match the contract. This module is the trust boundary between that
text and the graph, so ``parse_verdict`` either returns a validated ``Verdict`` or
raises ``ValueError`` — it never guesses at an ambiguous reply.

Required fields are strict; surplus keys are ignored. A model that adds a commentary
field is not ambiguous about the *verdict* itself, and a retry costs a whole reviewer
session, so tolerating extra keys is the cheaper trade. A missing or wrong-typed
required field, or a ``verdict`` outside ``approve``/``reject``, *is* an error: those
are exactly the places where the graph would otherwise act on a guess.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

_FENCED_JSON_RE = re.compile(r"```json[^\n]*\n(.*?)```", re.DOTALL)


class Finding(BaseModel):
    """One reviewer finding; ``file``/``note`` default to empty."""

    # strict: a wrong-typed field is an error, not coerced (JSON types map exactly
    # onto these fields, so a conforming reply is unaffected).
    model_config = ConfigDict(strict=True, extra="ignore")

    severity: Literal["blocker", "major", "minor"]
    file: str = ""
    note: str = ""


class Verdict(BaseModel):
    """A validated reviewer verdict — the only shape the graph acts on."""

    # extra="ignore" (also pydantic's default): surplus keys are tolerated, per the
    # module docstring. Written out so the intent is explicit rather than inherited.
    # strict: lax mode would coerce e.g. "false" to False; a wrong-typed required
    # field must raise instead of being read as a verdict the model never gave.
    model_config = ConfigDict(strict=True, extra="ignore")

    verdict: Literal["approve", "reject"]
    risky: bool
    findings: list[Finding]
    summary: str


def extract_json_object(text: str) -> dict:
    """Return the JSON object contained in ``text``; raise ``ValueError`` if none.

    Fenced ```json blocks take precedence (the prompt asks for one) and the *last*
    one that parses as an object wins, because a model that shows a draft and then
    its final answer ends with the answer. Otherwise the text is scanned left to
    right for the first ``{`` at which ``raw_decode`` yields an object, which
    tolerates prose — and stray braces — before and after the reply.
    """
    for block in reversed(_FENCED_JSON_RE.findall(text)):
        try:
            loaded = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded

    decoder = json.JSONDecoder()
    index = text.find("{")
    while index != -1:
        try:
            loaded, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        if isinstance(loaded, dict):
            return loaded
        index = text.find("{", index + 1)

    raise ValueError("reviewer output contains no JSON object")


def _summarize(error: ValidationError) -> str:
    """A short, bounded rendering of pydantic's errors — never the reply text."""
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in error.errors()
    )


def parse_verdict(text: str) -> Verdict:
    """Validate a reviewer's reply into a ``Verdict``, or raise ``ValueError``.

    The message names what failed (extraction vs. schema) without dumping the reply.
    """
    try:
        payload = extract_json_object(text)
    except ValueError as e:
        raise ValueError(f"reviewer verdict is unparseable: {e}") from e
    try:
        return Verdict.model_validate(payload)
    except ValidationError as e:
        raise ValueError(f"reviewer verdict is invalid: {_summarize(e)}") from e
