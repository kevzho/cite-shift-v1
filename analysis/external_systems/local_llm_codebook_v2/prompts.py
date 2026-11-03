"""Frozen prompt text for the project's own local-LLM codebook-v2 measured system.

The definitions below are copied from annotations/CODEBOOK_v2_DRAFT.md as FROZEN on
2026-10-03 (sha256 2acafc2e0d9b72b3ade90527048656dab41a67361493fe45c6f2a7b943bc7639)
sections 2-5 (synced 3 Oct from the 26 Sep draft, sha256 4d0af9be...; Pass 3 is dropped from the
study, so PASS3 below is retained only for the record and is not run), and from annotations/POLARITY_CODEBOOK_DRAFT.md
(sha256 87c49da7d81b439309623cbb064b83a57ada8d1722c2a9b7081728b137715052)
"Labels" + "Decision rules" for rhetorical_polarity, which v2 section 6 imports.

Deliberately EXCLUDED:
  * section 8 (worked anchors) - they quote canonical labels and rationales;
  * section 0 (motivation) - it quotes canonical-record statistics and rationales;
  * the one sentence in section 2 that points at anchor D2;
  * the 'Grounding' literature columns (not needed to apply a label; saves tokens);
  * sections 7, 9, 10 (calibration / reliability / crosswalk - not coding rules).

The prompt is frozen here as literal text (not re-read from the draft at run time) so
that the measured system stays pinned even when the draft codebook is edited.
Pass structure follows codebook section 1: three independent calls per edge; no pass
sees another pass's output.
"""
from __future__ import annotations

CODEBOOK_V2_SHA256 = "2acafc2e0d9b72b3ade90527048656dab41a67361493fe45c6f2a7b943bc7639"  # frozen 2026-10-03
POLARITY_SHA256 = "87c49da7d81b439309623cbb064b83a57ada8d1722c2a9b7081728b137715052"

PREAMBLE = """You are coding one citation edge (a citing paper citing a cited paper) with a fixed research codebook. You see only the evidence given: the citing paper's title, the cited paper's title, and every citing sentence (citance) for this edge. Use only that evidence. Apply the definitions exactly as written. Answer with one JSON object and nothing else.

Evidence rules:
- More than one citance: code the edge on the STRONGEST citance - the one that supports the most specific reading - and report its number in "strongest_citance".
- One citance, several cited papers: code only what the citance says about THIS cited paper. A bulk list gives every member the same weak evidence.
"""

PASS1 = """Fields to code in this pass: cited_contribution_type, citation_function.

cited_contribution_type - choose the contribution the CITED paper primarily makes (Wobbrock & Kientz, 2016):
- empirical: New knowledge from observation or data: experiments, user studies, field studies, surveys, log analyses
- artifact: A new system, tool, technique, toolkit, prototype, or design, arising from generative design work
- methodological: New knowledge about methods for research or practice: a method, procedure, measure, or instrument
- theoretical: New or improved concepts, definitions, models, principles, or frameworks
- dataset: A new reusable corpus, benchmark, or repository (standard tests or data), with characterization of it
- survey: A review or synthesis of existing work on a topic
- opinion: An argument meant to change the reader's mind
- unclear: The evidence does not show what the cited paper contributes

citation_function - why the citing context invokes the cited work. Assign one.
- background: Situates the citing work: prior work, context, or motivation, without the cited work entering the citing paper's procedure or argument as a component
- uses_method: Executes a procedure, measure, or instrument the cited work specifies
- applies_framework: Thinks with the cited work's concepts, vocabulary, or model without executing a procedure
- builds_on_finding: Treats a cited finding as evidence or premise for the citing paper's own claim
- reuses_artifact: Uses or builds on a cited system, tool, or dataset
- compares_against: Compares or contrasts the citing work, its results, or its approach with the cited work
- other: Clear evidence, but no category fits - say what in the note
- unclear: Cannot tell what the citation is doing

uses_method versus applies_framework: did they EXECUTE it (ran the procedure, got output) or THINK WITH it? If both, uses_method.

Return JSON with keys, in this order: "strongest_citance" (integer: the Citance number), "cited_contribution_type", "citation_function", "note" (at most 25 words).
"""

PASS2 = """Fields to code in this pass: credits_contribution, uptake_mode, uptake_degree, claim_dependent.
Each field is answered independently - no field's value implies another's.

credits_contribution - does the citance identify WHAT was contributed?
- specific: The citance names or describes the particular contribution - the artifact, the finding, the method, the claim - such that a reader could say what the cited paper did
- generic: The citance acknowledges the cited work only as part of an area, a list, or a gesture, without identifying its particular contribution
- unclear: Cannot tell
Specific crediting is not use.

uptake_mode - what, if anything, does the citing paper take from the cited paper? The citing paper ...
- method: runs a procedure, measure, or instrument from it
- artifact: uses or builds on a system, tool, or dataset from it
- finding: relies on a finding from it as a premise or evidence
- extension: modifies, adapts, or extends the contribution
- framing: adopts its problem framing, concepts, or identified gap to define its own problem
- design_knowledge: draws design ideas, patterns, or inspiration from it
- contrast: engages the contribution in order to argue against it or differ from it
- none: takes nothing from it: acknowledgement, context, or illustration only
- unclear: cannot tell
Realization is not endorsement: critique is a mode of uptake, recorded as contrast.
framing and design_knowledge are easy to miss. Look for them deliberately on background-looking citances: "inspired by", "following the notion of", "building on the observation that", a cited gap that the citing paper then fills.

uptake_degree - only if uptake_mode is not none or unclear:
- full: The contribution as a whole is taken up
- partial: A named, limited part is taken up - you can say which part (name it in the note)
- not_applicable: use this when uptake_mode is none or unclear
If the part cannot be named, choose full or reconsider none, and say why in the note.

claim_dependent - If you deleted the cited paper, would the citing paper's own claim, method, or artifact have to change? yes / no / unclear.
Framing and design-knowledge uptake will often be claim_dependent: no.
Evidence versus origin - the test asks whether a proposition the citing paper asserts would lose its support:
- yes: the citation IS THE EVIDENCE for a claim the citing paper makes; deleting it leaves that claim unsupported.
- no: the citation only says WHERE an idea, design choice or approach CAME FROM; deleting it removes the attribution, not the support for any claim (e.g. "inspired by" explains a design but evidences nothing).
Uptake and claim dependence are separate: an idea can be taken up (framing, design_knowledge) without any claim depending on it.

Independence rule. Every combination is legal. In particular, all of these are expected to occur:
- specific + none: credited precisely, not used
- generic + a mode other than none: used, credited vaguely
- claim_dependent no + framing or design_knowledge
If you notice yourself choosing one field's value because of another, re-read the citance for the second field alone.

Return JSON with keys, in this order: "strongest_citance" (integer: the Citance number), "credits_contribution", "uptake_mode", "uptake_degree", "claim_dependent", "note" (at most 25 words).
"""

PASS3 = """Fields to code in this pass: centrality, rhetorical_polarity.

centrality - core / bulk / unclear. bulk: the cited paper appears in a list of citations sharing one claim. core: it is singled out and discussed on its own. Textual prominence, judged from the citance form. Do not treat citation count or sentence length as centrality.

rhetorical_polarity - the citing author's stance toward the specific cited contribution, not the sentiment of the topic.
- supports: The context accepts, relies on, endorses, confirms, validates, or positively extends the cited contribution.
- neutral_use: The context uses, describes, attributes, or situates the contribution without an identifiable positive or negative evaluation.
- qualifies: The context accepts or uses the contribution while explicitly limiting its scope, conditions, generality, or adequacy.
- contrasts: The context challenges, contradicts, criticizes, rejects, or explicitly positions the citing work against the cited contribution.
- mixed: The available context contains more than one stance and no single stance faithfully represents the edge.
- unclear: The text is missing, ambiguous, or insufficient to determine stance.

Quotable-span test for supports: to label supports you must be able to point at the words in the citance that carry the evaluation and quote them in "evaluative_span". If you cannot quote a span, the answer is neutral_use. Evaluative spans look like: well-established, robust, has been shown, convincingly, seminal, effective, confirms, validates, consistent with, supports our. These are NOT evaluative spans: building on the work; using the work as a premise; citing it as the source of a fact; grouping it with other work; describing what it did at any length.

Decision rules:
1. Label the stance expressed in the provided context, not your opinion of either paper.
2. Do not infer support merely because a method or artifact is used.
3. Do not infer contrast merely because papers are compared.
4. Acknowledgement without evaluation is neutral_use.
5. A stated limitation with continued use is normally qualifies, not contrasts.
6. Use mixed only when distinct stances are actually present.
7. Use unclear rather than guessing from a title or venue.

Return JSON with keys, in this order: "strongest_citance" (integer: the Citance number), "centrality", "rhetorical_polarity", "evaluative_span" (quoted words, or empty string), "note" (at most 25 words).
"""

ENUMS = {
    "cited_contribution_type": ["empirical", "artifact", "methodological", "theoretical", "dataset",
                                "survey", "opinion", "unclear"],
    "citation_function": ["background", "uses_method", "applies_framework", "builds_on_finding",
                          "reuses_artifact", "compares_against", "other", "unclear"],
    "credits_contribution": ["specific", "generic", "unclear"],
    "uptake_mode": ["method", "artifact", "finding", "extension", "framing", "design_knowledge",
                    "contrast", "none", "unclear"],
    "uptake_degree": ["full", "partial", "not_applicable"],
    "claim_dependent": ["yes", "no", "unclear"],
    "centrality": ["core", "bulk", "unclear"],
    "rhetorical_polarity": ["supports", "neutral_use", "qualifies", "contrasts", "mixed", "unclear"],
}

PASS_FIELDS = {
    "pass1": ["strongest_citance", "cited_contribution_type", "citation_function", "note"],
    "pass2": ["strongest_citance", "credits_contribution", "uptake_mode", "uptake_degree",
              "claim_dependent", "note"],
    "pass3": ["strongest_citance", "centrality", "rhetorical_polarity", "evaluative_span", "note"],
}
PASS_TEXT = {"pass1": PASS1, "pass2": PASS2, "pass3": PASS3}

FORBIDDEN_IN_PROMPT = ["canonical", "anchor", "§8", "background_bulk", "realizes_contribution"]


def schema(pass_name: str) -> dict:
    props = {}
    for f in PASS_FIELDS[pass_name]:
        if f == "strongest_citance":
            props[f] = {"type": "integer"}
        elif f in ("note", "evaluative_span"):
            props[f] = {"type": "string"}
        else:
            props[f] = {"type": "string", "enum": ENUMS[f]}
    return {"type": "object", "properties": props, "required": PASS_FIELDS[pass_name]}


def system_prompt(pass_name: str) -> str:
    text = PREAMBLE + "\n" + PASS_TEXT[pass_name]
    low = text.lower()
    for bad in FORBIDDEN_IN_PROMPT:
        assert bad.lower() not in low, f"forbidden string in prompt: {bad}"
    return text


def evidence(citing_title: str, cited_title: str, citances: list[str]) -> str:
    lines = [f"Citing paper title: {citing_title or '(not available)'}",
             f"Cited paper title: {cited_title or '(not available)'}",
             f"Citances ({len(citances)}):"]
    lines += [f"Citance {i}: {c.strip()}" for i, c in enumerate(citances, 1)]
    return "\n".join(lines)
