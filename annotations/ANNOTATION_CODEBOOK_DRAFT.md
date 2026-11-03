# Integrated citation-use annotation codebook

Frozen 29 August 2026 (`annotations/CODEBOOK_FREEZE.json`). Drafted with AI assistance; the research team
derived, justified and froze the constructs and definitions. On 31 October 2026 the original status note was
replaced by this paragraph (the frozen text has SHA-256 `fe36f574…`); no construct, definition or instruction
changed.

## Evidence shown to annotators

- cited paper title;
- citing paper title; and
- stored citation sentence/context.

Existing labels, rationales, corpus/source, venue, section, model outputs, and
aggregate findings remain hidden. Use an `unclear` label when the supplied
evidence cannot support a decision; do not infer from venue or presumed field.

### The `tier` column

The validation files carry a `tier` column with two values, `core` and
`extension`, and the rows are already in tier order. **Code every `core` row
before starting any `extension` row, and work down the file in the order given.**
Do not sort, filter, or reorder the file, and do not skip ahead --- the row order
is part of the sampling design, not a convenience.

If you run out of time, stop where you are and tell me where you stopped. A
partly finished `extension` tier is expected and is not a problem; a `core` tier
with gaps in it is, because `core` is the sample the analysis is committed to.

### One citance, several cited papers

A citance often cites several papers at once --- *"prior work has shown X
[12, 13, 14, 15]"* --- while you are asked about only one of them. In round-one
calibration this was roughly half of all rows, so it is the normal case, not an
edge case.

**Rule.** Label as though the claim the citance makes applies to the paper you
are asked about, and **tick `multi_reference` in the flag column**. Do not use
`unclear` for this: `unclear` means you could not read the evidence, and here
the evidence is perfectly readable --- it just describes a group. Flagging is
what lets us measure how often it happens and say what it costs.

### An edge with more than one citance

Citances on one edge are separated by `|||`. They may come from different
sections and may do different things.

**Rule.** Judge the **union** of them. If one citance uses the cited method and
another only names it as related work, both are true of the edge, so record both
under the multi-label instruction below and mark the primary as usual. Do not
average them and do not code only the first.

## Pass A: compact citation intent

Assign exactly one:

| Label | Operational definition |
|---|---|
| `background` | Situates the work, gives general context, or names related work without using a method or result |
| `method` | Uses, adopts, or discusses a method, procedure, tool, dataset, or experimental resource |
| `result` | Uses, compares with, supports, or disputes a finding or result |
| `unclear` | The supplied context is insufficient or does not fit |

This pass measures what a compact representation can express. Do not attempt
to encode contribution realization, centrality, or stance inside the compact
label.

## Pass B: multidimensional citation use

### Cited contribution type

Assign **every** type that applies, separated by semicolons, and mark one as
primary with a trailing `*`. A paper can genuinely be more than one thing: a
system evaluated in a study is both `artifact` and `empirical`.

- `artifact`;
- `dataset`;
- `empirical`;
- `methodological`;
- `opinion`;
- `survey`;
- `theoretical`; or
- `unclear`.

Example: `artifact*; empirical` --- both apply, artifact is primary.

**Which one is primary.** The contribution the citing paper's own claim,
method, or artifact depends on most directly. This is the same test used for
contribution realization below; apply it the same way in both places.

This is what the cited paper contributes, not what the citing sentence does.
If title and citation context do not establish the contribution, use
`unclear` and identify the additional evidence needed in the note.

### Citation function

Assign one of the canonical functions:

- `motivates_background`;
- `builds_on_finding`;
- `compares_against`;
- `applies_framework`;
- `reuses_artifact`;
- `uses_method`;
- `other`; or
- `unclear`.

Function answers why the citing context invokes the cited work. Assign one.

**`applies_framework` against `uses_method` --- did they *execute* it?** If the
citing paper ran a procedure the cited paper specified and got output from it,
that is `uses_method`. If they thought *with* the cited work --- its framing,
its vocabulary, its way of carving up the problem --- without running anything,
that is `applies_framework`. When both are true, choose `uses_method`: it is the
more specific claim.

*Executed:* "We measured workload using the NASA-TLX [12]." --- `uses_method`.
*Thought with:* "Following the sociotechnical view [12], we treat data access as
an organisational problem." --- `applies_framework`.

**`other` against `unclear`.** These are different states and both are legal
here. Use `other` when the evidence is clear but none of the named functions
fits. Use `unclear` when you cannot tell what the citation is doing. If you
reach for `unclear` because you suspect a better answer exists but cannot find
it, that is `other` --- say what was missing in the note.

### Contribution realization

Answer this in **two steps, in order**. Most disagreement in round-one
calibration came from coders answering the second question without having
settled the first.

**Step 1 --- the gate. Did the citing paper *use* the cited contribution at
all, or only mention it?**

The test: *if you deleted the cited paper, would the citing paper's own claim,
method, or artifact change?* If nothing would change, the citation is
background --- record `no_background` and stop. This holds no matter how much
detail the citance gives about the cited work. Describing a finding accurately
is not the same as using it.

**Step 2 --- the degree. Only if you passed the gate: did they use the whole
contribution, or part of it?**

| Label | Definition |
|---|---|
| `yes` | The contribution as a whole is used, applied, extended, validated, critiqued, or transferred |
| `partial` | A limited part of it is used, or it is used indirectly but substantively |
| `no_background` | Failed the gate: context or acknowledgement without use |
| `unclear` | You cannot tell from this citance |

*Gate failed:* "Cell technology must be developed to eliminate abnormal
iPSC-derived cells from the body (suicide switch) [39]." --- the citing paper's
argument stands without [39]. `no_background`.

*Gate passed, whole:* "We measured workload using the NASA-TLX [12]." --- the
method is taken entire. `yes`.

*Gate passed, part:* "We normalised within cancer type, unlike [21], which
pooled across types." --- one design choice of [21] is engaged, not its
findings. `partial`.

Critique or contradiction may still be `yes`: realization is not endorsement.

**When you mark `partial`, fill in `partial_reason`:** `part` if only part of
the contribution was used, or `type_mismatch` if the paper was used
substantively but not as the contribution type it was labelled. These are
different problems and we currently cannot tell them apart.

**A note on `unclear` here.** This dimension has no `other`, so `unclear` has to
carry both "I cannot tell from this citance" and "the categories do not fit."
Use it for either and say which in the note.

### Centrality

| Label | Definition |
|---|---|
| `core` | The cited work is individually discussed or materially supports a method, result, argument, comparison, or design decision |
| `bulk` | The citation appears primarily in a grouped list or broad contextual statement |
| `unclear` | The stored context cannot distinguish core from bulk |

Do not treat citation count or sentence length as centrality.

### Rhetorical polarity

Assign `supports`, `neutral_use`, `qualifies`, `contrasts`, `mixed`, or
`unclear` using
[`POLARITY_CODEBOOK_DRAFT.md`](POLARITY_CODEBOOK_DRAFT.md).

## Pass separation

- Complete Pass A and Pass B in different sessions or in a human-approved
  counterbalanced order.
- The files use the same edge IDs but different row orders.
- Do not consult the other pass while coding.
- Do not revise definitions after viewing final-sample agreement or model
  results.

## Reliability and adjudication

Report raw agreement, per-label support/confusion, and an appropriate nominal
chance-corrected reliability measure for each dimension before adjudication.
Preserve `unclear` and rare labels. Adjudication creates a new decision file
and never overwrites either annotator's original labels.

Five label values have no canonical equivalent and are marked
`exclude_nonmappable` in `label_crosswalk.csv`: `unclear` on contribution type,
function and realization, plus `other` on function and `qualifies` on polarity.
**Any comparison against the canonical labels states the proportion of rows
excluded as non-mappable.** The exclusion is a property of the two schemes not
lining up, and reporting a comparison without its exclusion rate would present a
subset as though it were the whole.

### Thresholds, declared in advance

Set per dimension, because the dimensions are not equally answerable from a
citance. Centrality, polarity and compact intent ask about the sentence in front
of the coder; contribution type, function and realization ask about a paper that
is not in the evidence at all.

| Dimension | Minimum |
|---|---|
| `compact_intent` | 0.60 |
| `centrality` | 0.60 |
| `rhetorical_polarity` | 0.60 |
| `cited_contribution_type` | 0.50 |
| `citation_function` | 0.50 |
| `contribution_realization` | 0.50 |

**A dimension that misses its threshold is reported, not dropped.** It appears
in the results with its reliability figure and an explicit statement that no
claim rests on it. Removing an underperforming dimension after seeing its number
is selective reporting, and this rule exists so that decision is already made.
