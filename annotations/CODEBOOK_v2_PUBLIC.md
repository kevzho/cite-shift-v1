> Public copy of the codebook frozen on 3 Oct 2026 (full text sha256 2acafc2e0d9b72b3ade90527048656dab41a67361493fe45c6f2a7b943bc7639). Quotations of the canonical record's rationales (3) are withheld; every definition, rule, label and anchor citance is unchanged.

# Citation-use annotation codebook — v2

**Status: FROZEN 3 October 2026** (`annotations/CODEBOOK_v2_FREEZE.json`). Drafted 26 September 2026.
The file name keeps `_DRAFT` so existing paths resolve. Passes 1–2 are in force; Pass 3 (§6) is
dropped. After the freeze the text does not change; interpretations go in the decision log.

Supersedes `ANNOTATION_CODEBOOK_DRAFT.md` (v1, frozen 29 August 2026) for all
coding begun after v2 is frozen. v1 remains the instrument of record for the
500-edge pass locked 26 September 2026.

> **v1 and v2 codings are never pooled.** They are different instruments. The
> v1 pass is evidence about v1; v2 codings are evidence about v2.

### Freeze gates — all four required before any v2 coding counts

1. **D-3 resolved** — signed off by the project lead, 28 September 2026.
2. **Every reference verified** by the lead against the source (§References).
   **Met 3 Oct 2026.**
3. **Anchors (§8) reviewed and coded** by the lead. **Met 3 Oct 2026.**
4. **Calibration shows every critical cell (§7) is reachable** — at least one
   observation in each after the 40-item calibration. **Met 3 Oct 2026** (5 / 8 / 11).

---

## 0. Why v2 exists

### Defect 1 — vocabulary fusion

v1 uses the word *background* twice with two meanings: as a citation-function
category (`motivates_background`), and as the outcome of the realization gate
("the citation is background — record `no_background` and stop"). The function
label was inherited from an LLM build-on typology in which it *meant* "not a
build-on": in the 109 HCI edges carrying that typology,
`llm_buildon_type = motivates_background` coincides with
`llm_is_genuine = FALSE` in **54 of 54** cases.

**Evidence.** In the lead's v1 pass (500 edges), all 269 `motivates_background`
edges are `no_background`. Function fully determines realization in 473/500
rows (94.6%). NMI(function, realization), geometric normalization, = 0.700,
against 0.308 (HCI) and 0.314 (Nature OA) in the canonical record and 0.887 in the
automated proposal pass. (The manuscript's 0.969 for that pass normalizes by the
smaller entropy; on that scale the v1 pass is 0.953. Name the normalization whenever
NMI is reported.) Within-background uptake is 0/269 — an identity, not an estimate,
matching the automated pass (0/253). On the same 500 edges the canonical record gives 128/304 = 42.1%.

**The canonical record uses the same vocabulary but a different boundary.** Its realization scale is
`yes` / `partial` / `no_background`, with no stated threshold: `partial` is "some use, but vague or
secondary"; `no_background` is "background/decorative citation; the contribution is not really used"
(`yiren_artifacts/README.md.csv`). Its function definition of `motivates_background` also says "no
specific use of the contribution", yet 289 HCI and 421 Nature OA background edges are coded `partial`
or `yes`. The fusion is one-directional there: `no_background` implies background (387/395 HCI,
543/549 Nature OA), but background does not imply `no_background` (387/676 HCI, 543/964 Nature OA). v1 made it
two-directional by replacing the vague boundary with an explicit gate (the deletion test).

### Defect 2 — a use-only test that cannot see framing or design knowledge

v1's gate asks: *if you deleted the cited paper, would the citing paper's own
claim, method, or artifact change?* That operationalizes **claim dependence**.
It cannot detect uptake of problem framing, concepts, or design knowledge — the
routes by which HCI contributions are argued to travel (Oulasvirta & Hornbæk,
2016; Zimmerman et al., 2007; Höök & Löwgren, 2012) — because a citing paper's
claim usually survives deleting what inspired it.

### Defect 3 — two constructs under one field name

`label_crosswalk.csv` maps canonical `realizes_contribution` to v1
`contribution_realization` as *mechanical, identical*. They are not the same
construct, **by their own written definitions.** The canonical sheet defines `yes` as
*"the citance clearly uses/credits a specific contribution of the cited paper"*
(`yiren_artifacts/README.md.csv`, line 20): crediting counts as realization. v1 defines
realization as **use** only, and says so: *"Describing a finding accurately is not the
same as using it."* Consistent with the canonical definition, its rationales frequently
deny use in the cells it codes as realization:

| Canonical cell | n | Rationale contains explicit non-use language* |
|---|---|---|
| HCI `background_bulk` × `no` | 387 | 70% |
| HCI `background_bulk` × `partial` | 261 | **58%** |
| HCI `background_bulk` × `yes` | 28 | 36% |
| Nature OA `background_bulk` × `partial` | 388 | **50%** |
| HCI `uses_method` × `yes` | 30 | 0% |
| Nature OA `uses_method` × `yes` | 103 | 1% |

\* Lexical screen: *does not (directly) adopt/reuse/use/apply/extend*,
*rather than adopting/reusing/using*, *without adopting*. A screen, not a
coding — superseded by the lead's hand-coded check (§9.4).

Example, canonical `background_bulk` × `partial`: *[canonical rationale withheld from the public release]*

Read by its rationales, canonical realization for background citations tracks
whether the citance **specifically engages what the cited paper contributed** —
names the artifact, states the finding — more than whether the citing paper
**uses** it. That is a coherent construct. It is not v1's, and the project's
claims about "uptake" were written as though it were.

### What v2 changes

1. The two constructs become **separate fields**: `credits_contribution`
   (specificity) and `uptake_mode` / `uptake_degree` (use).
2. v1's deletion test is **kept as its own field**, `claim_dependent`, so it is
   measured rather than imposed as a gate.
3. **No realization value reuses a function word.** The function category is
   renamed `background`, matching SciCite (Cohan et al., 2019); no realization
   label contains *background*.
4. Realization is coded in a **separate pass with function hidden**.
5. **Worked anchors** for the cells the study depends on.
6. **Every definition cited.**

### What contrasts are clean

v2 changes several things at once, so **v1 versus v2** can only be reported as
the effect of the instrument as a whole. Two contrasts *within* v2 are clean —
same coder, same items, same pass:

- `claim_dependent` versus `uptake_*` → isolates the operationalization (Defect 2)
- `credits_contribution` versus `uptake_*` → isolates the construct split (Defect 3)

---

## 1. Pass structure

| Pass | Fields | Shown to the coder |
|---|---|---|
| **1** | `cited_contribution_type`, `citation_function` | evidence (§2) |
| **2** | `credits_contribution`, `uptake_mode`, `uptake_degree`, `claim_dependent` | evidence only — **no Pass 1 labels** |
| **3** | `centrality`, `rhetorical_polarity` | evidence only |

Each pass is a separate file with an independent item order, completed
separately. Minimum 24 hours between finishing one pass and starting the next on
the same items. A coder who cannot see their function label cannot derive
realization from it.

## 2. Evidence shown

Citing title, cited title, and every citance for the edge. No abstracts, no full
text, no existing labels, no model output.

**More than one citance.** Code the edge on the **strongest** citance — the one
that supports the most specific reading — and record which in
`annotator_note`. Anchor D2 (§8) is the canonical case: its first citance is
background, a later one adopts the cited framework.

**One citance, several cited papers.** Code only what the citance says about
*this* cited paper. A bulk list gives every member the same weak evidence.

---

## 3. Cited contribution type — Pass 1

Wobbrock & Kientz (2016). Choose the contribution the **cited** paper primarily
makes.

| Label | Definition |
|---|---|
| `empirical` | New knowledge from observation or data: experiments, user studies, field studies, surveys, log analyses |
| `artifact` | A new system, tool, technique, toolkit, prototype, or design, arising from generative design work |
| `methodological` | New knowledge about methods for research or practice: a method, procedure, measure, or instrument |
| `theoretical` | New or improved concepts, definitions, models, principles, or frameworks |
| `dataset` | A new reusable corpus, benchmark, or repository (standard tests or data), with characterization of it |
| `survey` | A review or synthesis of existing work on a topic |
| `opinion` | An argument meant to change the reader's mind |
| `unclear` | The evidence does not show what the cited paper contributes |

## 4. Citation function — Pass 1

Why the citing context invokes the cited work. Assign one.

| Label | Definition | Grounding |
|---|---|---|
| `background` | Situates the citing work: prior work, context, or motivation, without the cited work entering the citing paper's procedure or argument as a component | SciCite *background* (Cohan et al., 2019); ACL-ARC *Background*, *Motivation* (Jurgens et al., 2018); Teufel *Neut*, *PMot* |
| `uses_method` | Executes a procedure, measure, or instrument the cited work specifies | ACL-ARC *Uses*; Teufel *PUse*; Moravcsik & Murugesan *operational* |
| `applies_framework` | Thinks with the cited work's concepts, vocabulary, or model without executing a procedure | Moravcsik & Murugesan *conceptual*; Teufel *PBas* |
| `builds_on_finding` | Treats a cited finding as evidence or premise for the citing paper's own claim | Teufel *PSup*, *PBas* |
| `reuses_artifact` | Uses or builds on a cited system, tool, or dataset | Teufel *PUse*; ACL-ARC *Uses* |
| `compares_against` | Compares or contrasts the citing work, its results, or its approach with the cited work | Teufel *CoCo* family; ACL-ARC *CompareOrContrast*; SciCite *Result Comparison* |
| `other` | Clear evidence, but no category fits — say what in the note | |
| `unclear` | Cannot tell what the citation is doing | |

`uses_method` versus `applies_framework` is v1's rule, unchanged: did they
**execute** it (ran the procedure, got output) or **think with** it? If both,
`uses_method`.

**Function says nothing about realization.** A `background` citation may credit
the cited contribution specifically, and may take it up. Pass 2 decides that, on
different evidence, without seeing this label.

---

## 5. Realization — Pass 2 (function hidden)

Four fields. Code all four for every item. **Each is answered independently —
no field's value implies another's.**

**Thresholds.** Each field has its own explicit boundary, taken from a line the
literature already draws. The canonical record's single `yes` / `partial` /
`no_background` scale states none (§0), which is why one label could carry both
crediting and use.

| Field | The boundary | Grounding |
|---|---|---|
| `credits_contribution` = `specific` | a reader could say what the cited paper specifically did | defined here (§5.1) |
| `uptake_mode` ≠ `none` | the citing paper **uses or extends** something from it, in one of the named modes | Valenzuela et al. (2015) *important* (uses or extends) vs *incidental* |
| `uptake_degree` = `partial` | a **named, limited part** is taken up — you can say which part | anchors B1, B2 |
| `claim_dependent` = `yes` | deleting the cited paper would force a change in the citing paper's own claim, method, or artifact | Moravcsik & Murugesan (1975) *organic* (necessity) |

These are four separate tests. Passing one says nothing about the others.

### 5.1 `credits_contribution` — does the citance identify *what* was contributed?

| Label | Definition |
|---|---|
| `specific` | The citance names or describes the particular contribution — the artifact, the finding, the method, the claim — such that a reader could say what the cited paper did |
| `generic` | The citance acknowledges the cited work only as part of an area, a list, or a gesture, without identifying its particular contribution |
| `unclear` | Cannot tell |

*Grounding.* `generic` corresponds to Moravcsik & Murugesan's (1975)
*perfunctory* reference — "mainly an acknowledgment that some other work in the
same general area has been performed." No established scheme names `specific`
directly; v2 defines it to make explicit what the canonical record's rationales
track (§0, Defect 3). **Specific crediting is not use.**

### 5.2 `uptake_mode` — what, if anything, does the citing paper take from it?

| Label | The citing paper … | Grounding |
|---|---|---|
| `method` | runs a procedure, measure, or instrument from it | Teufel *PUse*; ACL-ARC *Uses* |
| `artifact` | uses or builds on a system, tool, or dataset from it | Teufel *PUse*; Wobbrock & Kientz *artifact*, *dataset* |
| `finding` | relies on a finding from it as a premise or evidence | Teufel *PSup* |
| `extension` | modifies, adapts, or extends the contribution | Teufel *PModi*; ACL-ARC *Extends* |
| `framing` | adopts its problem framing, concepts, or identified gap to define its own problem | Oulasvirta & Hornbæk (2016) *conceptual problems*; Teufel *PMot*; Moravcsik & Murugesan *conceptual* |
| `design_knowledge` | draws design ideas, patterns, or inspiration from it | Zimmerman et al. (2007); Höök & Löwgren (2012) |
| `contrast` | engages the contribution in order to argue against it or differ from it | Moravcsik & Murugesan *negational* (confirmative/negational); Teufel *Weak*, *CoCo-* |
| `none` | takes nothing from it: acknowledgement, context, or illustration only | Moravcsik & Murugesan *perfunctory* |
| `unclear` | cannot tell | |

**Realization is not endorsement** (v1 rule, kept): critique is a mode of
uptake, recorded as `contrast`.

**`framing` and `design_knowledge` are the modes v1 could not see.** Look for
them deliberately on `background`-looking citances: *inspired by*, *following
the notion of*, *building on the observation that*, a cited gap that the citing
paper then fills.

### 5.3 `uptake_degree` — only if `uptake_mode` is not `none` or `unclear`

| Label | Definition |
|---|---|
| `full` | The contribution as a whole is taken up |
| `partial` | A named, limited part is taken up — you can say which part |

Leave blank when `uptake_mode` is `none` or `unclear`. (v1's "or taken up
indirectly but substantively" is dropped: it had no test, the same gap as the
canonical `partial`. If the part cannot be named, choose `full` or reconsider
`none`, and say why in the note.)

### 5.4 `claim_dependent` — v1's deletion test, kept as a measurement

*If you deleted the cited paper, would the citing paper's own claim, method, or
artifact have to change?* `yes` / `no` / `unclear`.

This is the v1 gate, verbatim, reported as its own field instead of deciding the
others. Framing and design-knowledge uptake will often be `claim_dependent: no`
— that combination is exactly what v1 erased.

**Evidence versus origin** (added 3 Oct, from anchors B1 and B2). The test asks
whether a *proposition* the citing paper asserts would lose its support:

- `yes` — the citation **is the evidence** for a claim the citing paper makes.
  Deleting it leaves that claim unsupported (anchor B1: the cited work is the
  evidence for the paper's claim about the state of the kawaii construct and its
  measurement).
- `no` — the citation only says **where an idea, design choice or approach came
  from**. Deleting it removes the attribution, not the support for any claim
  (anchor B2: "inspired by" explains the workshop's design but evidences
  nothing).

Uptake and claim dependence are separate: an idea can be taken up (`framing`,
`design_knowledge`) without any claim depending on it.

### 5.5 Independence rule

Every combination is legal. In particular, all of these are expected to occur:

- `specific` + `none` — credited precisely, not used
- `generic` + a mode other than `none` — used, credited vaguely
- `claim_dependent: no` + `framing` or `design_knowledge`

If you notice yourself choosing one field's value *because of* another, stop and
re-read the citance for the second field alone.

## 6. Centrality and polarity — Pass 3

**`centrality`** — `core` / `bulk` / `unclear`. `bulk`: the cited paper appears
in a list of citations sharing one claim. `core`: it is singled out and
discussed on its own. Textual prominence, judged from the citance form.
Grounding: Moravcsik & Murugesan's organic/perfunctory facet — this field takes
its *prominence* side; `claim_dependent` takes its *necessity* side.

**`rhetorical_polarity`** — the citing author's stance toward the cited work, not
the sentiment of the topic. v1 labels unchanged: `supports`, `neutral_use`,
`contrasts`, `qualifies`, `mixed`, `unclear`. Grounding: Teufel et al. (2006);
Athar (2011).

---

## 7. Critical cells — the calibration gate

v1 made these unobservable. After calibration, **each must hold at least one
observation**, or v2 has not fixed the problem and must be revised before real
coding:

1. `background` × `uptake_mode` ∈ {`finding`, `framing`, `design_knowledge`}
2. `background` × `specific` × `none`
3. `claim_dependent: no` × `uptake_mode` not in {`none`, `unclear`}

A zero after calibration is a codebook defect, not a finding.

---

## 8. Anchors

Drawn from the canonical record **outside** the 840-edge frozen sample
(seed 20260926). Each anchor was coded by the lead on the full citance set
before the proposals were compared (3 Oct 2026; `v2_prep/anchors_8.csv`), then
confirmed or replaced below. Anchors never enter the coding sample.

**A1 — specific, not used.** *Personalizing Content Moderation on Social Media.*
> "a user can choose to customize some element of moderation (e.g., via personal
> controls for unwanted content [22])"

Canonical: `background_bulk` / `partial`. Rationale: *[canonical rationale withheld from the public release]*
**Confirmed:** `background` · `specific` · `none` · `claim_dependent: no`.
*Boundary ruling:* not `artifact`. [22] is an example inside the design-space
category; no method, finding, artifact, framing or design knowledge is taken.

**A2 — credited, not used; canonical says partial.** *Eliciting and focusing
geographic volunteer work.*
> "The system has been widely studied … in terms of user's participation and
> behavioural analysis [19,15,16]."

Canonical: `background_bulk` / `partial`.
Proposed: `background` · `generic` · `none` · `claim_dependent: no`.
**Replaced:** `background` · `specific` · `none` · `claim_dependent: no`.
On the full citance set the strongest citance is *"In [19], for example, the
authors investigated the techniques and moti[vations] …"*: the cited work gets
specific credit, but the citing paper only summarizes it — nothing is adopted or
extended. Still no use, so canonical `partial` remains unsupported.

**B1 — framing (a cited gap).** *Kawaii Computing.*
> "However, almost all research has related to visual kawaii [70]."

Canonical: `background_bulk` / `partial`.
Proposed: `background` · `specific` · `framing` / `partial` · `claim_dependent: no`.
**Replaced:** `background` · `specific` · `framing` / `partial` · `claim_dependent: yes`.
The cited work helps define the conceptual and measurement problem the citing
paper works within (strongest citance: *"We did not have a validated measure for
kawaii, which remains elusive …"*). Reason for `claim_dependent: yes`: the
citation is the evidence for a concrete claim the citing paper makes about the
state of the kawaii construct and its measurement; deleting it leaves that claim
unsupported.

**B2 — design knowledge. v1's blind spot, exactly.** *Synaesthetic-Translation Tool.*
> "Our workshop is also inspired by past work on making the familiar strange
> through techniques for noticing [2] and synesthetic aides [21]."

Canonical: `background_bulk` / `partial`. Rationale: *[canonical rationale withheld from the public release]*
Proposed: `background` · `specific` · `design_knowledge` / `partial` ·
`claim_dependent: no`. v1 would record `no_background`; the workshop's design
came from [21].
**Confirmed:** `background` · `specific` · `design_knowledge` / `partial` ·
`claim_dependent: no`. "Inspired by" passes the uptake threshold; what is taken
is a design idea, not a framing of the research problem. The citation explains
where the design choice came from but supplies no evidence for any proposition
the paper makes, so deleting it changes no claim. (The lead first coded
`claim_dependent: yes`, revised on review, 3 Oct.) This is the §7 cell-3
pattern: uptake without claim dependence.

**C1 — perfunctory.** *CUI @ Auto-UI.*
> "There have been events examining human-machine collaboration [6,21] …"

Canonical: `background_bulk` / `no`.
**Confirmed:** `background` · `generic` · `none` · `claim_dependent: no`.
(`bulk` belonged to Pass 3, now dropped.)

**C2 — perfunctory.** *Integration of absolute multi-omics …*
> "the added value of metaproteomics has already been demonstrated in numerous
> examples …"

Canonical: `background_bulk` / `no`.
**Confirmed:** `background` · `generic` · `none` · `claim_dependent: no`.
One member of a long evidentiary bundle for a field-level statement.

**D1 — clear use.** *Making spaces: how design workbooks work.*
> "… the production of six substantial design workbooks [13] …"

Canonical: `uses_method` / `yes`.
**Confirmed:** `uses_method` · `specific` · `method` / `full` · `claim_dependent: yes`.
The citing paper reports producing six design workbooks as part of its process.

**D2 — multiple citances.** *Sketching AI Concepts with Capabilities and Examples.*
First citance: *"the high failure rate likely stems from poor quality ideation
[96,98,104]"* — background. A later citance adopts the cited "Do-Reason-Know"
framework for the system description. Code on the later citance (§2).
Function is a boundary between `applies_framework` and `uses_method`; the lead
decides.
**Lead ruling:** `applies_framework` · `specific` · `framing` / `full` ·
`claim_dependent: yes`. The system description explicitly *adopts* the
Do-Reason-Know framework (thought with, not executed), and the Concept Card is
described as inspired by it.

---

## 9. Reliability design — single coder

There is one coder. v2 therefore reports **stability**, never reproducibility.

### 9.1 Stability (test–retest)

Krippendorff's (2018, ch. 12) *stability*: the same coder, the same items, at two times.
The weakest of his three reliability types — it shows a coder applies a
consistent rule, not that another coder would agree. **Never reported as
inter-coder reliability.**

- Stratified retest sample (by corpus and Pass 1 function), **fixed before v2
  Pass 1 begins**.
- Minimum **four weeks** between passes, recorded. Blind: the retest file shows
  no prior labels, in a new order.
- Statistic fixed in advance: Cohen's κ (1960) per field, with raw agreement.

**Why v1 stability would have been uninformative.** A rule that fixes
realization from function reproduces perfectly over time. High test–retest on v1
would have measured the rule's determinism, not the coder's judgment. Stability
becomes informative only on an instrument that requires independent judgments.

### 9.2 Machine–coder agreement — a model evaluation, not reliability

A frozen model — pinned version, pinned prompt, temperature 0 — labels the same
items; agreement with the lead's labels is reported as **how well an automated
system reproduces one reader's judgments**. It is not reliability: the model is
the measured system, not a peer rater.

The model **must not have pre-annotated any v2 item.**

### 9.3 Drift

If both 9.1 and 9.2 run, the frozen model's labels are identical at both
timepoints, so any change in coder–model agreement between the two is the
**coder** moving.

### 9.4 Replacing the lexical screen

Before freeze, the lead hand-codes a random 50 canonical `background_bulk` ×
{`partial`, `yes`} rationales from outside the frozen sample for one question:
*does the rationale assert that the citing paper uses the cited contribution?*
That figure replaces the §0 lexical screen.

### 9.5 Schedule

| Date | |
|---|---|
| by 30 Sep | Gates 2–3 met (gate 1 met 28 Sep); §9.4 done |
| 1 Oct | Calibration (40 items); gate 4; **freeze** |
| 1–2 Oct | Retest stratum coded first |
| 29–30 Oct | Retest pass — each item ≥ 28 days after its first coding (1 Oct items on 29 Oct, 2 Oct items on 30 Oct) |

If the freeze slips, the retest slips by the same number of days. If it then
cannot finish before submission, report stability as future work. **Never
shorten the interval.**

---

## 10. Crosswalk

| v2 | v1 | canonical | Status |
|---|---|---|---|
| `background` | `motivates_background` | `background_bulk` | mechanical rename |
| other functions | unchanged | per `label_crosswalk.csv` | mechanical |
| `claim_dependent` | the v1 gate | — | conceptual: v1 gate text reused verbatim |
| `credits_contribution` | — | `realizes_contribution`? | **none — the relationship is an empirical question** |
| `uptake_mode` / `uptake_degree` | `contribution_realization` | `realizes_contribution`? | **none — the relationship is an empirical question** |

Do not map any v2 realization field onto canonical `realizes_contribution`.
Which v2 field canonical realization tracks is a result, not an assumption.

---

## References

**Verified by the lead against the sources, 3 Oct 2026 (gate 2).** Audit:
`annotations/generated/v2_prep/reference_check_17.csv`. Scope notes: Oulasvirta &
Hornbæk, Zimmerman et al. and Höök & Löwgren ground the `framing` and
`design_knowledge` modes conceptually; none of them defines a prior citation
category by those names. Athar is a citation-polarity precedent, not the source
of this codebook's polarity labels. Two open checks before publication: the
Moravcsik & Murugesan quotation against the p. 90 scan, and any page-specific
Krippendorff citation against a 4th-edition copy (cite ch. 12 otherwise).

- Athar, A. (2011). Sentiment analysis of citations using sentence structure-based features. *Proceedings of the ACL 2011 Student Session*, 81–87.
- Cohan, A., Ammar, W., van Zuylen, M., & Cady, F. (2019). Structural scaffolds for citation intent classification in scientific publications. *NAACL-HLT 2019*, 3586–3596. (Classes: Background, Method, Result Comparison.)
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement, 20*(1), 37–46.
- Duan, C., & Tan, Z. (2025). Semantically orthogonal framework for citation classification: Disentangling intent and content. In *Linking Theory and Practice of Digital Libraries (TPDL 2025)*, LNCS, 183–206. https://doi.org/10.1007/978-3-032-05409-8_12 (arXiv:2601.05103 is a later repository version.)
- Höök, K., & Löwgren, J. (2012). Strong concepts: Intermediate-level knowledge in interaction design research. *ACM Transactions on Computer-Human Interaction, 19*(3), Article 23, 23:1–23:18.
- Jurgens, D., Kumar, S., Hoover, R., McFarland, D., & Jurafsky, D. (2018). Measuring the evolution of a scientific field through citation frames. *Transactions of the ACL, 6*, 391–406. (Six functions: Background, Motivation, Uses, Extension, Comparison or Contrast, Future.)
- Krippendorff, K. (2018). *Content analysis: An introduction to its methodology* (4th ed.). SAGE. Ch. 12, Reliability.
- Kunnath, S. N., Herrmannova, D., Pride, D., & Knoth, P. (2022). A meta-analysis of semantic classification of citations. *Quantitative Science Studies, 2*(4), 1170–1215. (Online 2021.)
- Lauscher, A., Ko, B., Kuehl, B., Johnson, S., Cohan, A., Jurgens, D., & Lo, K. (2022). MultiCite: Modeling realistic citations requires moving beyond the single-sentence single-label setting. *NAACL 2022*, 1875–1889.
- Moravcsik, M. J., & Murugesan, P. (1975). Some results on the function and quality of citations. *Social Studies of Science, 5*(1), 86–92. (Four dichotomies: conceptual/operational, organic/perfunctory, evolutionary/juxtapositional, confirmative/negational.)
- Oulasvirta, A., & Hornbæk, K. (2016). HCI research as problem-solving. *CHI 2016*, 4956–4967.
- Teufel, S., Siddharthan, A., & Tidhar, D. (2006). Automatic classification of citation function. *EMNLP 2006*, 103–110. (Twelve labels: Weak, CoCoGM, CoCo-, CoCoR0, CoCoXY, PBas, PUse, PModi, PMot, PSim, PSup, Neut; "CoCo family" means the four CoCo labels.)
- Valenzuela, M., Ha, V., & Etzioni, O. (2015). Identifying meaningful citations. *Scholarly Big Data: AI Perspectives, Challenges, and Ideas* (AAAI Workshop, Technical Report WS-15-13), 21–26. Important = 14.6% (69/465) of their released dataset — not a general rate.
- Wobbrock, J. O., & Kientz, J. A. (2016). Research contributions in human-computer interaction. *interactions, 23*(3), 38–44.
- Zimmerman, J., Forlizzi, J., & Evenson, S. (2007). Research through design as a method for interaction design research in HCI. *CHI 2007*, 493–502.

**Known gap.** MultiCite argues citations are multi-label; v2 records one primary
`uptake_mode` with any secondary in the note. A limitation, stated.
