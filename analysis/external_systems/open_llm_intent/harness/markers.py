"""Locate numeric citation markers in a citance and wrap the target one(s) in <cite>...</cite>.

MultiCite (Lauscher et al., NAACL 2022) models were trained on text in which the *target*
citation marker is wrapped in <cite>...</cite> (e.g. "BERT <cite>[2]</cite> and Open-GPT [3]").
test_edges.csv does not say which marker in a citance is the target, so this module resolves it
deterministically and records HOW it was resolved (``resolution``):

  intersection   the edge has >= 2 citances and exactly one marker number occurs in all of them
                 -> only that number is wrapped (inside a multi-number group, the number and its
                 trailing separator are wrapped, as in MultiCite's training data "<cite>[1,</cite> 6]")
  single_group   exactly one marker group in the citance -> the whole group is wrapped
                 (target inside the group is not identified)
  ambiguous      several candidate groups -> every candidate group is wrapped
  no_marker      no numeric marker found -> text passed unchanged

No human label or model output is used here. Deterministic (pure regex).
"""
import re

NUM = r"\d{1,3}"
SEP = r"\s*[,–—\-]\s*"
BRACKET = rf"\[\s*{NUM}(?:{SEP}{NUM})*\s*\]"
# run of bracket groups, optionally preceded by bare numbers ("30,[36][37][38]") -> one group
BRACKET_RUN = re.compile(rf"(?<![\w\]])(?:{NUM}\s*,\s*)*{BRACKET}(?:\s*,?\s*{BRACKET})*")
PAREN = re.compile(rf"\(\s*{NUM}(?:{SEP}{NUM})*\s*\)")
# Nature-style flattened superscripts: " 51,52 ." / " 31 associated" / " 16 (Supp"
BARE = re.compile(rf"(?<=[A-Za-z0-9\)\]])\s({NUM}(?:\s*,\s*{NUM}|\s*[–\-]\s*{NUM})*)(?=\s*[.,;:]|\s+[a-z(]|\s*$)")


def _numbers(s):
    out = set()
    for a, b in re.findall(rf"({NUM})(?:\s*[–\-]\s*({NUM}))?", s):
        a = int(a)
        if b and 0 < int(b) - a < 50:
            out.update(range(a, int(b) + 1))
        else:
            out.add(a)
            if b:
                out.add(int(b))
    return out


def find_groups(text, corpus):
    """Return list of (start, end, numbers) marker groups, non-overlapping, in text order."""
    groups = []
    for rx in (BRACKET_RUN, PAREN):
        for m in rx.finditer(text):
            if any(not (m.end() <= s or m.start() >= e) for s, e, _ in groups):
                continue
            groups.append((m.start(), m.end(), _numbers(m.group(0))))
    if not groups and corpus.startswith("nature"):
        for m in BARE.finditer(text):
            groups.append((m.start(1), m.end(1), _numbers(m.group(1))))
    return sorted(groups)


def _wrap_number_in_group(text, s, e, n):
    """Wrap one number inside a group, MultiCite style ("<cite>[1,</cite> 6]")."""
    seg = text[s:e]
    m = re.search(rf"(\[\s*)?(?<!\d){n}(?!\d)(\s*\]|\s*,)?", seg)
    if m is None:  # number only inside a range such as [10-14]: wrap the whole group
        return text[:s] + "<cite>" + seg + "</cite>" + text[e:]
    return text[:s + m.start()] + "<cite>" + m.group(0) + "</cite>" + text[s + m.end():]


def mark_edge(citances, corpus):
    """citances: list[str] of one edge. Returns (marked_list, resolution, target_numbers)."""
    per = [find_groups(c, corpus) for c in citances]
    target = None
    if len(citances) >= 2:
        sets = [set().union(*[g[2] for g in gs]) if gs else set() for gs in per]
        inter = set.intersection(*sets) if sets else set()
        if len(inter) == 1:
            target = next(iter(inter))
        elif len(inter) > 1:
            target = inter  # ambiguous among these numbers
    marked, res = [], []
    for c, gs in zip(citances, per):
        if not gs:
            marked.append(c); res.append("no_marker"); continue
        if isinstance(target, int):
            hit = [g for g in gs if target in g[2]]
            if hit:
                s, e, nums = hit[0]
                if len(nums) == 1:
                    marked.append(c[:s] + "<cite>" + c[s:e] + "</cite>" + c[e:])
                else:
                    marked.append(_wrap_number_in_group(c, s, e, target))
                res.append("intersection"); continue
        cand = gs
        if isinstance(target, set):
            cand = [g for g in gs if g[2] & target] or gs
        if len(cand) == 1:
            s, e, _ = cand[0]
            marked.append(c[:s] + "<cite>" + c[s:e] + "</cite>" + c[e:])
            res.append("single_group" if len(gs) == 1 else "ambiguous")
            continue
        out = c
        for s, e, _ in sorted(cand, reverse=True):
            out = out[:s] + "<cite>" + out[s:e] + "</cite>" + out[e:]
        marked.append(out); res.append("ambiguous")
    tnums = sorted(target) if isinstance(target, set) else ([target] if target is not None else [])
    return marked, res, tnums
