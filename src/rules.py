"""Small lookup over a few California rules that govern independent medical review.

The rules are short paraphrases of the statute text, each with its section number and a link to
the published statute. This is a research tool, not legal advice. Read the linked statute for the
exact wording.
"""
import re

URL_1374_33 = "https://law.justia.com/codes/california/2022/code-hsc/division-2/chapter-2-2/article-5-55/section-1374-33"
URL_1370_4 = "https://law.justia.com/codes/california/code-hsc/division-2/chapter-2-2/article-5/section-1370-4/"

RULES = [
    {
        "id": "HSC-1374.33(b)",
        "citation": "California Health and Safety Code section 1374.33(b)",
        "url": URL_1374_33,
        "text": ("Reviewers decide whether the disputed service was medically necessary for the "
                 "patient's specific needs. They consider peer-reviewed evidence on how well the "
                 "service works, nationally recognized professional standards, expert opinion, "
                 "generally accepted standards of medical practice, and treatments likely to help "
                 "when other treatments are not clinically effective."),
        "keywords": "medically necessary medical necessity evidence peer-reviewed standards expert opinion "
                    "accepted practice effective effectiveness treatment benefit not effective",
    },
    {
        "id": "HSC-1374.33(d)",
        "citation": "California Health and Safety Code section 1374.33(d)",
        "url": URL_1374_33,
        "text": ("Each review must say whether the service is medically necessary and explain why, "
                 "citing the patient's condition and the records. With several reviewers, the "
                 "majority view wins. If the reviewers are tied, the decision goes in favor of "
                 "providing the service."),
        "keywords": "decision tie tied majority reviewers analysis medically necessary in favor of "
                    "providing the service overturned",
    },
    {
        "id": "HSC-1374.33(f)",
        "citation": "California Health and Safety Code section 1374.33(f)",
        "url": URL_1374_33,
        "text": ("The state director adopts the review organization's decision, and the decision is "
                 "binding on the health plan."),
        "keywords": "binding plan director adopts determination decision final health plan must",
    },
    {
        "id": "HSC-1374.33(c)",
        "citation": "California Health and Safety Code section 1374.33(c)",
        "url": URL_1374_33,
        "text": ("The written decision is due within 30 days of the application. If the service has "
                 "not been provided and there is an imminent and serious threat to the patient's "
                 "health, the review is expedited to three days."),
        "keywords": "deadline days time 30 days expedited urgent imminent serious threat three days "
                    "timeline delay",
    },
    {
        "id": "HSC-1370.4(a)",
        "citation": "California Health and Safety Code section 1370.4(a)",
        "url": URL_1370_4,
        "text": ("For an experimental or investigational therapy, the patient must have a "
                 "life-threatening or seriously debilitating condition. A physician must certify "
                 "that standard therapies have failed, are not appropriate, or are not more "
                 "beneficial. The recommended therapy must be likely to be more beneficial than "
                 "standard options, supported by a physician certification or two medical and "
                 "scientific sources. The therapy must have been denied, and would be covered "
                 "except that the plan calls it experimental or investigational."),
        "keywords": "experimental investigational therapy life-threatening seriously debilitating "
                    "standard therapies failed physician certification two sources denied clinical trial "
                    "unproven new treatment",
    },
    {
        "id": "HSC-1370.4(d)",
        "citation": "California Health and Safety Code section 1370.4(d)",
        "url": URL_1370_4,
        "text": ("For experimental or investigational therapies, medical and scientific evidence "
                 "includes peer-reviewed studies and journals, indexed medical literature, "
                 "recognized drug compendia, federal research findings, and peer-reviewed abstracts "
                 "from major medical meetings."),
        "keywords": "experimental investigational medical scientific evidence peer-reviewed studies journals "
                    "literature compendia research abstracts",
    },
]

_WORD = re.compile(r"[a-z0-9\-]+")


def _words(text):
    return set(_WORD.findall((text or "").lower()))


def lookup_rules(query, k=3):
    """Return up to k rules that best match the query, most relevant first.

    Matching is simple keyword overlap. A rule with no overlap is never returned, so a query the
    rules don't cover gives an empty list instead of a made-up match.
    """
    q = _words(query)
    if not q or k <= 0:
        return []
    scored = []
    for rule in RULES:
        overlap = len(q & _words(rule["keywords"]))
        if overlap:
            scored.append((overlap, rule))
    scored.sort(key=lambda x: -x[0])
    return [dict(r) for _, r in scored[:k]]
