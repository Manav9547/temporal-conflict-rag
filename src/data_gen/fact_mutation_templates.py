"""
Parameterized templates for generating (query, fresh_doc, stale_doc) fact-mutation
triples across 5 domains. Each domain has several "subtopics" (e.g. within
corporate_policy: remote work, parental leave, PTO) and, per subtopic, one or
more phrasing variants of the same underlying fact template.

Design goal: fresh_doc and stale_doc differ in exactly one thing — the fact
slot (a number, version string, or price) and its associated date — everything
else in the sentence is identical. This is what forces a retriever to learn a
genuine temporal/factual signal instead of exploiting lexical differences: a
bag-of-words or keyword-overlap retriever cannot tell d+ from d_r, since they
are near-identical strings.

Splits are assigned by *domain*, not by individual triple, to avoid leakage:
a template pattern seen at train time should not reappear at test time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

TRAIN_DOMAINS = ["corporate_policy", "software_version", "regulatory_guideline"]
OOD_VAL_DOMAIN = "pricing_tier"
OOD_TEST_DOMAIN = "product_spec"
ALL_DOMAINS = TRAIN_DOMAINS + [OOD_VAL_DOMAIN, OOD_TEST_DOMAIN]

FRESH_YEARS = [2025, 2026]
STALE_YEARS = [2018, 2019, 2020, 2021, 2022]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Boilerplate clauses appended (identically) to both fresh_doc and stale_doc.
# These add shared tokens that dilute the differing fact-slot tokens, pushing
# lexical (Jaccard) overlap toward the proposal's "near 1.0" target — without
# this, short single-sentence templates only reach ~0.6 overlap since the
# differing value/date tokens are a large fraction of a short sentence.
BOILERPLATE = {
    "corporate_policy": "This policy applies to all full-time employees and is reviewed annually by HR.",
    "software_version": "This information is maintained in the project's official release notes.",
    "regulatory_guideline": "Compliance with this figure is monitored by the relevant oversight authority.",
    "pricing_tier": "Pricing includes standard support and is subject to applicable taxes.",
    "product_spec": "Specifications are verified by the manufacturer's quality assurance team.",
}


def _combo_names(rng: random.Random, part1: list[str], part2: list[str], n: int, joiner: str = " ") -> list[str]:
    """Combinatorially generate up to n unique 'part1 + joiner + part2' names,
    used to scale entity pools without hand-listing hundreds of names."""
    combos = [f"{a}{joiner}{b}" for a in part1 for b in part2]
    rng.shuffle(combos)
    if n > len(combos):
        raise ValueError(f"Requested {n} unique combos but only {len(combos)} possible")
    return combos[:n]


# Fixed local seed so entity pools are deterministic across runs, independent
# of the value/date sampling rng passed into generate_domain_triples.
_NAME_RNG = random.Random(1234)


@dataclass
class Subtopic:
    label: str
    unit: str
    value_range: tuple[int, int]
    doc_templates: list[str]      # each has {entity} {label} {value} {unit} {date}
    query_templates: list[str]    # each has {entity} {label}

    def sample_value_pair(self, rng: random.Random) -> tuple[int, int]:
        lo, hi = self.value_range
        fresh = rng.randint(lo, hi)
        stale = rng.randint(lo, hi)
        while stale == fresh:
            stale = rng.randint(lo, hi)
        return fresh, stale


def _sample_date(rng: random.Random, years: list[int]) -> str:
    return f"{rng.choice(MONTHS)} {rng.choice(years)}"


def _version_value(rng: random.Random) -> str:
    return f"{rng.randint(1, 5)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"


# ---------------------------------------------------------------------------
# Entity pools (fictional, to avoid colliding with the generator's real-world
# pretraining knowledge and confounding the Stage-3 conflict labels).
# ---------------------------------------------------------------------------

# Word banks combined combinatorially (see _combo_names) to scale each entity
# pool into the hundreds without hand-listing every name. Sizes below are
# chosen so the final dataset lands in the proposal's 800-1500 triple range
# (see generate_retrieval_triples.py for the per-domain math).

_COMPANY_ADJ = [
    "Nimbus", "Alder", "Vantage", "Cobalt", "Fenwick", "Solace", "Greymark", "Halcyon",
    "Ironwood", "Lumen", "Marrow", "Northpoint", "Orbital", "Pallas", "Quillfeather",
    "Redshift", "Silverline", "Tessellate", "Underwood", "Vireo", "Wrenfield", "Axiom",
    "Brightbell", "Cordwood", "Driftwood", "Emberlynn", "Fallowfield", "Gravelrock",
    "Hearthstone", "Ivorytusk",
]
_COMPANY_NOUN = [
    "Systems", "Robotics", "Dynamics", "Analytics", "Labs", "Technologies", "Industries",
    "Data", "Ventures", "Software", "Biotech", "Logistics", "Materials", "Energy", "Media",
    "Cloud", "Financial", "AI", "Manufacturing", "Health", "Consulting", "Motors",
    "Insurance", "Foods", "Airlines", "Retail", "Agriculture", "Mining", "Realty", "Pharma",
]
COMPANY_NAMES = _combo_names(_NAME_RNG, _COMPANY_ADJ, _COMPANY_NOUN, n=100)

_PKG_PREFIX = [
    "fast", "stream", "auth", "queue", "shard", "grid", "token", "flow", "batch", "crypto",
    "vector", "trace", "log", "nether", "pixel", "torch", "data", "serve", "cluster", "hash",
    "proto", "pipe", "schema", "byte", "query", "cache", "net",
]
_PKG_SUFFIX = [
    "vector", "pipe", "kit", "wright", "cache", "loom", "forge", "mesh", "smith", "layer",
    "bind", "wire", "lathe", "hook", "span", "widget", "fitter",
]
PACKAGE_NAMES = _combo_names(_NAME_RNG, _PKG_PREFIX, _PKG_SUFFIX, n=100, joiner="")

REGULATION_METRICS = [
    ("Directive EC-{n}", "maximum continuous work hours", "hours", (6, 14)),
    ("Statute RG-{n}", "minimum wage", "USD per hour", (8, 25)),
    ("Guideline EM-{n}", "emission limit", "grams CO2 per km", (80, 200)),
    ("Regulation SF-{n}", "minimum safety inspection interval", "months", (1, 24)),
    ("Ordinance WT-{n}", "maximum water usage", "liters per day", (100, 500)),
]
REGULATION_N_RANGE = 60  # instances per metric, i.e. n = 1..60

_SAAS_ADJ = _COMPANY_ADJ[:15]
_SAAS_NOUN = [
    "CRM", "Analytics Suite", "Helpdesk", "Payroll", "Scheduler", "Inventory", "Docs",
    "ERP", "Messaging", "Forms", "Billing", "Storage", "Dashboards", "Notes", "Backup",
]
SAAS_PRODUCTS = _combo_names(_NAME_RNG, _SAAS_ADJ, _SAAS_NOUN, n=40)
PLAN_TIERS = ["Basic", "Pro", "Enterprise"]

_MODEL_WORD = [
    "Aster", "Bramble", "Cinder", "Driftline", "Ellsworth", "Fenmoor", "Glasswick",
    "Hollowmere", "Ivycross", "Juniper", "Kestrel", "Larchmont", "Mossgate", "Nightfall",
    "Oakspire", "Petrel", "Quailridge", "Rowanmere", "Slatehollow", "Thistledown",
]
_MODEL_CODE = [f"{letter}{n}" for letter in "XQVMRSTPKLJHNGF" for n in range(1, 21)]
PRODUCT_MODELS = _combo_names(_NAME_RNG, _MODEL_WORD, _MODEL_CODE, n=40)
PRODUCT_SPECS = [
    ("battery life", "hours", (4, 40)),
    ("storage capacity", "GB", (64, 2048)),
    ("weight", "grams", (100, 900)),
    ("max resolution", "megapixels", (8, 108)),
]


def _corporate_policy_subtopics() -> list[Subtopic]:
    return [
        Subtopic(
            label="remote work", unit="days per week", value_range=(1, 5),
            doc_templates=[
                "{entity}'s remote work policy allows {value} {unit}, effective {date}.",
                "As of {date}, {entity} permits {value} {unit} of remote work.",
            ],
            query_templates=[
                "What is {entity}'s current remote work policy?",
                "How many {unit} of remote work does {entity} allow?",
            ],
        ),
        Subtopic(
            label="parental leave", unit="weeks", value_range=(4, 26),
            doc_templates=[
                "{entity}'s parental leave policy grants {value} {unit}, effective {date}.",
                "As of {date}, {entity} offers {value} {unit} of parental leave.",
            ],
            query_templates=[
                "What is {entity}'s current parental leave policy?",
                "How many {unit} of parental leave does {entity} offer?",
            ],
        ),
        Subtopic(
            label="PTO accrual", unit="days per year", value_range=(10, 30),
            doc_templates=[
                "{entity}'s PTO accrual policy is {value} {unit}, effective {date}.",
                "As of {date}, employees at {entity} accrue {value} {unit} of PTO.",
            ],
            query_templates=[
                "What is {entity}'s current PTO accrual rate?",
                "How many {unit} of PTO do employees at {entity} accrue?",
            ],
        ),
    ]


def generate_domain_triples(domain: str, rng: random.Random, n_per_entity: int = 1) -> list[dict]:
    """Generate raw (pre-split) fact-mutation records for one domain."""
    records = []

    if domain == "corporate_policy":
        for subtopic in _corporate_policy_subtopics():
            for entity in COMPANY_NAMES:
                fresh_val, stale_val = subtopic.sample_value_pair(rng)
                fresh_date = _sample_date(rng, FRESH_YEARS)
                stale_date = _sample_date(rng, STALE_YEARS)
                doc_t = rng.choice(subtopic.doc_templates)
                query_t = rng.choice(subtopic.query_templates)
                records.append({
                    "domain": domain,
                    "subtopic": subtopic.label,
                    "entity": entity,
                    "query": query_t.format(entity=entity, unit=subtopic.unit),
                    "fresh_value": str(fresh_val),
                    "fresh_date": fresh_date,
                    "fresh_doc": doc_t.format(entity=entity, value=fresh_val, unit=subtopic.unit, date=fresh_date) + " " + BOILERPLATE[domain],
                    "stale_value": str(stale_val),
                    "stale_date": stale_date,
                    "stale_doc": doc_t.format(entity=entity, value=stale_val, unit=subtopic.unit, date=stale_date) + " " + BOILERPLATE[domain],
                })

    elif domain == "software_version":
        doc_templates = [
            "The recommended version of {entity} is {value}, released {date}.",
            "As of {date}, {entity} should be pinned to version {value}.",
        ]
        query_templates = [
            "What is the recommended version of {entity}?",
            "Which version of {entity} should be used?",
        ]
        for entity in PACKAGE_NAMES:
            fresh_val = _version_value(rng)
            stale_val = _version_value(rng)
            while stale_val == fresh_val:
                stale_val = _version_value(rng)
            fresh_date = _sample_date(rng, FRESH_YEARS)
            stale_date = _sample_date(rng, STALE_YEARS)
            doc_t = rng.choice(doc_templates)
            query_t = rng.choice(query_templates)
            records.append({
                "domain": domain,
                "subtopic": "package_version",
                "entity": entity,
                "query": query_t.format(entity=entity),
                "fresh_value": fresh_val,
                "fresh_date": fresh_date,
                "fresh_doc": doc_t.format(entity=entity, value=fresh_val, date=fresh_date) + " " + BOILERPLATE[domain],
                "stale_value": stale_val,
                "stale_date": stale_date,
                "stale_doc": doc_t.format(entity=entity, value=stale_val, date=stale_date) + " " + BOILERPLATE[domain],
            })

    elif domain == "regulatory_guideline":
        doc_templates = [
            "Under {entity}, the {metric} is {value} {unit}, as of {date}.",
            "As of {date}, {entity} sets the {metric} at {value} {unit}.",
        ]
        query_templates = [
            "What is the {metric} under {entity}?",
            "What does {entity} currently set as the {metric}?",
        ]
        for name_template, metric, unit, value_range in REGULATION_METRICS:
            for n in range(1, REGULATION_N_RANGE + 1):
                entity = name_template.format(n=n)
                lo, hi = value_range
                fresh_val = rng.randint(lo, hi)
                stale_val = rng.randint(lo, hi)
                while stale_val == fresh_val:
                    stale_val = rng.randint(lo, hi)
                fresh_date = _sample_date(rng, FRESH_YEARS)
                stale_date = _sample_date(rng, STALE_YEARS)
                doc_t = rng.choice(doc_templates)
                query_t = rng.choice(query_templates)
                records.append({
                    "domain": domain,
                    "subtopic": metric,
                    "entity": entity,
                    "query": query_t.format(entity=entity, metric=metric),
                    "fresh_value": str(fresh_val),
                    "fresh_date": fresh_date,
                    "fresh_doc": doc_t.format(entity=entity, metric=metric, value=fresh_val, unit=unit, date=fresh_date) + " " + BOILERPLATE[domain],
                    "stale_value": str(stale_val),
                    "stale_date": stale_date,
                    "stale_doc": doc_t.format(entity=entity, metric=metric, value=stale_val, unit=unit, date=stale_date) + " " + BOILERPLATE[domain],
                })

    elif domain == "pricing_tier":
        doc_templates = [
            "The {plan} plan for {entity} costs ${value}/month, effective {date}.",
            "As of {date}, {entity}'s {plan} plan is priced at ${value}/month.",
        ]
        query_templates = [
            "How much does the {plan} plan for {entity} cost?",
            "What is the current price of {entity}'s {plan} plan?",
        ]
        price_ranges = {"Basic": (5, 30), "Pro": (30, 100), "Enterprise": (100, 500)}
        for entity in SAAS_PRODUCTS:
            for plan in PLAN_TIERS:
                lo, hi = price_ranges[plan]
                fresh_val = rng.randint(lo, hi)
                stale_val = rng.randint(lo, hi)
                while stale_val == fresh_val:
                    stale_val = rng.randint(lo, hi)
                fresh_date = _sample_date(rng, FRESH_YEARS)
                stale_date = _sample_date(rng, STALE_YEARS)
                doc_t = rng.choice(doc_templates)
                query_t = rng.choice(query_templates)
                records.append({
                    "domain": domain,
                    "subtopic": plan,
                    "entity": entity,
                    "query": query_t.format(entity=entity, plan=plan),
                    "fresh_value": str(fresh_val),
                    "fresh_date": fresh_date,
                    "fresh_doc": doc_t.format(entity=entity, plan=plan, value=fresh_val, date=fresh_date) + " " + BOILERPLATE[domain],
                    "stale_value": str(stale_val),
                    "stale_date": stale_date,
                    "stale_doc": doc_t.format(entity=entity, plan=plan, value=stale_val, date=stale_date) + " " + BOILERPLATE[domain],
                })

    elif domain == "product_spec":
        doc_templates = [
            "The {entity} has a {spec} of {value} {unit}, as per the {date} specification update.",
            "As of {date}, the {entity}'s {spec} is rated at {value} {unit}.",
        ]
        query_templates = [
            "What is the {spec} of the {entity}?",
            "What is the current {spec} rating for the {entity}?",
        ]
        for entity in PRODUCT_MODELS:
            for spec, unit, value_range in PRODUCT_SPECS:
                lo, hi = value_range
                fresh_val = rng.randint(lo, hi)
                stale_val = rng.randint(lo, hi)
                while stale_val == fresh_val:
                    stale_val = rng.randint(lo, hi)
                fresh_date = _sample_date(rng, FRESH_YEARS)
                stale_date = _sample_date(rng, STALE_YEARS)
                doc_t = rng.choice(doc_templates)
                query_t = rng.choice(query_templates)
                records.append({
                    "domain": domain,
                    "subtopic": spec,
                    "entity": entity,
                    "query": query_t.format(entity=entity, spec=spec),
                    "fresh_value": str(fresh_val),
                    "fresh_date": fresh_date,
                    "fresh_doc": doc_t.format(entity=entity, spec=spec, value=fresh_val, unit=unit, date=fresh_date) + " " + BOILERPLATE[domain],
                    "stale_value": str(stale_val),
                    "stale_date": stale_date,
                    "stale_doc": doc_t.format(entity=entity, spec=spec, value=stale_val, unit=unit, date=stale_date) + " " + BOILERPLATE[domain],
                })

    else:
        raise ValueError(f"Unknown domain: {domain}")

    return records
