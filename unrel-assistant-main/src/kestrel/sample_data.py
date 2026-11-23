"""Small demo corpus for tests and local runs (synthetic; not real policy)."""
DEMO_DOCS = {
    "policies/retention.md": (
        "Supplier certificates must be retained for seven years after the contract ends. "
        "Audit records are retained for ten years. Expired certificates must be archived, not deleted."),
    "policies/recall.md": (
        "A lot recall requires written notice to affected customers within 24 hours of the decision. "
        "The recall coordinator must log the notice and the affected lots in the quality system. "
        "LOT-00044 is currently under recall review and must not be shipped."),
    "catalog/sku-099.md": (
        "SKU-099 is the 250 ml stainless thermal flask. It ships in cartons of 12 units. "
        "SKU-099 is governed by clause 4.2 of the packaging standard, which requires double-wall insulation."),
    "policies/workflow.md": (
        "A lot may move from QUARANTINE to RELEASED only after two quality sign-offs. "
        "Transitions to REJECTED require a written reason and are final."),
}
