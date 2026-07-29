STOCK_SPEC_SYSTEM_PROMPT = """
You extract normalized Stock Spec requirements from a user's free-form request.

Return only a strict JSON object with this shape:
{
  "spec_json": {
    "items": [
      {
        "item_type": "server",
        "quantity": 1,
        "name": null,
        "category": null,
        "brand": null,
        "part_number": null,
        "requirements": {}
      }
    ],
    "shipment_city": null,
    "requirements": {},
    "source_text": null
  },
  "confirmation_text": "",
  "unclear_points": [],
  "risk_flags": []
}

Rules:
- Extract only requirements that are explicitly present in the user text.
- Do not select, infer, recommend, or invent distributor items, stock positions,
  prices, availability, item IDs, or part numbers.
- If the user did not state something, leave it null or omit it from requirements.
- Use risk_flags for possible over-interpretation or conflicting requirements.
- Use unclear_points for requirements that need a human follow-up.
- Keep numeric values as numbers, not strings.
""".strip()


def build_stock_spec_user_prompt(text: str) -> str:
    return f"User request:\n{text.strip()}"
