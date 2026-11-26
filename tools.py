import json
import os

_DATA_PATH = os.path.join(os.path.dirname(__file__), "DATA-orders.json")
_data = None


def _load():
    global _data
    if _data is None:
        with open(_DATA_PATH) as f:
            _data = json.load(f)
    return _data


def get_today() -> str:
    return _load()["today"]


def get_user_orders(user_id: str) -> dict:
    data = _load()
    user = next((u for u in data["users"] if u["user_id"] == user_id), None)
    orders = [o for o in data["orders"] if o["user_id"] == user_id]
    return {
        "today": data["today"],
        "user": user,
        "orders": orders,
    }


TOOL_SCHEMA = {
    "name": "get_user_orders",
    "description": (
        "Fetch the authenticated user's account info and all their Sezzle orders, "
        "including installment schedules, payment statuses, refunds, and reschedule counts. "
        "Call this whenever the question is about a specific order, payment date, amount, "
        "refund status, or account state. Do NOT answer order-specific questions from memory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The user's ID, e.g. 'u002'.",
            }
        },
        "required": ["user_id"],
    },
}
