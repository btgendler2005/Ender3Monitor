"""Suggested sell-price calculation for a finished print.

Standard maker formula (all rates come from Settings, editable in the UI):

    material    = (grams / 1000) * filament_price_per_kg
    electricity = (watts / 1000) * hours * electricity_rate_per_kwh
    machine     = hours * machine_rate_per_hour   # wear + depreciation + failure buffer
    labor       = labor_flat                      # flat post-processing fee
    ───────────────────────────────────────────────
    cost        = material + electricity + machine + labor
    price       = cost * markup_multiplier

Filament weight is the manual `filament_grams` setting (your slicer reports it
per model). Print time is measured by the monitor.
"""
from __future__ import annotations

from typing import List, Optional


def compute_price(print_seconds: Optional[float], grams: float, settings) -> Optional[dict]:
    """Compute the cost breakdown + suggested price, or None if pricing is off."""
    if not settings.get("pricing_enabled"):
        return None

    hours = max(0.0, float(print_seconds or 0)) / 3600.0
    grams = max(0.0, float(grams or 0))

    material = (grams / 1000.0) * float(settings.get("filament_price_per_kg"))
    electricity = (float(settings.get("printer_watts")) / 1000.0) * hours \
        * float(settings.get("electricity_rate_per_kwh"))
    machine = hours * float(settings.get("machine_rate_per_hour"))
    labor = float(settings.get("labor_flat"))
    cost = material + electricity + machine + labor
    markup = float(settings.get("markup_multiplier"))
    price = cost * markup

    return {
        "currency": settings.get("currency_symbol"),
        "hours": hours,
        "grams": grams,
        "markup": markup,
        "material": material,
        "electricity": electricity,
        "machine": machine,
        "labor": labor,
        "cost": cost,
        "price": price,
    }


def format_price_lines(p: Optional[dict]) -> List[str]:
    """Render the price block for the completion summary (text channels)."""
    if not p:
        return []
    c = p["currency"]

    def money(x: float) -> str:
        return f"{c}{x:.2f}"

    return [
        "",
        f"💵 Suggested price: {money(p['price'])}",
        f"   cost {money(p['cost'])} ×{p['markup']:g}  "
        f"(mat {money(p['material'])} · power {money(p['electricity'])} · "
        f"machine {money(p['machine'])} · labor {money(p['labor'])})",
        f"   based on {p['grams']:.0f} g over {p['hours']:.1f} h",
    ]
