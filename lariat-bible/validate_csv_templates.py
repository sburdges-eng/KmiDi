"""Validate all CSV templates in data/csv_templates/ for required columns and structure.

This script performs case-insensitive, whitespace-normalized header checks and
offers alias suggestions when headers differ by casing, punctuation, or common
synonyms. It also reports unexpected extra columns and prints a sample row to
help diagnose problems.
"""

import os
import unicodedata
import pandas as pd

TEMPLATE_DIR = "data/csv_templates"

# Canonical templates (expected header names)
TEMPLATES = {
    "vendor_product_comparison_template.csv": [
        "product", "sysco_per_lb", "shamrock_per_lb", "savings_per_lb", "savings_percent",
        "preferred_vendor", "sysco_split_per_lb", "split_vs_shamrock"
    ],
    "spice_comparison_template.csv": [
        "item", "sysco_pack", "sysco_case_price", "sysco_per_lb", "sysco_split_price",
        "sysco_split_per_lb", "shamrock_pack", "shamrock_case_price", "shamrock_per_lb",
        "savings_per_lb", "savings_percent", "preferred_vendor", "monthly_savings_estimate"
    ],
    "invoice_template.csv": [
        "date", "vendor", "invoice_number", "item_code", "item_description", "pack_size",
        "quantity", "unit_price", "total_price", "category"
    ],
    "order_guide_template.csv": [
        "vendor", "category", "item_code", "item_name", "pack_size", "case_price",
        "split_price", "unit_price_per_lb", "notes", "last_updated"
    ],
    "recipe_cost_template.csv": [
        "recipe_name", "ingredient", "quantity", "unit", "cost_per_unit", "total_cost",
        "yield_servings", "cost_per_serving", "category"
    ],
    "inventory_template.csv": [
        "item_name", "category", "current_stock", "unit", "par_level", "reorder_point",
        "preferred_vendor", "last_order_date", "cost_per_unit", "total_value"
    ],
    "catering_event_template.csv": [
        "event_date", "event_name", "client_name", "guest_count", "menu_selection",
        "total_cost", "sale_price", "profit", "profit_margin", "status"
    ],
    "equipment_maintenance_template.csv": [
        "equipment_name", "equipment_type", "location", "last_service_date", "next_service_date",
        "service_interval_days", "service_provider", "last_service_cost", "notes", "status"
    ],
}

# Common aliases: map many common header variants to canonical names
ALIASES = {
    # inventory / item name
    "item": "item_name",
    "name": "item_name",
    "product": "item_name",
    # ids and codes
    "item_code": "item_code",
    "sku": "item_code",
    # quantity / qty
    "qty": "quantity",
    "quantity_": "quantity",
    # price / cost
    "price": "unit_price",
    "unitprice": "unit_price",
    "cost": "cost_per_unit",
    "unit_cost": "cost_per_unit",
    "total": "total_price",
    "total_cost": "total_cost",
    # vendor
    "vendor_name": "vendor",
    # dates
    "date": "date",
    "eventdate": "event_date",
    # generic mappings
    "category_name": "category",
}


def _canonical(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    # Normalize unicode, strip whitespace and lower-case
    s = unicodedata.normalize("NFKC", s).strip().lower()
    # Remove common punctuation used in headers
    s = s.replace(" ", "_").replace("-", "_").replace(".", "")
    # collapse duplicate underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _read_with_fallback(path):
    # Try utf-8, then utf-8-sig (BOM), then latin-1
    errors = []
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception as e:
            errors.append((enc, e))
    # If all fail, raise the last exception
    raise errors[-1][1]


def validate_template(fname, required_cols):
    path = os.path.join(TEMPLATE_DIR, fname)
    try:
        df, used_enc = _read_with_fallback(path)
        raw_cols = list(df.columns)

        # canonicalize column names
        canon_map = {c: _canonical(c) for c in raw_cols}
        inverse_map = {}
        for orig, canon in canon_map.items():
            inverse_map.setdefault(canon, []).append(orig)

        # Build expected canonical set (for exact matching and aliasing)
        expected_canon = { _canonical(c) for c in required_cols }

        found = set()
        alias_suggestions = {}

        # Exact matches by canonicalized header
        for canon, originals in inverse_map.items():
            if canon in expected_canon:
                # If canonicalized header matches an expected canonical name,
                # map it back to the canonical required column(s)
                for req in required_cols:
                    if _canonical(req) == canon:
                        found.add(req)
                        if len(originals) > 1:
                            alias_suggestions[req] = originals

        # For missing required columns, look for aliases
        missing = []
        for req in required_cols:
            if req in found:
                continue
            req_canon = _canonical(req)
            # search aliases dict for possible matches
            # check if any raw header canonical maps to an alias that points to this req
            matched = False
            for raw in raw_cols:
                raw_canon = _canonical(raw)
                # direct alias mapping
                if raw_canon in ALIASES and ALIASES[raw_canon] == req:
                    alias_suggestions[req] = [raw]
                    matched = True
                    found.add(req)
                    break
                # fuzzy canonical equality (e.g., 'item' -> 'item_name')
                if raw_canon == req_canon:
                    found.add(req)
                    matched = True
                    break
            if not matched:
                missing.append(req)

        # Unexpected columns: those that do not map to any expected canonical or alias
        unexpected = []
        for raw in raw_cols:
            raw_canon = _canonical(raw)
            mapped = False
            # if raw canonical is in expected set
            if raw_canon in expected_canon:
                mapped = True
            # if raw canonical maps via ALIASES
            if not mapped and raw_canon in ALIASES:
                mapped = True
            if not mapped:
                unexpected.append(raw)

        # Reporting
        if missing:
            print(f"{fname}: MISSING required columns: {missing} (encoding: {used_enc})")
            if alias_suggestions:
                for req, raws in alias_suggestions.items():
                    print(f"  Suggestion: map {raws} -> {req}")
        else:
            print(f"{fname}: OK (all required columns present) (encoding: {used_enc})")
            if alias_suggestions:
                for req, raws in alias_suggestions.items():
                    print(f"  Note: header(s) {raws} map to {req}")

        if unexpected:
            print(f"  Unexpected columns (will be ignored): {unexpected}")

        # Print a small sample row to help debugging
        try:
            sample = df.dropna(how='all').head(1)
            if not sample.empty:
                print("  Sample row:")
                # show dict of column->value for readability
                print("   ", sample.to_dict(orient='records')[0])
        except Exception:
            pass

    except Exception as e:
        print(f"{fname}: ERROR loading file: {e}")

def main():
    print("Validating CSV templates in:", TEMPLATE_DIR)
    for fname, required_cols in TEMPLATES.items():
        validate_template(fname, required_cols)

if __name__ == "__main__":
    main()
