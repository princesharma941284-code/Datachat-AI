"""Fixed sample database used by the accuracy benchmark.

The data is generated with a fixed random seed, so every run (and every
teammate's laptop, and GitHub Actions) sees exactly the same rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42

CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Chennai",
    "Kolkata", "Hyderabad", "Pune", "Jaipur",
]
SEGMENTS = ["Consumer", "Corporate", "Small Business"]
STATUSES = ["Delivered", "Shipped", "Cancelled", "Returned"]

FIRST_NAMES = ["Aarav", "Diya", "Rohan", "Sneha", "Kabir", "Meera", "Vikram", "Ananya"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Patel"]

PRODUCTS = [
    ("Laptop", "Electronics", 55000), ("Smartphone", "Electronics", 22000),
    ("Headphones", "Electronics", 3500), ("Smart Watch", "Electronics", 8000),
    ("Bluetooth Speaker", "Electronics", 4500),
    ("Office Chair", "Furniture", 7500), ("Study Desk", "Furniture", 9500),
    ("Bookshelf", "Furniture", 6200), ("Sofa Set", "Furniture", 32000),
    ("Table Lamp", "Furniture", 1500),
    ("Notebook Pack", "Stationery", 250), ("Ball Pen Box", "Stationery", 120),
    ("Marker Set", "Stationery", 450), ("Desk Organizer", "Stationery", 650),
    ("Planner Diary", "Stationery", 380),
    ("Denim Jacket", "Clothing", 3200), ("Running Shoes", "Clothing", 4100),
    ("Cotton T-Shirt", "Clothing", 700), ("Formal Shirt", "Clothing", 1800),
    ("Winter Hoodie", "Clothing", 2600),
]


def build_sample_tables(n_customers: int = 40, n_orders: int = 600) -> dict:
    """Return {"customers": df, "products": df, "orders": df}."""
    rng = np.random.default_rng(SEED)

    customers = pd.DataFrame(
        {
            "customer_id": range(1, n_customers + 1),
            "customer_name": [
                f"{FIRST_NAMES[i % len(FIRST_NAMES)]} {LAST_NAMES[i // len(FIRST_NAMES)]}"
                for i in range(n_customers)
            ],
            "city": rng.choice(CITIES, n_customers),
            "segment": rng.choice(SEGMENTS, n_customers),
            "signup_date": pd.to_datetime("2024-06-01")
            + pd.to_timedelta(rng.integers(0, 580, n_customers), unit="D"),
        }
    )

    products = pd.DataFrame(
        {
            "product_id": range(1, len(PRODUCTS) + 1),
            "product_name": [p[0] for p in PRODUCTS],
            "category": [p[1] for p in PRODUCTS],
            "unit_price": [float(p[2]) for p in PRODUCTS],
        }
    )

    product_ids = rng.integers(1, len(PRODUCTS) + 1, n_orders)
    quantity = rng.choice([1, 2, 3, 4, 5, 6], n_orders, p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05])
    discount = rng.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], n_orders)
    unit_price = products.set_index("product_id").loc[product_ids, "unit_price"].to_numpy()

    orders = pd.DataFrame(
        {
            "order_id": range(1001, 1001 + n_orders),
            "customer_id": rng.integers(1, n_customers + 1, n_orders),
            "product_id": product_ids,
            "order_date": pd.to_datetime("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 638, n_orders), unit="D"),
            "quantity": quantity,
            "discount": discount,
            "status": rng.choice(STATUSES, n_orders, p=[0.7, 0.15, 0.08, 0.07]),
            "total_amount": np.round(quantity * unit_price * (1 - discount), 2),
        }
    )

    return {"customers": customers, "products": products, "orders": orders}


# What a user would type into the semantic layer for this data (Day 2).
# It is applied only in "semantic" mode of the benchmark.
SAMPLE_DEFINITIONS = {
    "tables": {
        "orders": {
            "description": "One row per customer order.",
            "columns": {
                "total_amount": {
                    "description": "Order value after discount: quantity x unit_price x (1 - discount).",
                    "synonyms": ["revenue", "sales", "order amount"],
                },
                "discount": {"description": "Fraction between 0 and 1 (0.2 means 20 percent)."},
                "status": {"description": "Delivery state of the order."},
            },
        },
        "customers": {"description": "One row per customer."},
        "products": {"description": "One row per product in the catalogue."},
    },
    "metrics": [
        {
            "name": "revenue",
            "table": "orders",
            "expression": "SUM(total_amount)",
            "synonyms": ["sales", "income"],
            "description": "Total order value.",
        },
        {
            "name": "average order value",
            "table": "orders",
            "expression": "AVG(total_amount)",
            "synonyms": ["average order amount", "AOV"],
            "description": "Average value of one order.",
        },
    ],
}
