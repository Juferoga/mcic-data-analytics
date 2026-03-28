#!/usr/bin/env python3
"""Demo: Text-to-Number Encoders in Analitica."""

import pandas as pd
import numpy as np

from analitica.transformers import (
    LabelEncoder,
    OneHotEncoder,
    TargetEncoder,
    HashEncoder,
)


def main():
    print("=" * 60)
    print("Analitica - Text-to-Number Encoders Demo")
    print("=" * 60)

    # Sample data with categorical features
    data = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "city": ["NY", "LA", "NY", "SF", "LA", "SF"],
            "status": ["active", "inactive", "active", "active", "inactive", "active"],
            "segment": ["A", "B", "C", "A", "B", "C"],
            "revenue": [100, 200, 150, 300, 250, 180],
            "purchased": [1, 0, 1, 1, 0, 1],
        }
    )

    print("\n📊 Original Data:")
    print(data.to_string())

    # LabelEncoder
    print("\n" + "-" * 40)
    print("🔄 LabelEncoder (Ordinal encoding)")
    print("-" * 40)
    encoder = LabelEncoder(columns=["status"])
    result = encoder.fit_transform(data[["status"]])
    print(result)

    # OneHotEncoder
    print("\n" + "-" * 40)
    print("🔄 OneHotEncoder (Nominal encoding)")
    print("-" * 40)
    encoder = OneHotEncoder(columns=["city"])
    result = encoder.fit_transform(data[["city"]])
    print(result)

    # TargetEncoder
    print("\n" + "-" * 40)
    print("🔄 TargetEncoder (Supervised encoding)")
    print("-" * 40)
    encoder = TargetEncoder(smoothing=1.0)
    result = encoder.fit_transform(data[["segment"]], data["purchased"])
    print(result)
    print(
        f"\n📈 Purchase rate by segment: A={data[data.segment == 'A'].purchased.mean():.2f}, "
        f"B={data[data.segment == 'B'].purchased.mean():.2f}, "
        f"C={data[data.segment == 'C'].purchased.mean():.2f}"
    )

    # HashEncoder
    print("\n" + "-" * 40)
    print("🔄 HashEncoder (High cardinality)")
    print("-" * 40)
    urls = pd.DataFrame(
        {
            "url": [
                "https://site-a.com/page",
                "https://site-b.com/page",
                "https://site-c.com/page",
                "https://site-a.com/about",
            ]
        }
    )
    encoder = HashEncoder(n_bins=4, n_functions=3)
    result = encoder.fit_transform(urls)
    print(result)

    print("\n" + "=" * 60)
    print("✅ All encoders working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
