#!/usr/bin/env python3
"""Fill derived peak throughput in GPU profiles from per-SM architecture rates.

Published values (listed under ``throughput_published``) are never changed.
Every other throughput field is recomputed from ``fakegpu/data/throughput_rates.json``
and listed under ``throughput_derived``; fields without a rate, SM count or boost
clock are removed so stale derivations cannot survive a profile correction.
"""

from __future__ import annotations

import argparse

from profile_yaml import read_block, write_block

# Importing profile_yaml has put the repository root on sys.path.
from fakegpu.profile_catalog import THROUGHPUT_FIELDS, load_profiles, profile_directory
from fakegpu.throughput import derive_throughput


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if any profile is stale.")
    args = parser.parse_args()
    root = profile_directory()
    stale = []
    for profile in load_profiles().values():
        path = root / profile.profile_path
        scalars, _ = read_block(path)
        basis = {key: value for key, value in profile.throughput_basis.items() if value == "published"}
        values: dict[str, object] = {key: value for key, value in scalars.items() if key not in THROUGHPUT_FIELDS}
        values.update({key: profile.throughput[key] for key in basis})
        for key, value in derive_throughput(profile).items():
            if key not in basis:
                values[key] = value
                basis[key] = "derived"
        if args.check:
            current = dict(profile.throughput)
            expected = {key: float(values[key]) for key in THROUGHPUT_FIELDS if key in values}
            if current != expected or profile.throughput_basis != basis:
                stale.append(profile.id)
        elif write_block(path, values, basis):
            stale.append(profile.id)
    if args.check:
        if stale:
            print(f"{len(stale)} profiles have stale derived throughput, e.g. {', '.join(stale[:5])}; "
                  "run python3 scripts/derive_profile_throughput.py")
            return 1
        print("Derived throughput matches the architecture rate table")
        return 0
    print(f"Updated derived throughput in {len(stale)} profiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
