#!/usr/bin/env python3
"""S05 multi-view control with equal 8/8/8 per-view observation exposure."""
from train_simple4d_core_mv_thickslice_v1 import main

if __name__ == "__main__":
    main(equal_budget=True)
