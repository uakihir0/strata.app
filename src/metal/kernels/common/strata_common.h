// src/metal/kernels/common/strata_common.h - helpers shared by the kernels; static inline only, no kernels.
#pragma once
#include <metal_stdlib>

static inline float strata_twice_plus_one(float x) { return x * 2.0f + 1.0f; }
