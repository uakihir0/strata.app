# Metal binding and ordering checks

One Objective-C++ program, `binding.mm`, with its shaders embedded as a string. It checks five things that
[04-metal-kernels.md](../../04-metal-kernels.md), sections 4 and 8, rely on. All buffers are
`StorageModeShared | HazardTrackingModeUntracked`, as in the engine ([05](../../05-memory-and-residency.md),
section 7).

| Test | What it checks |
| --- | --- |
| 1 `addr` | A kernel reads weights through a table of `uint64` GPU addresses (`buffer.gpuAddress + offset`, offsets 32-B aligned, declared in MSL as `struct { device const uchar* role[3]; }`). The weight buffer is only in a residency set attached to the queue: no `setBuffer`, no `useResource`. It loads a float, a half and a uint at offsets 0, 4 and 8. |
| 2 `offset` | A `device const half*` argument bound with `setBuffer:offset:` at offset 1184 (32-B aligned, not 256-B aligned) |
| 3 `order` | 64 dependent dispatches (ping-pong `y = fma(x, 1.0001, 1)` over 1M floats) in one encoder: serial dispatch type; concurrent with `memoryBarrierWithScope:MTLBarrierScopeBuffers` after each dispatch; concurrent with no barrier. Compared bitwise with the CPU. |
| 4 `fence` | Two command buffers on one queue: the second reads what the first wrote. With and without an `MTLFence` (`updateFence` / `waitForFence`). 20 runs each. |
| 5 `cost` | GPU time (`GPUEndTime - GPUStartTime`) of 1,000 small dependent dispatches (64 threadgroups of 256) under each policy, and of 500 steps of two independent half-size dispatches |

## Build and run

```bash
clang++ -std=c++17 -O2 -fobjc-arc binding.mm -framework Metal -framework Foundation -o binding
./binding
```

## Results (2026-10-03, M5 Pro, macOS 27.0, Xcode 27.0)

| Test | Result |
| --- | --- |
| 1 | All 4,096 entries correct |
| 2 | Correct |
| 3 | Exact for all three policies, three runs each. With dispatches this large the unordered run happened to come out right; test 5 shows the race. |
| 4 | With the fence: 0 of 20 runs wrong. **Without the fence: 20 of 20 wrong**: the second command buffer starts before the first finishes when the buffers are Untracked. |
| 5 | Per dispatch: serial 0.87 µs; concurrent + barrier after each 1.11-1.14 µs; concurrent with no barrier 0.40-0.54 µs (and a wrong result in one of three runs). Two independent halves per step: serial 0.85 ms per 1,000 dispatches, concurrent with one barrier per pair 0.76-0.78 ms. With 1M-float dispatches all policies cost the same (3.25-3.46 ms per 200). |

`MTLCommandBuffer.h` documents `MTLDispatchTypeSerial` as "Command encoder dispatches are executed in dispatched
order", and `MTLResource.h` says that for Untracked resources "command encoder dependencies for this resource are
tracked manually with MTLFence". Tests 3-5 agree with both statements.
