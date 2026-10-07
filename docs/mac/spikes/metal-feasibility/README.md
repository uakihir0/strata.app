# Metal feasibility micro-benchmarks

Five small Objective-C++ programs that measured the numbers in
[appendix-a-measurements.md](../../appendix-a-measurements.md). Each one is self-contained: one `.mm` file,
the shader source embedded as a string, no project files.

| File | What it measures |
| --- | --- |
| `bw.mm` | Device limits (`recommendedMaxWorkingSetSize`, `maxBufferLength`, GPU families), GPU streaming read bandwidth of a shared buffer, GPU gather of expert-sized (1.31 MiB) blocks, CPU read bandwidth, and GPU reads of an mmap'd file wrapped with `newBufferWithBytesNoCopy`. |
| `ssd.mm` | Cold SSD random reads with `F_NOCACHE` (1.31 MiB and 16 KiB blocks, 1-32 threads), and whether a GPU touch of 1 MiB of a cold no-copy mmap buffer makes the whole buffer resident. |
| `conc.mm` | GPU-alone read bandwidth vs GPU + CPU reading at the same time (does a CPU expert path add bandwidth on unified memory?), the GPU cost of dependent dispatches in one encoder, and the command-buffer round trip. |
| `mapflags.mm` | Which file mappings `newBufferWithBytesNoCopy` accepts (`MAP_PRIVATE`, `MAP_SHARED`, windows at an offset, unaligned pointers and lengths), checked by comparing a GPU XOR of the words with a CPU XOR. Writes and deletes a 256 MiB file in `/tmp`. |
| `mm.mm` | Matrix-multiply throughput with Metal Performance Primitives tensor ops (`mpp::tensor_ops::matmul2d`): `mm` (half, 64x32 tile), `mm64` (half, 64x64 tile), `mmi8` (int8 -> int32, throughput only). |

## Build and run

Requires macOS 27 and Xcode 27 (Metal Shading Language 4.0+ for `mm.mm`).

```bash
# in this folder
for f in bw ssd conc mm mapflags; do clang++ -std=c++17 -O3 -fobjc-arc $f.mm -framework Metal -framework Foundation -o $f; done

dd if=/dev/urandom of=/tmp/blob4g.bin bs=16m count=256   # a 4 GiB file for bw's mmap test
./bw /tmp/blob4g.bin
./ssd        # writes two 8 GiB files to /tmp/strata-spike/ (create the folder first); delete them afterwards
./conc
./mm; K=mm64 ./mm; K=mmi8 ./mm
./mapflags
```

`ssd` writes its test files with `F_NOCACHE` so that they are not in the page cache when it reads them back, so
the reads come from the SSD. Delete `/tmp/strata-spike/cold*.bin` and `/tmp/blob4g.bin` after the run (24 GiB).

`mmi8` prints a wrong check value on purpose: it reinterprets the half-precision test data as int8, so only its
throughput number is meaningful.
