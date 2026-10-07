// Host program of the numerics spike (README.md). Runs the kernels of numerics.metal from each library that
// build.sh writes and compares them with the CPU bit for bit. Built with -ffp-contract=off, so a*b + c on the CPU
// is two roundings.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float fbits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint16_t hbits(_Float16 h) { uint16_t u; memcpy(&u, &h, 2); return u; }

// Distance in ulps between two floats; both NaN counts as 0, one NaN as "huge".
static int64_t ord(float f) {
    uint32_t u = bits(f);
    return (u & 0x80000000u) ? -(int64_t)(u & 0x7fffffffu) : (int64_t)u;
}
static int64_t ulps(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return (std::isnan(a) && std::isnan(b)) ? 0 : (int64_t)1 << 40;
    return llabs(ord(a) - ord(b));
}

static id<MTLBuffer> in_buf(const void *p, size_t n) {
    return [g_dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
}
static id<MTLBuffer> out_buf(size_t n) { return [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared]; }

static id<MTLComputePipelineState> pipe(id<MTLLibrary> lib, const char *name, std::string *err = nullptr) {
    id<MTLFunction> f = [lib newFunctionWithName:@(name)];
    if (!f) {
        if (err) *err = "no such function";
        return nil;
    }
    NSError *e = nil;
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:f error:&e];
    if (!p && err) *err = e.localizedDescription.UTF8String;
    return p;
}

// One dispatch of n threads in groups of tg. Returns the GPU time in ms.
static double run(id<MTLComputePipelineState> p, const std::vector<id<MTLBuffer>> &b, NSUInteger n, NSUInteger tg) {
    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    for (NSUInteger i = 0; i < b.size(); ++i) [e setBuffer:b[i] offset:0 atIndex:i];
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "command buffer failed: %s\n", cb.error.localizedDescription.UTF8String);
        exit(1);
    }
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}

static std::vector<float> gpu_unary(id<MTLLibrary> lib, const char *k, const std::vector<float> &a) {
    id<MTLComputePipelineState> p = pipe(lib, k);
    if (!p) return {};
    id<MTLBuffer> A = in_buf(a.data(), a.size() * 4), O = out_buf(a.size() * 4);
    run(p, {A, O}, a.size(), 256);
    const float *o = (const float *)O.contents;
    return std::vector<float>(o, o + a.size());
}
static std::vector<float> gpu_binary(id<MTLLibrary> lib, const char *k, const std::vector<float> &a,
                                     const std::vector<float> &b) {
    id<MTLComputePipelineState> p = pipe(lib, k);
    if (!p) return {};
    id<MTLBuffer> A = in_buf(a.data(), a.size() * 4), B = in_buf(b.data(), b.size() * 4), O = out_buf(a.size() * 4);
    run(p, {A, B, O}, a.size(), 256);
    const float *o = (const float *)O.contents;
    return std::vector<float>(o, o + a.size());
}

// Bitwise and ulp comparison against a CPU reference.
static void report(const char *what, const std::vector<float> &g, const std::vector<float> &a,
                   const std::function<float(size_t)> &ref) {
    if (g.empty()) {
        printf("  %-34s (not in this library)\n", what);
        return;
    }
    size_t diff = 0;
    int64_t worst = 0;
    size_t worst_i = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        float r = ref(i);
        if (bits(g[i]) != bits(r) && !(std::isnan(g[i]) && std::isnan(r))) {
            ++diff;
            int64_t u = ulps(g[i], r);
            if (u > worst) worst = u, worst_i = i;
        }
    }
    printf("  %-34s %10zu of %zu differ (%.4f%%), max %lld ulp", what, diff, g.size(), 100.0 * diff / g.size(),
           (long long)worst);
    if (worst) printf(" at x = %.9g (gpu %.9g, ref %.9g)", a[worst_i], g[worst_i], ref(worst_i));
    printf("\n");
}

static std::mt19937_64 g_rng(20261003);
static float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(g_rng); }
// sign * 2^e * (1 + u), e uniform in [elo, ehi]: covers many magnitudes.
static float spread(int elo, int ehi, bool pos = false) {
    int e = std::uniform_int_distribution<int>(elo, ehi)(g_rng);
    float m = 1.0f + uni(0.0f, 1.0f);
    float s = (pos || (g_rng() & 1)) ? 1.0f : -1.0f;
    return s * std::ldexp(m, e);
}

static void numeric_tests(id<MTLLibrary> lib) {
    // 1. Contraction of a*b + c.
    {
        const size_t n = 1 << 22;
        std::vector<float> a(n), b(n), c(n);
        for (size_t i = 0; i < n; ++i) a[i] = uni(-1, 1), b[i] = uni(-1, 1), c[i] = uni(-1, 1);
        std::vector<size_t> decisive;
        for (size_t i = 0; i < n; ++i) {
            float p = a[i] * b[i];
            float unf = p + c[i];
            if (bits(unf) != bits(std::fmaf(a[i], b[i], c[i]))) decisive.push_back(i);
        }
        for (const char *k : {"k_mad", "k_fma", "k_mad_pragma_off"}) {
            id<MTLComputePipelineState> p = pipe(lib, k);
            id<MTLBuffer> A = in_buf(a.data(), n * 4), B = in_buf(b.data(), n * 4), C = in_buf(c.data(), n * 4),
                          O = out_buf(n * 4);
            run(p, {A, B, C, O}, n, 256);
            const float *o = (const float *)O.contents;
            size_t as_unfused = 0, as_fused = 0;
            for (size_t i : decisive) {
                float p2 = a[i] * b[i];
                if (bits(o[i]) == bits(p2 + c[i])) ++as_unfused;
                if (bits(o[i]) == bits(std::fmaf(a[i], b[i], c[i]))) ++as_fused;
            }
            printf("  %-34s on %zu inputs where fused != unfused: %zu equal unfused, %zu equal fma\n", k,
                   decisive.size(), as_unfused, as_fused);
        }
    }
    // 2. Division and reciprocal, correctly rounded on the CPU.
    {
        const size_t n = 1 << 24;
        std::vector<float> x(n), y(n), three(n, 3.0f);
        for (size_t i = 0; i < n; ++i) x[i] = spread(-20, 20), y[i] = spread(-20, 20);
        report("x / y", gpu_binary(lib, "k_div", x, y), x, [&](size_t i) { return x[i] / y[i]; });
        report("precise::divide(x, y)", gpu_binary(lib, "k_pdiv", x, y), x, [&](size_t i) { return x[i] / y[i]; });
        report("1.0f / x", gpu_unary(lib, "k_recip", x), x, [&](size_t i) { return 1.0f / x[i]; });
        report("fast::divide(x, 3)", gpu_unary(lib, "k_fdiv", x), x, [&](size_t i) { return x[i] / 3.0f; });
        // x / 127, the quantizer's d = amax / 127 and x / d, over typical activation magnitudes
        std::vector<float> am(n), d127(n, 127.0f);
        for (size_t i = 0; i < n; ++i) am[i] = spread(-12, 8, true);
        report("amax / 127", gpu_binary(lib, "k_div", am, d127), am, [&](size_t i) { return am[i] / 127.0f; });
    }
    // 3. sqrt, rsqrt.
    {
        const size_t n = 1 << 22;
        std::vector<float> x(n);
        for (size_t i = 0; i < n; ++i) x[i] = spread(-40, 40, true);
        report("sqrt(x)", gpu_unary(lib, "k_sqrt", x), x, [&](size_t i) { return std::sqrt(x[i]); });
        report("rsqrt(x) vs 1/sqrt in double", gpu_unary(lib, "k_rsqrt", x), x,
               [&](size_t i) { return (float)(1.0 / std::sqrt((double)x[i])); });
    }
    // 4. exp, log1p, sigmoid against double references.
    {
        const size_t n = 1 << 22;
        std::vector<float> x(n), l(n), s(n);
        for (size_t i = 0; i < n; ++i) x[i] = uni(-87, 88), l[i] = uni(-0.999f, 100.0f), s[i] = uni(-30, 30);
        auto rexp = [&](size_t i) { return (float)std::exp((double)x[i]); };
        report("exp(x) vs double", gpu_unary(lib, "k_exp", x), x, rexp);
        report("precise::exp(x) vs double", gpu_unary(lib, "k_pexp", x), x, rexp);
        report("fast::exp(x) vs double", gpu_unary(lib, "k_fexp", x), x, rexp);
        std::vector<float> cexp(n), clog1p(n);
        for (size_t i = 0; i < n; ++i) cexp[i] = std::exp(x[i]), clog1p[i] = std::log1p(l[i]);
        report("CPU expf(x) vs double", cexp, x, rexp);
        auto rlog1p = [&](size_t i) { return (float)std::log1p((double)l[i]); };
        report("log(1+x) vs double", gpu_unary(lib, "k_log1p_naive", l), l, rlog1p);
        report("log1p_k(x) vs double", gpu_unary(lib, "k_log1p", l), l, rlog1p);
        report("CPU log1pf(x) vs double", clog1p, l, rlog1p);
        // softplus over the GDN's range of a*dt + bias: compare with upstream's CPU form and with double
        std::vector<float> sp(n);
        for (size_t i = 0; i < n; ++i) sp[i] = uni(-30, 30);
        std::vector<float> gsp = gpu_unary(lib, "k_softplus", sp);
        report("softplus vs CPU log1pf(expf)", gsp, sp,
               [&](size_t i) { return sp[i] > 20.0f ? sp[i] : std::log1p(std::exp(sp[i])); });
        report("softplus vs double", gsp, sp, [&](size_t i) {
            return sp[i] > 20.0f ? sp[i] : (float)std::log1p(std::exp((double)sp[i]));
        });
        std::vector<float> sg = gpu_unary(lib, "k_sigmoid", s);
        report("1/(1+exp(-x)) vs double", sg, s, [&](size_t i) { return (float)(1.0 / (1.0 + std::exp(-(double)s[i]))); });
        report("1/(1+exp(-x)) vs CPU float", sg, s, [&](size_t i) { return 1.0f / (1.0f + std::exp(-s[i])); });
    }
    // 5. Rounding functions at ties.
    {
        std::vector<float> x;
        for (int k = -2000; k <= 2000; ++k) x.push_back(k + 0.5f), x.push_back(k + 0.25f), x.push_back((float)k);
        for (int i = 0; i < (1 << 20); ++i) x.push_back(uni(-1e6f, 1e6f));
        // values where trunc(x + copysign(0.5, x)) or floor(x + 0.5) would be wrong, and signed zeros
        const float edge[] = {0.49999997f, -0.49999997f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 8388609.0f,
                              -8388609.0f, 8388608.5f, -0.25f, -0.0f, 0.0f, -1e-30f, 1e-30f};
        const size_t nedge = sizeof edge / sizeof edge[0];
        for (float e : edge) x.push_back(e);
        report("rint(x) vs rintf", gpu_unary(lib, "k_rint", x), x, [&](size_t i) { return std::rint(x[i]); });
        report("round(x) vs roundf", gpu_unary(lib, "k_round", x), x, [&](size_t i) { return std::round(x[i]); });
        report("trunc(x) vs truncf", gpu_unary(lib, "k_trunc", x), x, [&](size_t i) { return std::trunc(x[i]); });
        std::vector<float> rd = gpu_unary(lib, "k_round", x), ri = gpu_unary(lib, "k_rint", x);
        printf("  %-34s", "round / rint of the edge values");
        for (size_t i = x.size() - nedge; i < x.size(); ++i)
            printf(" %.9g->%g/%g%s", x[i], rd[i], ri[i],
                   bits(rd[i]) == bits(std::round(x[i])) && bits(ri[i]) == bits(std::rint(x[i])) ? "" : "(!)");
        printf("\n");
    }
    // 6. float -> half: every half midpoint and its neighbours, random values, and the edges.
    {
        std::vector<float> x;
        for (uint32_t h = 0; h < 0x7c00; ++h) {  // finite positive halves
            _Float16 lo, hi;
            uint16_t hl = (uint16_t)h, hh = (uint16_t)(h + 1);
            memcpy(&lo, &hl, 2), memcpy(&hi, &hh, 2);
            double mid = ((double)lo + (double)hi) / 2;  // exact in float: halves have 11 significant bits
            float m = (float)mid;
            for (float v : {m, std::nextafter(m, 0.0f), std::nextafter(m, 1e30f), (float)lo})
                x.push_back(v), x.push_back(-v);
        }
        for (int i = 0; i < (1 << 22); ++i) x.push_back(spread(-27, 16));
        for (uint32_t u : {0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0x7f812345u,
                           0xffc00001u, 0x477fe000u /*65504*/, 0x477fefffu, 0x477ff000u /*65520*/, 0x47800000u,
                           0x4e6e6b28u /*1e9*/, 0x33000000u /*2^-25*/, 0x33000001u, 0x33800000u /*2^-24*/,
                           0x38800000u /*2^-14*/, 0x387fc000u, 0x00000001u /*f32 denormal*/})
            x.push_back(fbits(u));
        id<MTLComputePipelineState> p = pipe(lib, "k_half");
        id<MTLBuffer> A = in_buf(x.data(), x.size() * 4), O = out_buf(x.size() * 2);
        run(p, {A, O}, x.size(), 256);
        const uint16_t *o = (const uint16_t *)O.contents;
        size_t diff = 0, nan_diff = 0;
        uint32_t first = 0;
        bool have_first = false;
        for (size_t i = 0; i < x.size(); ++i) {
            uint16_t r = hbits((_Float16)x[i]);
            if (o[i] != r) {
                bool nan = std::isnan(x[i]);
                (nan ? nan_diff : diff)++;
                if (!nan && !have_first) first = bits(x[i]), have_first = true;
            }
        }
        printf("  %-34s %zu non-NaN and %zu NaN of %zu differ from the CPU's RNE conversion", "half(x)", diff,
               nan_diff, x.size());
        if (have_first) printf(" (first: 0x%08x)", first);
        printf("\n");
        size_t base = x.size() - 19;
        printf("  %-34s", "half() of the edge values");
        for (size_t i = base; i < x.size(); ++i) printf(" %08x->%04x", bits(x[i]), o[i]);
        printf("\n");
    }
}

// simd_sum against an explicit xor butterfly, a sequential sum and a pairwise tree.
static void simd_tests(id<MTLLibrary> lib) {
    const size_t n = 1 << 20;
    std::vector<float> a(n);
    for (size_t i = 0; i < n; ++i) a[i] = uni(-1, 1) * std::ldexp(1.0f, std::uniform_int_distribution<int>(-10, 10)(g_rng));
    id<MTLComputePipelineState> p = pipe(lib, "k_simd");
    id<MTLBuffer> A = in_buf(a.data(), n * 4), O = out_buf(n * 16);
    run(p, {A, O}, n, 1024);
    const float *o = (const float *)O.contents;
    size_t groups = n / 32, bfly_ok = 0, sum_is_bfly = 0, sum_is_seq = 0, sum_is_tree = 0, sum_uniform = 0, max_ok = 0,
           sum_is_up = 0;
    for (size_t g = 0; g < groups; ++g) {
        const float *v = &a[g * 32];
        float bf[32], tr[32];
        for (int l = 0; l < 32; ++l) bf[l] = v[l], tr[l] = v[l];
        for (int m = 16; m > 0; m >>= 1) {
            float nb[32];
            for (int l = 0; l < 32; ++l) nb[l] = bf[l] + bf[l ^ m];
            memcpy(bf, nb, sizeof bf);
        }
        float up[32];
        for (int l = 0; l < 32; ++l) up[l] = v[l];
        for (int m = 1; m < 32; m <<= 1) {
            float nb[32];
            for (int l = 0; l < 32; ++l) nb[l] = up[l] + up[l ^ m];
            memcpy(up, nb, sizeof up);
        }
        bool lanes_up = true;
        for (int l = 0; l < 32; ++l) lanes_up &= bits(o[4 * (g * 32 + l)]) == bits(up[l]);
        sum_is_up += lanes_up;
        float seq = 0;
        for (int l = 0; l < 32; ++l) seq += v[l];
        for (int w = 32; w > 1; w /= 2)
            for (int l = 0; l < w / 2; ++l) tr[l] = tr[2 * l] + tr[2 * l + 1];
        bool lanes_bfly = true, uni_sum = true, lanes_max = true;
        float mx = v[0];
        for (int l = 1; l < 32; ++l) mx = std::max(mx, v[l]);
        for (int l = 0; l < 32; ++l) {
            const float *r = &o[4 * (g * 32 + l)];
            lanes_bfly &= bits(r[1]) == bits(bf[l]);
            uni_sum &= bits(r[0]) == bits(o[4 * g * 32]);
            lanes_max &= bits(r[2]) == bits(mx) && bits(r[3]) == bits(mx);
        }
        float s = o[4 * g * 32];
        bfly_ok += lanes_bfly, sum_uniform += uni_sum, max_ok += lanes_max;
        sum_is_bfly += bits(s) == bits(bf[0]);
        sum_is_seq += bits(s) == bits(seq);
        sum_is_tree += bits(s) == bits(tr[0]);
    }
    printf("  of %zu simdgroups: the xor butterfly (16..1) equals the CPU one in %zu; simd_sum is the same in every lane"
           " in %zu, equals the 16..1 butterfly in %zu, a sequential sum in %zu, a pairwise tree in %zu, the 1..16"
           " butterfly in every lane in %zu; simd_max and the max butterfly equal the CPU max in %zu\n",
           groups, bfly_ok, sum_uniform, sum_is_bfly, sum_is_seq, sum_is_tree, sum_is_up, max_ok);
}

static void device_tests(id<MTLLibrary> lib, NSString *dir) {
    printf("device: %s; maxThreadgroupMemoryLength %lu; maxThreadsPerThreadgroup %lu\n", g_dev.name.UTF8String,
           (unsigned long)g_dev.maxThreadgroupMemoryLength, (unsigned long)g_dev.maxThreadsPerThreadgroup.width);
    {
        id<MTLComputePipelineState> p = pipe(lib, "k_ids");
        const size_t n = 4096;
        id<MTLBuffer> O = out_buf(n * 20);
        run(p, {O}, n, 1024);
        const uint32_t *o = (const uint32_t *)O.contents;
        size_t ok = 0;
        for (size_t i = 0; i < n; ++i)
            ok += o[5 * i] == i % 32 && o[5 * i + 1] == (i % 1024) / 32 && o[5 * i + 2] == 32 &&
                  o[5 * i + 3] == 0xAAAAAAAAu && o[5 * i + 4] == 0;
        printf("k_ids: threadExecutionWidth %lu, maxTotalThreadsPerThreadgroup %lu; %zu of %zu threads have lane = i %% 32,"
               " simdgroup = (i %% 1024) / 32, width 32, ballot low word 0xAAAAAAAA and high word 0\n",
               (unsigned long)p.threadExecutionWidth, (unsigned long)p.maxTotalThreadsPerThreadgroup, ok, n);
    }
    {
        std::string err;
        id<MTLComputePipelineState> p = pipe(lib, "k_tg32k", &err);
        if (!p) {
            printf("k_tg32k (32,768 B): pipeline refused: %s\n", err.c_str());
        } else {
            id<MTLBuffer> O = out_buf(1024 * 4);
            run(p, {O}, 1024, 1024);
            const float *o = (const float *)O.contents;
            size_t ok = 0;
            for (int i = 0; i < 1024; ++i) ok += o[i] == 8191 - i;
            printf("k_tg32k (32,768 B): staticThreadgroupMemoryLength %lu, maxTotalThreadsPerThreadgroup %lu, %zu of 1024"
                   " correct\n",
                   (unsigned long)p.staticThreadgroupMemoryLength, (unsigned long)p.maxTotalThreadsPerThreadgroup, ok);
        }
        NSString *over = [dir stringByAppendingPathComponent:@"tg_over.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:over]) {
            NSError *e = nil;
            id<MTLLibrary> ol = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:over] error:&e];
            std::string err2;
            id<MTLComputePipelineState> q = ol ? pipe(ol, "k_tg_over", &err2) : nil;
            if (q)
                printf("k_tg_over (32,784 B): pipeline created, staticThreadgroupMemoryLength %lu\n",
                       (unsigned long)q.staticThreadgroupMemoryLength);
            else
                printf("k_tg_over (32,784 B): pipeline refused: %s\n", ol ? err2.c_str() : e.localizedDescription.UTF8String);
        }
    }
}

static std::vector<uint64_t> grid(int n) {
    std::vector<uint64_t> v;
    uint64_t x = 1;
    for (int i = 0; i < n; ++i) {
        x += 0x9E3779B97F4A7C15ull;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        v.push_back(z ^ (z >> 31));
    }
    return v;
}

// CPU results of the first 256 threads of the cb and ch kernels for one table.
template <class T> static void cb_want(const std::vector<T> &g, std::vector<uint64_t> &cb, std::vector<uint64_t> &ch) {
    const uint32_t mask = (uint32_t)g.size() - 1, lookups = 2048;
    cb.assign(256, 0), ch.assign(256, 0);
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t s = i * 2654435761u + 1u;
        uint64_t acc = 0;
        for (uint32_t k = 0; k < lookups; ++k) {
            s ^= s << 13, s ^= s >> 17, s ^= s << 5;
            acc += (uint64_t)(int64_t)g[s & mask];
        }
        cb[i] = acc;
        uint32_t c[4];
        for (int j = 0; j < 4; ++j) c[j] = i * 2654435761u + 1u + j;
        for (uint32_t k = 0; k < lookups / 4; ++k)
            for (int j = 0; j < 4; ++j) c[j] = (c[j] ^ (uint32_t)(g[c[j] & mask] >> 7)) + k;
        ch[i] = (uint64_t)c[0] + c[1] + c[2] + c[3];
    }
}

static void codebook_bench(id<MTLLibrary> lib) {
    const size_t n = 1 << 20;
    const int lookups = 2048;
    auto time = [&](const char *k, id<MTLBuffer> G, const uint64_t *want) {
        id<MTLComputePipelineState> p = pipe(lib, k);
        id<MTLBuffer> O = out_buf(n * 8);
        double best = 1e30;
        for (int r = 0; r < 6; ++r) best = std::min(best, run(p, {O, G}, n, 256));
        bool ok = !want || memcmp(O.contents, want, 256 * 8) == 0;
        printf("  %-22s %7.3f ms = %6.1f G lookups/s%s\n", k, best, (double)n * lookups / (best * 1e-3) / 1e9,
               ok ? "" : "  WRONG RESULT");
    };
    std::vector<uint64_t> g1k = grid(1024), g512(g1k.begin(), g1k.begin() + 512);
    std::vector<int8_t> kv = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    std::vector<uint64_t> cb, ch;
    time("k_cb_none", in_buf(g512.data(), 8), nullptr);
    struct Set { const char *suf; id<MTLBuffer> buf; };
    cb_want(kv, cb, ch);
    std::vector<uint64_t> cb16 = cb, ch16 = ch;
    cb_want(g512, cb, ch);
    std::vector<uint64_t> cb512 = cb, ch512 = ch;
    cb_want(g1k, cb, ch);
    const std::vector<uint64_t> *wants[3][2] = {{&cb16, &ch16}, {&cb512, &ch512}, {&cb, &ch}};
    Set sets[3] = {{"_16", in_buf(kv.data(), kv.size())}, {"_512", in_buf(g512.data(), 512 * 8)},
                   {"_1k", in_buf(g1k.data(), 1024 * 8)}};
    for (int t = 0; t < 3; ++t)
        for (int c = 0; c < 2; ++c)
            for (const char *space : {"program", "constant", "device", "threadgroup"}) {
                std::string k = std::string(c ? "k_ch_" : "k_cb_") + space + sets[t].suf;
                time(k.c_str(), sets[t].buf, wants[t][c]->data());
            }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        NSString *dir = argc > 1 ? @(argv[1]) : @"out";
        g_dev = MTLCreateSystemDefaultDevice();
        g_q = [g_dev newCommandQueue];
        for (const char *v : {"safe", "safe_contract", "fast"}) {
            NSString *path = [dir stringByAppendingPathComponent:[NSString stringWithFormat:@"%s.metallib", v]];
            NSError *e = nil;
            id<MTLLibrary> lib = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&e];
            if (!lib) {
                fprintf(stderr, "%s: %s\n", path.UTF8String, e.localizedDescription.UTF8String);
                return 1;
            }
            if (!strcmp(v, "safe")) device_tests(lib, dir);
            printf("\n== %s.metallib\n", v);
            numeric_tests(lib);
            simd_tests(lib);
            if (!strcmp(v, "safe")) {
                printf("  codebook lookups, 2^20 threads x 2048 lookups, best of 6:\n");
                codebook_bench(lib);
            }
        }
    }
    return 0;
}
