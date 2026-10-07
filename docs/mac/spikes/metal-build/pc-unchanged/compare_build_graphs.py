"""Compare the build graphs of two configured CMake trees (docs/mac/08-build-and-distribution.md, section 3.4).

    python3 compare_build_graphs.py UPSTREAM_BUILD PROJECT_BUILD

Both trees must be configured with the same options, Ninja and CMAKE_EXPORT_COMPILE_COMMANDS=ON.  The source and
build roots that CMake recorded in each CMakeCache.txt become SRC and BUILD (in both spellings when a symlink is on
the way: /tmp is /private/tmp on macOS), then the compile commands and the whole build.ninja must be equal.  Exit 0
when they are, 1 with the differences when not.
"""
import difflib
import json
import os
import sys


def roots(build):
    cache = {}
    with open(os.path.join(build, "CMakeCache.txt"), encoding="utf-8") as f:
        for line in f:
            name, _, value = line.rstrip("\n").partition("=")
            cache[name.split(":")[0]] = value
    pairs = []
    for key, name in (("CMAKE_HOME_DIRECTORY", "SRC"), ("CMAKE_CACHEFILE_DIR", "BUILD")):
        for p in {cache[key], os.path.realpath(cache[key])}:
            pairs.append((p, name))
    return sorted(pairs, key=lambda x: -len(x[0]))   # longest first: the build folder may sit inside the source


def norm(text, pairs):
    for p, name in pairs:
        text = text.replace(p, name)
    return text


def commands(build, pairs):
    with open(os.path.join(build, "compile_commands.json"), encoding="utf-8") as f:
        entries = json.load(f)
    out = {}
    for e in entries:
        key = (norm(e["file"], pairs), norm(e.get("output", ""), pairs))
        out[key] = norm(e.get("command") or " ".join(e["arguments"]), pairs)
    return out


def main(ub, fb):
    up, fp = roots(ub), roots(fb)
    problems = []
    cu, cf = commands(ub, up), commands(fb, fp)
    for k in sorted(set(cu) - set(cf)):
        problems.append(f"only upstream compiles {k[0]} -> {k[1]}")
    for k in sorted(set(cf) - set(cu)):
        problems.append(f"only the project compiles {k[0]} -> {k[1]}")
    for k in sorted(set(cu) & set(cf)):
        if cu[k] != cf[k]:
            problems.append(f"{k[0]}: upstream compiles it with\n  {cu[k]}\nthe project with\n  {cf[k]}")
    with open(os.path.join(ub, "build.ninja"), encoding="utf-8") as f:
        nu = norm(f.read(), up).splitlines()
    with open(os.path.join(fb, "build.ninja"), encoding="utf-8") as f:
        nf = norm(f.read(), fp).splitlines()
    diff = list(difflib.unified_diff(nu, nf, "upstream/build.ninja", "project/build.ninja", lineterm="", n=1))
    if diff:
        problems.append("build.ninja differs:\n" + "\n".join(diff[:200]))
    if problems:
        print("\n".join(problems))
        return 1
    print(f"same build: {len(cu)} compile commands, {len(nu)} lines of build.ninja")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
