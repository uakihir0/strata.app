# Project origin and source provenance

This project is maintained in its own repository rather than as a GitHub fork. The local checkout has no configured Git remote, so no public repository or owner is asserted here.

The code base includes source imported from [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33d0e0711bc414ad992b640e1a37dd814`. This makes the implementation source-derived. Rewriting Git history or maintaining a separate repository does not make copied code clean-room work and does not remove copyright, license, or attribution obligations. Preserve the root and third-party license files, source notices, and the provenance reference in `UPSTREAM_SYNC`.

The supported product target is Apple Silicon on macOS, using Metal. Windows and Linux product, setup, packaging and CI paths have been removed. CUDA sources and standalone parity harnesses remain only as offline numerical-reference and provenance inputs; they are not built by the Mac product or included in its release. Ignored `build-dev` outputs are generated locally and can be removed after validation; they are not source files.
