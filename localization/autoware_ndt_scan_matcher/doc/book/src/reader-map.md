# How to read this book

This book covers the **node crate** (`autoware_ndt_scan_matcher_rs`) — the C ABI, the `Host`
vtable, the ROS 2 node shell, and the C++→Rust port map. The **engine crate**
(`realtime_ndt_scan_matcher`) — the NDT algorithm, the align hot path, covariance/TPE, and the
real-time / `no_std` guarantees — has its **own book** under
`realtime_ndt_scan_matcher/doc/book/`. Start where your goal sits.

| You are… | Start with | Then |
|---|---|---|
| **An upstream / PR reviewer** | [Behavior equivalence and verification](port/verification.md) | [Quality Gates](quality/hardening.md) (lints) → [System overview](arch/overview.md); *Divergences from upstream* in the engine book |
| **A maintainer / contributor** | [Part III — Architecture](arch/overview.md) (overview → the FFI boundary) | [Symbol map](port/symbol-map.md) → [Test taxonomy](quality/tests.md); the engine internals in the engine book |
| **A ROS integrator** | [Part II — Getting Started](start/build-and-test.md) (build, run the node) | [The FFI boundary](arch/ffi-boundary.md) → [Parameter reference](appendix/parameters.md) |
| **A Rust crate user / RT engineer** | the **engine crate book** (using the crate, features, the engine, the align hot path, WCET, `no_std`) | back here for [the FFI boundary](arch/ffi-boundary.md) and the ROS node |

## Conventions

- **Code that is meant to compile** is shown as tested `rust` doctests copied from the crate's
  rustdoc. **Illustrative snippets and excerpts** use `rust,ignore` or `text` and are not compiled.
- **Paths** like `src/ffi.rs` are relative to this (node) crate's root
  (`autoware_ndt_scan_matcher/autoware_ndt_scan_matcher_rs/`). A path in the engine crate is
  written `realtime_ndt_scan_matcher/src/…`.
- **`Aw*`** names are C-ABI FFI types; **`Ndt*` / `ScanMatcher`** are Rust-side types.
- **TP** = transform probability, **NVTL** = nearest-voxel transformation likelihood
  (see the [Glossary](appendix/glossary.md)).
- Each chapter ends with a **Source** note listing the in-tree files it distills, so a reader can go
  from the prose to the authoritative code.
