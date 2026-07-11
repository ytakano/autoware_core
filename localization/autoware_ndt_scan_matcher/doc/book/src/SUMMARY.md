# Summary

[Introduction](introduction.md)
[How to read this book](reader-map.md)

# Part I — Concepts

- [Why a Rust port](concepts/why-rust.md)
- [Scope and non-goals](concepts/scope.md)

# Part II — Getting Started

- [Build and test](start/build-and-test.md)
- [Running the ROS node](start/ros-node.md)

# Part III — Architecture

- [System overview](arch/overview.md)
- [The FFI boundary](arch/ffi-boundary.md)
    - [C ABI types and view types](arch/ffi-types.md)
    - [ffi_ptr helpers and guard macros](arch/ffi-ptr.md)
    - [Panic containment and status codes](arch/panic-containment.md)
    - [The Host abstraction and C vtables](arch/host-vtable.md)

# Part IV — The C++ to Rust Port

- [Behavior equivalence and verification](port/verification.md)
- [C++ to Rust map](port/symbol-map.md)

# Part V — Quality Gates

- [Lint gates and suppression policy](quality/hardening.md)
- [Test taxonomy](quality/tests.md)
- [Benchmarking](quality/benchmarks.md)

# Appendices

- [Glossary](appendix/glossary.md)
- [Parameter reference](appendix/parameters.md)
- [Module index](appendix/modules.md)
- [References](appendix/references.md)

---

> The **engine crate** (`realtime_ndt_scan_matcher`) — the NDT algorithm, the align hot path,
> covariance/TPE, and the real-time / `no_std` guarantees — has its **own book** under
> `realtime_ndt_scan_matcher/doc/book/`.
