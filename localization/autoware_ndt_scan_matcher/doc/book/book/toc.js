// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><a href="reader-map.html">How to read this book</a></li><li class="chapter-item expanded affix "><li class="part-title">Part I — Concepts</li><li class="chapter-item expanded "><a href="concepts/ndt-primer.html"><strong aria-hidden="true">1.</strong> NDT scan matching primer</a></li><li class="chapter-item expanded "><a href="concepts/scores.html"><strong aria-hidden="true">2.</strong> Scores: TP and NVTL</a></li><li class="chapter-item expanded "><a href="concepts/why-rust.html"><strong aria-hidden="true">3.</strong> Why a Rust port</a></li><li class="chapter-item expanded "><a href="concepts/scope.html"><strong aria-hidden="true">4.</strong> Scope and non-goals</a></li><li class="chapter-item expanded affix "><li class="part-title">Part II — Getting Started</li><li class="chapter-item expanded "><a href="start/build-and-test.html"><strong aria-hidden="true">5.</strong> Build and test</a></li><li class="chapter-item expanded "><a href="start/features.html"><strong aria-hidden="true">6.</strong> Feature flags and build configurations</a></li><li class="chapter-item expanded "><a href="start/using-the-crate.html"><strong aria-hidden="true">7.</strong> Using the Rust crate</a></li><li class="chapter-item expanded "><a href="start/ros-node.html"><strong aria-hidden="true">8.</strong> Running the ROS node</a></li><li class="chapter-item expanded affix "><li class="part-title">Part III — Architecture</li><li class="chapter-item expanded "><a href="arch/overview.html"><strong aria-hidden="true">9.</strong> System overview</a></li><li class="chapter-item expanded "><a href="arch/ffi-boundary.html"><strong aria-hidden="true">10.</strong> The FFI boundary</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="arch/ffi-types.html"><strong aria-hidden="true">10.1.</strong> C ABI types and view types</a></li><li class="chapter-item expanded "><a href="arch/ffi-ptr.html"><strong aria-hidden="true">10.2.</strong> ffi_ptr helpers and guard macros</a></li><li class="chapter-item expanded "><a href="arch/panic-containment.html"><strong aria-hidden="true">10.3.</strong> Panic containment and status codes</a></li><li class="chapter-item expanded "><a href="arch/host-vtable.html"><strong aria-hidden="true">10.4.</strong> The Host abstraction and C vtables</a></li></ol></li><li class="chapter-item expanded "><a href="arch/engine.html"><strong aria-hidden="true">11.</strong> The NDT engine</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="arch/engine-state.html"><strong aria-hidden="true">11.1.</strong> Engine state and the config API</a></li><li class="chapter-item expanded "><a href="arch/concurrency.html"><strong aria-hidden="true">11.2.</strong> Concurrency and interior mutability</a></li><li class="chapter-item expanded "><a href="arch/scratch.html"><strong aria-hidden="true">11.3.</strong> MatchScratch and the align entry points</a></li></ol></li><li class="chapter-item expanded "><a href="arch/align.html"><strong aria-hidden="true">12.</strong> The align hot path</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="arch/voxel-grid.html"><strong aria-hidden="true">12.1.</strong> Voxel grid and kd-tree</a></li><li class="chapter-item expanded "><a href="arch/derivatives.html"><strong aria-hidden="true">12.2.</strong> Serial and parallel derivatives</a></li></ol></li><li class="chapter-item expanded "><a href="arch/covariance.html"><strong aria-hidden="true">13.</strong> Covariance estimation</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="arch/tpe.html"><strong aria-hidden="true">13.1.</strong> The TPE pose search</a></li></ol></li><li class="chapter-item expanded "><a href="arch/map-update.html"><strong aria-hidden="true">14.</strong> Map update</a></li><li class="chapter-item expanded "><a href="arch/portability.html"><strong aria-hidden="true">15.</strong> Portability and the Host ports</a></li><li class="chapter-item expanded affix "><li class="part-title">Part IV — The C++ to Rust Port</li><li class="chapter-item expanded "><a href="port/verification.html"><strong aria-hidden="true">16.</strong> Behavior equivalence and verification</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="port/differential.html"><strong aria-hidden="true">16.1.</strong> Differential testing</a></li><li class="chapter-item expanded "><a href="port/trace-verification.html"><strong aria-hidden="true">16.2.</strong> Trace-based state-machine verification</a></li><li class="chapter-item expanded "><a href="port/numeric-parity.html"><strong aria-hidden="true">16.3.</strong> Numeric parity</a></li></ol></li><li class="chapter-item expanded "><a href="port/divergences.html"><strong aria-hidden="true">17.</strong> Divergences from upstream</a></li><li class="chapter-item expanded "><a href="port/symbol-map.html"><strong aria-hidden="true">18.</strong> C++ to Rust map</a></li><li class="chapter-item expanded affix "><li class="part-title">Part V — Real-Time and no_std</li><li class="chapter-item expanded "><a href="rt/wcet.html"><strong aria-hidden="true">19.</strong> The WCET contract</a></li><li class="chapter-item expanded "><a href="rt/zero-alloc.html"><strong aria-hidden="true">20.</strong> Zero-allocation guarantees</a></li><li class="chapter-item expanded "><a href="rt/mt.html"><strong aria-hidden="true">21.</strong> The mt multi-core engine</a></li><li class="chapter-item expanded "><a href="rt/panic-free.html"><strong aria-hidden="true">22.</strong> Panic-free, bounded execution</a></li><li class="chapter-item expanded affix "><li class="part-title">Part VI — Quality Gates</li><li class="chapter-item expanded "><a href="quality/hardening.html"><strong aria-hidden="true">23.</strong> Lint gates and suppression policy</a></li><li class="chapter-item expanded "><a href="quality/tests.html"><strong aria-hidden="true">24.</strong> Test taxonomy</a></li><li class="chapter-item expanded "><a href="quality/benchmarks.html"><strong aria-hidden="true">25.</strong> Benchmarking</a></li><li class="chapter-item expanded affix "><li class="part-title">Appendices</li><li class="chapter-item expanded "><a href="appendix/glossary.html"><strong aria-hidden="true">26.</strong> Glossary</a></li><li class="chapter-item expanded "><a href="appendix/parameters.html"><strong aria-hidden="true">27.</strong> Parameter reference</a></li><li class="chapter-item expanded "><a href="appendix/modules.html"><strong aria-hidden="true">28.</strong> Module index</a></li><li class="chapter-item expanded "><a href="appendix/references.html"><strong aria-hidden="true">29.</strong> References</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
