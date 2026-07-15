// Copyright 2024 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// LD_PRELOAD heap-allocation counter for the WCET C++ comparison (plan/ndt_wcet.md, M4).
//
// Counts malloc/calloc/realloc calls process-wide WITHOUT touching the upstream C++ sources
// (which must stay byte-identical). ndt_bench_replay's fixture mode looks up the two exported
// hooks via dlsym(RTLD_DEFAULT, ...) at runtime — if this library is preloaded it resets the
// counter after warmup and reads it after the timed align window, yielding allocations/align for
// each engine; if not preloaded, the hooks are absent and counting is skipped. The Rust engine is
// independently verified zero-alloc by engine/tests/zero_alloc.rs; this interposer is how we get
// the same number for the C++ engine.
//
// Build:  cc -O2 -shared -fPIC -o alloc_count.so alloc_count.c -ldl
// Use:    LD_PRELOAD=$PWD/alloc_count.so ndt_bench_replay --fixture ...
//
// dlsym(RTLD_NEXT, "calloc") may itself call calloc on some glibc versions; a small static
// bootstrap buffer breaks that recursion (standard interposer trick).

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>

static void * (*real_malloc)(size_t) = NULL;
static void * (*real_calloc)(size_t, size_t) = NULL;
static void * (*real_realloc)(void *, size_t) = NULL;
static void (*real_free)(void *) = NULL;

/* Total allocation events (malloc + calloc + growing realloc) since the last reset. */
static _Atomic unsigned long long g_allocs = 0;

/* Bootstrap arena for the calloc dlsym recursion. */
static char boot_buf[4096];
static size_t boot_used = 0;

static void init_real(void)
{
  if (!real_malloc) real_malloc = (void *(*)(size_t))dlsym(RTLD_NEXT, "malloc");
  if (!real_calloc) real_calloc = (void *(*)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
  if (!real_realloc) real_realloc = (void *(*)(void *, size_t))dlsym(RTLD_NEXT, "realloc");
  if (!real_free) real_free = (void (*)(void *))dlsym(RTLD_NEXT, "free");
}

/* Exported hooks (found via dlsym(RTLD_DEFAULT, ...) by ndt_bench_replay). */
void alloc_count_reset(void)
{
  __atomic_store_n(&g_allocs, 0ULL, __ATOMIC_SEQ_CST);
}

unsigned long long alloc_count_get(void)
{
  return __atomic_load_n(&g_allocs, __ATOMIC_SEQ_CST);
}

void * malloc(size_t size)
{
  init_real();
  __atomic_add_fetch(&g_allocs, 1ULL, __ATOMIC_RELAXED);
  return real_malloc(size);
}

void * calloc(size_t nmemb, size_t size)
{
  if (!real_calloc) {
    /* dlsym itself may call calloc: satisfy it from the static bootstrap arena. */
    void * (*rc)(size_t, size_t) =
      (void *(*)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
    if (!rc) {
      size_t total = nmemb * size;
      if (boot_used + total > sizeof(boot_buf)) return NULL;
      void * p = boot_buf + boot_used;
      boot_used += total;
      memset(p, 0, total);
      return p;
    }
    real_calloc = rc;
  }
  __atomic_add_fetch(&g_allocs, 1ULL, __ATOMIC_RELAXED);
  return real_calloc(nmemb, size);
}

void * realloc(void * ptr, size_t size)
{
  init_real();
  __atomic_add_fetch(&g_allocs, 1ULL, __ATOMIC_RELAXED);
  return real_realloc(ptr, size);
}

void free(void * ptr)
{
  /* Bootstrap-arena pointers are never freed (leaked by design; the arena is 4 KiB once). */
  if ((char *)ptr >= boot_buf && (char *)ptr < boot_buf + sizeof(boot_buf)) return;
  init_real();
  real_free(ptr);
}
