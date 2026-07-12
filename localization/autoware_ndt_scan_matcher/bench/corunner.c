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

// Minimal one-resource-at-a-time interference co-runner for the WCET measurement campaign
// (plan/ndt_timing_measurement_policy.md, "Interference Sensitivity Experiment"). No
// stress-ng dependency; the campaign orchestrator (bench/wcet_campaign.py) compiles this
// file, pins it with taskset to a CPU sharing the intended hardware resource, and kills it
// with SIGTERM when the series ends.
//
//   corunner membw [mib]   large-array streaming copy (memory bandwidth; default 256 MiB)
//   corunner llc   [mib]   pointer-chase over an LLC-sized working set (default 32 MiB)
//   corunner fp            dense multiply-add loop on a small array (FP execution units)
//   corunner calib         fixed-work FP spin; prints elapsed nanoseconds and exits
//                          (throttling guard: frequency reporting is stale on
//                          nohz_full/isolated cores, so the campaign guards on measured
//                          sustained speed instead)
//
// The interference modes loop until terminated. Build: cc -O2 -o corunner corunner.c

#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;
static void on_term(int sig)
{
  (void)sig;
  g_stop = 1;
}

// Volatile sink so the optimizer cannot delete the loops.
static volatile double g_sink_f = 0.0;
static volatile size_t g_sink_u = 0;

static int run_membw(size_t mib)
{
  const size_t n = mib * 1024 * 1024;
  char * a = malloc(n);
  char * b = malloc(n);
  if (a == NULL || b == NULL) {
    fprintf(stderr, "corunner: membw: cannot allocate 2x%zu MiB\n", mib);
    return 1;
  }
  memset(a, 0x5a, n);
  while (!g_stop) {
    memcpy(b, a, n);
    memcpy(a, b, n);
    g_sink_u += (size_t)(unsigned char)b[n / 2];
  }
  free(a);
  free(b);
  return 0;
}

static int run_llc(size_t mib)
{
  const size_t n = (mib * 1024 * 1024) / sizeof(size_t);
  size_t * next = malloc(n * sizeof(size_t));
  if (next == NULL) {
    fprintf(stderr, "corunner: llc: cannot allocate %zu MiB\n", mib);
    return 1;
  }
  // Sattolo shuffle: one full-cycle random permutation -> a pointer chase that defeats
  // prefetchers and touches the whole working set.
  for (size_t i = 0; i < n; ++i) next[i] = i;
  unsigned long long s = 0x9e3779b97f4a7c15ULL;
  for (size_t i = n - 1; i > 0; --i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const size_t j = (size_t)(s % i);
    const size_t t = next[i];
    next[i] = next[j];
    next[j] = t;
  }
  size_t p = 0;
  while (!g_stop) {
    for (size_t hop = 0; hop < 1u << 20; ++hop) p = next[p];
    g_sink_u += p;
  }
  free(next);
  return 0;
}

static int run_fp(void)
{
  double acc[8] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
  while (!g_stop) {
    for (long i = 0; i < 1L << 22; ++i) {
      for (int k = 0; k < 8; ++k) acc[k] = acc[k] * 1.0000001 + 1e-9;
    }
    g_sink_f += acc[0];
  }
  return 0;
}

// Fixed-work calibration spin (~100-200 ms at full clock): prints elapsed ns to stdout.
static int run_calib(void)
{
  double acc[8] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (long i = 0; i < 60L * (1L << 20); ++i) {
    for (int k = 0; k < 8; ++k) acc[k] = acc[k] * 1.0000001 + 1e-9;
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  g_sink_f += acc[0];
  const long long ns =
    (long long)(t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
  printf("%lld\n", ns);
  return 0;
}

int main(int argc, char ** argv)
{
  if (argc < 2) {
    fprintf(stderr, "usage: corunner membw|llc|fp|calib [mib]\n");
    return 2;
  }
  signal(SIGTERM, on_term);
  signal(SIGINT, on_term);
  const char * mode = argv[1];
  const size_t mib = (argc >= 3) ? (size_t)strtoul(argv[2], NULL, 10) : 0;
  if (strcmp(mode, "membw") == 0) return run_membw(mib > 0 ? mib : 256);
  if (strcmp(mode, "llc") == 0) return run_llc(mib > 0 ? mib : 32);
  if (strcmp(mode, "fp") == 0) return run_fp();
  if (strcmp(mode, "calib") == 0) return run_calib();
  fprintf(stderr, "corunner: unknown mode '%s'\n", mode);
  return 2;
}
