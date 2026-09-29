/* membw_probe.c — host memory bandwidth, measured the same way on any rig.
 *
 * report.sh compiles this on the rig (cc -O3 -march=native -pthread) and prints what it
 * measures. CPU-offload decode speed is set by host memory bandwidth, and the rated
 * MT/s is blind to the failures that matter (a lost channel reads the same 3200 MT/s),
 * so the number has to be measured — and measured well enough to compare with the
 * platform's peak, which the first probe was not (~90 GB/s on an 8-channel EPYC that
 * does ~160 with the fixes below).
 *
 * What differs between rigs, and how this handles it:
 *   - write-allocate: a plain store first READS the line into cache, traffic STREAM
 *     does not count → non-temporal stores on x86 (SSE2/AVX, every x86-64 CPU).
 *     Other architectures keep plain stores; the read figure is unaffected either way.
 *   - topology (SMT, CCDs, NUMA, sockets, P/E cores): one thread per PHYSICAL core
 *     (sysfs topology), pinned, each first-touching the slice it streams, so pages
 *     land on its node. A run with fewer threads than cores SPREADS them evenly over
 *     the cores, ordered by (socket, L3 domain = die, core) so the spread holds
 *     whatever the CPU numbering (die by die, or sockets interleaved even/odd):
 *     half the threads still reach every CCD / NUMA node / socket. Packed onto the
 *     first cores, they would sit on half the dies of a big EPYC or Threadripper.
 *   - where bandwidth saturates (a few cores on a desktop, most of them on a server):
 *     tries 1/4, 1/2 and all physical cores, reports the best and how many got it.
 *   - large last-level caches (desktop X3D, EPYC V-Cache up to ~1 GB): arrays sized
 *     to at least 4x the total L3, capped by available memory.
 *   - noise: threads created once, barrier-timed, best of MEMBW_REPS (5) per count.
 *
 * Output, one line:  triad=<GB/s> read=<GB/s> threads=<n> cores=<n> array_mib=<n>
 *                    l3_mib=<n> capped=<0|1> nt=<0|1>
 * or  skip=<reason>.  GB = 1e9 bytes; Triad counts 24 bytes per element, as STREAM
 * does; read counts 16 (two arrays).
 *
 * Test seams (scripts/tests/test-report-membw.sh): MEMBW_SYSFS_ROOT (a fake
 * /sys/devices/system/cpu tree), MEMBW_MEMINFO (a fake /proc/meminfo), MEMBW_ARRAY_MIB,
 * MEMBW_REPS, MEMBW_PLAN_ONLY=1 (print the plan, measure nothing).
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#if defined(__AVX__)
#include <immintrin.h>
#define NT_STEP 4
#elif defined(__SSE2__)
#include <emmintrin.h>
#define NT_STEP 2
#else
#define NT_STEP 0
#endif

#define MAXCPU 4096
static char SYS[1024];                 /* …/devices/system/cpu */
static int mocked;
static int cpus[MAXCPU], ncpus;        /* logical CPUs we may run on */
static int cores[MAXCPU], ncores;      /* one logical CPU per physical core */

static int read_str(const char *path, char *buf, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, (int)n, f)) { fclose(f); return -1; }
    fclose(f);
    buf[strcspn(buf, "\n")] = 0;
    return 0;
}
static long read_long(const char *path, long dflt) {
    char b[64];
    return read_str(path, b, sizeof b) == 0 ? strtol(b, 0, 10) : dflt;
}

static void find_cpus(void) {
    if (mocked) {                      /* the fake tree: cpu0 … cpuN while present */
        char p[1200];
        for (int c = 0; c < MAXCPU; c++) {
            snprintf(p, sizeof p, "%s/cpu%d", SYS, c);
            if (access(p, F_OK) != 0) break;
            cpus[ncpus++] = c;
        }
        return;
    }
    cpu_set_t s;
    if (sched_getaffinity(0, sizeof s, &s) == 0) {
        for (int c = 0; c < CPU_SETSIZE && ncpus < MAXCPU; c++) if (CPU_ISSET(c, &s)) cpus[ncpus++] = c;
    }
    if (ncpus == 0) { long n = sysconf(_SC_NPROCESSORS_ONLN); for (int c = 0; c < n && c < MAXCPU; c++) cpus[ncpus++] = c; }
}

struct core_ent { long pkg, l3, core; int cpu; };
static int cmp_core(const void *x, const void *y) {
    const struct core_ent *a = x, *b = y;
    if (a->pkg != b->pkg) return a->pkg < b->pkg ? -1 : 1;
    if (a->l3 != b->l3) return a->l3 < b->l3 ? -1 : 1;
    if (a->core != b->core) return a->core < b->core ? -1 : 1;
    return a->cpu - b->cpu;
}

/* one CPU per (package, core): SMT siblings share a core and its path to memory.
 * Sorted by (package, L3 id, core) so an even spread over the list is an even
 * spread over sockets and dies. */
static void find_cores(void) {
    static long seen_pkg[MAXCPU], seen_core[MAXCPU];
    int nseen = 0;
    char p[1200];
    for (int i = 0; i < ncpus; i++) {
        int c = cpus[i];
        snprintf(p, sizeof p, "%s/cpu%d/topology/physical_package_id", SYS, c);
        long pkg = read_long(p, -1);
        snprintf(p, sizeof p, "%s/cpu%d/topology/core_id", SYS, c);
        long core = read_long(p, -1);
        if (pkg < 0 || core < 0) { cores[ncores++] = c; continue; }   /* unknown: count it */
        int dup = 0;
        for (int j = 0; j < nseen; j++) if (seen_pkg[j] == pkg && seen_core[j] == core) { dup = 1; break; }
        if (dup) continue;
        seen_pkg[nseen] = pkg; seen_core[nseen] = core; nseen++;
        cores[ncores++] = c;
    }
    if (ncores == 0) { memcpy(cores, cpus, sizeof(int) * (size_t)ncpus); ncores = ncpus; }
    static struct core_ent ent[MAXCPU];
    for (int i = 0; i < ncores; i++) {
        int c = cores[i];
        long l3 = -1;
        for (int idx = 0; idx < 10; idx++) {
            char lvl[16];
            snprintf(p, sizeof p, "%s/cpu%d/cache/index%d/level", SYS, c, idx);
            if (read_str(p, lvl, sizeof lvl) != 0 || strcmp(lvl, "3") != 0) continue;
            snprintf(p, sizeof p, "%s/cpu%d/cache/index%d/id", SYS, c, idx);
            l3 = read_long(p, -1);
            break;
        }
        snprintf(p, sizeof p, "%s/cpu%d/topology/physical_package_id", SYS, c);
        ent[i].pkg = read_long(p, -1);
        snprintf(p, sizeof p, "%s/cpu%d/topology/core_id", SYS, c);
        ent[i].core = read_long(p, -1);
        ent[i].l3 = l3; ent[i].cpu = c;
    }
    qsort(ent, (size_t)ncores, sizeof ent[0], cmp_core);
    for (int i = 0; i < ncores; i++) cores[i] = ent[i].cpu;
}

/* total L3: each L3 instance counted once, keyed by its cache `id` (as lscpu does),
 * else by shared_cpu_list. A VM can report every vCPU sharing one list while giving
 * each its own id; the id is what bare metal and lscpu agree on. */
static long total_l3_bytes(void) {
    static char seen[256][256];
    int nseen = 0;
    long total = 0;
    char p[1200], lvl[16], sz[64], shared[256];
    for (int i = 0; i < ncpus; i++) {
        for (int idx = 0; idx < 10; idx++) {
            snprintf(p, sizeof p, "%s/cpu%d/cache/index%d/level", SYS, cpus[i], idx);
            if (read_str(p, lvl, sizeof lvl) != 0) continue;   /* a gap is not the end */
            if (strcmp(lvl, "3") != 0) continue;
            char idv[64];
            snprintf(p, sizeof p, "%s/cpu%d/cache/index%d/id", SYS, cpus[i], idx);
            if (read_str(p, idv, sizeof idv) == 0) {
                snprintf(shared, sizeof shared, "id:%s", idv);
            } else {
                snprintf(p, sizeof p, "%s/cpu%d/cache/index%d/shared_cpu_list", SYS, cpus[i], idx);
                char lst[200];
                if (read_str(p, lst, sizeof lst) == 0) snprintf(shared, sizeof shared, "cpus:%s", lst);
                else snprintf(shared, sizeof shared, "cpu:%d", cpus[i]);
            }
            int dup = 0;
            for (int j = 0; j < nseen; j++) if (strcmp(seen[j], shared) == 0) { dup = 1; break; }
            if (dup) continue;
            snprintf(p, sizeof p, "%s/cpu%d/cache/index%d/size", SYS, cpus[i], idx);
            if (read_str(p, sz, sizeof sz) != 0) continue;
            long v = strtol(sz, 0, 10);
            char u = sz[strspn(sz, "0123456789")];
            if (u == 'K') v *= 1024L; else if (u == 'M') v *= 1024L * 1024; else if (u == 'G') v *= 1024L * 1024 * 1024;
            total += v;
            if (nseen < 256) snprintf(seen[nseen++], sizeof seen[0], "%s", shared);
        }
    }
    return total;
}

static long mem_available_bytes(void) {
    const char *path = getenv("MEMBW_MEMINFO") ? getenv("MEMBW_MEMINFO") : "/proc/meminfo";
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb < 0 ? -1 : kb * 1024L;
}

/* the physical core thread `id` of `nt` runs on: spread evenly over the core list */
static int pin_for(long id, int nt) { return cores[(long)id * ncores / nt]; }

/* ── the measurement ───────────────────────────────────────────────────────── */
static double *A, *B, *C;
static long N;                          /* elements per array */
static int NT, REPS;
static pthread_barrier_t bar;
static double t0, best_triad, best_read;
static volatile uint64_t sink;

static double now(void) { struct timespec s; clock_gettime(CLOCK_MONOTONIC, &s); return s.tv_sec + s.tv_nsec / 1e9; }

static void *worker(void *arg) {
    long id = (long)arg;
    long lo = (N * id / NT) & ~7L, hi = id == NT - 1 ? N : (N * (id + 1) / NT) & ~7L;
    if (!mocked) {
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(pin_for(id, NT), &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }
    for (long i = lo; i < hi; i++) { A[i] = 1.0; B[i] = 2.0; C[i] = 0.0; }   /* first touch */
    for (int r = 0; r < REPS; r++) {
        /* Triad: c = a + 3b */
        pthread_barrier_wait(&bar); if (id == 0) t0 = now(); pthread_barrier_wait(&bar);
#if NT_STEP == 4
        __m256d k = _mm256_set1_pd(3.0);
        for (long i = lo; i < hi; i += 4)
            _mm256_stream_pd(&C[i], _mm256_add_pd(_mm256_load_pd(&A[i]), _mm256_mul_pd(k, _mm256_load_pd(&B[i]))));
        _mm_sfence();
#elif NT_STEP == 2
        __m128d k = _mm_set1_pd(3.0);
        for (long i = lo; i < hi; i += 2)
            _mm_stream_pd(&C[i], _mm_add_pd(_mm_load_pd(&A[i]), _mm_mul_pd(k, _mm_load_pd(&B[i]))));
        _mm_sfence();
#else
        for (long i = lo; i < hi; i++) C[i] = A[i] + 3.0 * B[i];
#endif
        pthread_barrier_wait(&bar);
        if (id == 0) { double g = 24.0 * N / (now() - t0) / 1e9; if (g > best_triad) best_triad = g; }
        /* read: integer sum over a and b (associative, so it vectorizes at -O3) */
        pthread_barrier_wait(&bar); if (id == 0) t0 = now(); pthread_barrier_wait(&bar);
        const uint64_t *ua = (const uint64_t *)A, *ub = (const uint64_t *)B;
        uint64_t acc = 0;
        for (long i = lo; i < hi; i++) acc += ua[i] + ub[i];
        sink += acc;
        pthread_barrier_wait(&bar);
        if (id == 0) { double g = 16.0 * N / (now() - t0) / 1e9; if (g > best_read) best_read = g; }
    }
    return 0;
}

static int run(int nt, double *triad, double *read) {
    size_t bytes = (size_t)N * sizeof(double);
    A = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    B = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    C = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (A == MAP_FAILED || B == MAP_FAILED || C == MAP_FAILED) return -1;
    NT = nt; best_triad = best_read = 0;
    pthread_barrier_init(&bar, 0, (unsigned)nt);
    pthread_t *th = calloc((size_t)nt, sizeof *th);
    for (long i = 0; i < nt; i++) pthread_create(&th[i], 0, worker, (void *)i);
    for (long i = 0; i < nt; i++) pthread_join(th[i], 0);
    free(th);
    pthread_barrier_destroy(&bar);
    munmap(A, bytes); munmap(B, bytes); munmap(C, bytes);   /* fresh pages per run: first touch again */
    *triad = best_triad; *read = best_read;
    return 0;
}

int main(void) {
    const char *root = getenv("MEMBW_SYSFS_ROOT");
    mocked = root && *root;
    snprintf(SYS, sizeof SYS, "%s", mocked ? root : "/sys/devices/system/cpu");
    REPS = getenv("MEMBW_REPS") ? atoi(getenv("MEMBW_REPS")) : 5;
    if (REPS < 1) REPS = 1;

    find_cpus();
    find_cores();
    long l3 = total_l3_bytes();
    const long MiB = 1024L * 1024;
    long per = 4 * l3 / 3;                               /* 3 arrays together >= 4x L3 */
    if (per < 256 * MiB) per = 256 * MiB;
    per = (per + 2 * MiB - 1) / (2 * MiB) * (2 * MiB);
    int capped = 0;
    long avail = mem_available_bytes();
    if (avail > 0 && per > avail / 2 / 3) { per = avail / 2 / 3 / (2 * MiB) * (2 * MiB); capped = 1; }
    if (getenv("MEMBW_ARRAY_MIB")) { per = atol(getenv("MEMBW_ARRAY_MIB")) * MiB; capped = 0; }
    if (per < 64 * MiB && !getenv("MEMBW_ARRAY_MIB")) { printf("skip=low-memory avail_mib=%ld\n", avail / MiB); return 0; }
    N = per / (long)sizeof(double);

    int counts[3] = { ncores / 4, ncores / 2, ncores }, nc = 0, plan[3];
    for (int i = 0; i < 3; i++) {
        int c = counts[i] < 1 ? 1 : counts[i], dup = 0;
        for (int j = 0; j < nc; j++) if (plan[j] == c) dup = 1;
        if (!dup) plan[nc++] = c;
    }
    if (getenv("MEMBW_PLAN_ONLY")) {
        printf("plan cpus=%d cores=%d threads=", ncpus, ncores);
        for (int i = 0; i < nc; i++) printf("%s%d", i ? "," : "", plan[i]);
        printf(" pins=");                /* <threads>:<cpus>|… — which CPUs each run uses */
        for (int i = 0; i < nc; i++) {
            printf("%s%d:", i ? "|" : "", plan[i]);
            for (long t = 0; t < plan[i]; t++) printf("%s%d", t ? "," : "", pin_for(t, plan[i]));
        }
        printf(" l3_mib=%ld array_mib=%ld capped=%d nt=%d\n", l3 / MiB, per / MiB, capped, NT_STEP ? 1 : 0);
        return 0;
    }
    double bt = 0, br = 0;
    int bn = 0;
    for (int i = 0; i < nc; i++) {
        double t, r;
        if (run(plan[i], &t, &r) != 0) { printf("skip=alloc-failed\n"); return 0; }
        if (t > bt) { bt = t; bn = plan[i]; }
        if (r > br) br = r;
    }
    printf("triad=%.1f read=%.1f threads=%d cores=%d array_mib=%ld l3_mib=%ld capped=%d nt=%d\n",
           bt, br, bn, ncores, per / MiB, l3 / MiB, capped, NT_STEP ? 1 : 0);
    return 0;
}
