/*
 * seed_hook.cpp
 *
 * Runtime GOT patcher for libcg.so.
 * Replaces std::random_device::_M_getval() in libcg.so's PLT with a
 * seeded xorshift32, making all deck shuffles and coin flips reproducible.
 *
 * Build:
 *   g++ -shared -fPIC -O2 -o libcg_seed_hook.so seed_hook.cpp -ldl
 *
 * Usage from Python (via sim.py):
 *   CgInstallHook(libcg_path)   -- call once after libcg.so is loaded
 *   CgSetSeed(seed)             -- call before BattleStart
 */

#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <atomic>

// ---------- seeded PRNG ----------

static std::atomic<uint32_t> g_state{42u};

static uint32_t xorshift32() {
    uint32_t s = g_state.load(std::memory_order_relaxed);
    if (!s) s = 1u;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    g_state.store(s, std::memory_order_relaxed);
    return s;
}

// Replacement for std::random_device::_M_getval().
// Called as a member function so 'this' arrives in rdi; we ignore it.
static unsigned int seeded_getval(void* /*self*/) {
    return xorshift32();
}

// ---------- GOT patcher ----------

static bool patch_got(const char* lib_path, const char* sym_mangled, void* replacement) {
    // RTLD_NOLOAD: get handle to already-loaded library without re-loading
    void* handle = dlopen(lib_path, RTLD_NOLOAD | RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "[seed_hook] dlopen(RTLD_NOLOAD) failed for %s: %s\n",
                lib_path, dlerror());
        return false;
    }

    struct link_map* lm = nullptr;
    if (dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0 || !lm) {
        fprintf(stderr, "[seed_hook] dlinfo failed\n");
        dlclose(handle);
        return false;
    }

    ElfW(Addr) base = lm->l_addr;  // load slide
    ElfW(Dyn)* dyn  = lm->l_ld;   // pointer to already-mapped .dynamic section

    ElfW(Rela)* rela_plt = nullptr;
    size_t      rela_count = 0;
    ElfW(Sym)*  symtab = nullptr;
    const char* strtab = nullptr;

    // After the dynamic linker applies R_RELATIVE relocations, d_un.d_ptr
    // values in DT_JMPREL / DT_SYMTAB / DT_STRTAB are already absolute
    // virtual addresses — use them as-is.
    for (ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; ++d) {
        switch (d->d_tag) {
            case DT_JMPREL:
                rela_plt = reinterpret_cast<ElfW(Rela)*>(d->d_un.d_ptr);
                break;
            case DT_PLTRELSZ:
                rela_count = d->d_un.d_val / sizeof(ElfW(Rela));
                break;
            case DT_SYMTAB:
                symtab = reinterpret_cast<ElfW(Sym)*>(d->d_un.d_ptr);
                break;
            case DT_STRTAB:
                strtab = reinterpret_cast<const char*>(d->d_un.d_ptr);
                break;
        }
    }

    if (!rela_plt || !symtab || !strtab || rela_count == 0) {
        fprintf(stderr, "[seed_hook] missing JMPREL/SYMTAB/STRTAB in %s\n", lib_path);
        dlclose(handle);
        return false;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    bool found = false;

    for (size_t i = 0; i < rela_count; ++i) {
        unsigned sym_idx = ELF64_R_SYM(rela_plt[i].r_info);
        const char* name = strtab + symtab[sym_idx].st_name;

        if (std::strcmp(name, sym_mangled) != 0)
            continue;

        // r_offset is an ELF virtual address (0-based); add base to get
        // the actual runtime address of the GOT slot.
        void** got_slot = reinterpret_cast<void**>(base + rela_plt[i].r_offset);

        // Make the page writable (handles RELRO), patch, restore.
        uintptr_t page = reinterpret_cast<uintptr_t>(got_slot)
                         & ~static_cast<uintptr_t>(page_size - 1);
        mprotect(reinterpret_cast<void*>(page), page_size, PROT_READ | PROT_WRITE);
        *got_slot = replacement;
        mprotect(reinterpret_cast<void*>(page), page_size, PROT_READ | PROT_WRITE);

        found = true;
        break;
    }

    if (!found)
        fprintf(stderr, "[seed_hook] symbol '%s' not found in PLT of %s\n",
                sym_mangled, lib_path);

    dlclose(handle);
    return found;
}

// ---------- public C API (called from Python via ctypes) ----------

static bool g_installed = false;

extern "C" {
    // Set the seed used for all future random_device calls inside libcg.so.
    // Call before BattleStart.
    void CgSetSeed(uint32_t seed) {
        g_state.store(seed ? seed : 1u, std::memory_order_relaxed);
    }

    // Patch libcg.so's GOT so _M_getval points to our seeded version.
    // libcg_path must be the exact path passed to ctypes when loading the lib.
    // Returns 1 on success, 0 on failure.
    // Safe to call multiple times.
    int CgInstallHook(const char* libcg_path) {
        if (g_installed) return 1;
        bool ok = patch_got(libcg_path,
                            "_ZNSt13random_device9_M_getvalEv",
                            reinterpret_cast<void*>(seeded_getval));
        g_installed = ok;
        return ok ? 1 : 0;
    }

    // Returns 1 if the hook has been successfully installed.
    int CgHookInstalled() {
        return g_installed ? 1 : 0;
    }
}
