import ctypes
import os
import platform
    
class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]

class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int)
    ]

os_name = platform.system()
if os_name == 'Windows':
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cg.dll")
elif os_name == "Darwin":
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcg.dylib")
elif platform.machine() in ('arm64', 'aarch64'):
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcg-arm64.so")
else:
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcg.so")
lib = ctypes.cdll.LoadLibrary(lib_path)

lib.GameInitialize()

# --- optional seed hook (Linux only) ---
_hook_lib = None
_hook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcg_seed_hook.so")
if os.path.exists(_hook_path):
    _hook_lib = ctypes.CDLL(_hook_path)
    _hook_lib.CgInstallHook.argtypes = [ctypes.c_char_p]
    _hook_lib.CgInstallHook.restype  = ctypes.c_int
    _hook_lib.CgSetSeed.argtypes     = [ctypes.c_uint]
    _hook_lib.CgSetSeed.restype      = None
    _hook_lib.CgHookInstalled.restype = ctypes.c_int
    ok = _hook_lib.CgInstallHook(lib_path.encode())
    if not ok:
        import warnings
        warnings.warn("seed_hook: GOT patch failed — set_seed() will have no effect")

def set_seed(seed: int) -> None:
    """Seed libcg.so's internal RNG so games are reproducible.

    Must be called before battle_start().  Has no effect if libcg_seed_hook.so
    has not been built (run cg/build_seed_hook.sh first).
    """
    if _hook_lib is None:
        raise RuntimeError(
            "libcg_seed_hook.so not found. Build it with:\n"
            "  bash cg/build_seed_hook.sh"
        )
    _hook_lib.CgSetSeed(ctypes.c_uint(seed))

lib.BattleStart.restype = StartData
lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]

lib.AgentStart.restype = ctypes.c_void_p

lib.BattleFinish.argtypes = [ctypes.c_void_p]

lib.GetBattleData.restype = SerialData
lib.GetBattleData.argtypes = [ctypes.c_void_p]

lib.Select.restype = ctypes.c_int
lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.VisualizeData.restype = ctypes.c_char_p
lib.VisualizeData.argtypes = [ctypes.c_void_p]

lib.SearchBegin.restype = ctypes.c_char_p
lib.SearchBegin.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int]

lib.SearchStep.restype = ctypes.c_char_p
lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.SearchEnd.argtypes = [ctypes.c_void_p]

lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]

lib.AllCard.restype = ctypes.c_char_p

lib.AllAttack.restype = ctypes.c_char_p

class Battle:
    battle_ptr = None
    obs = None
