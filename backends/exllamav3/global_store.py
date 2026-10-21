"""GLOBAL CONTINUATION STORE for exllamav3 ngram drafting (campaign: draft-map).

Replaces the session-local suffix-automaton proposer with a GLOBAL,
deployment-wide continuation store: n-gram keys harvested from every
live job's token stream (append-only, causal, per-job watermarks),
ladder lookup 16/8/4, longest key wins, top-1 continuation. Falls back
to the stock session-local SAM matcher on a store miss (never worse
than baseline).

Evidence base: infra-stack/draft-map/campaign.md (maps win at depth,
self-heal under staleness, accelerate with density, inert under
poisoning; recipe: ladder 16/8/4, min key, one tree shape).

Config via env (all optional):
  GS_ENABLE=1  GS_MIN_KEY=4  GS_LADDER=16,8,4  GS_TOPK=1
  GS_STORE=/path/store.bin  GS_DUMP_SECS=300
Author: draft-map campaign, 2026-10-09. Reviewable in isolation;
removal = delete import line.
"""
import atexit
import math
import os
import pickle
import threading
import time
from collections import Counter, defaultdict

try:
    from loguru import logger as _lg          # tabby dependency: routes to BOTH
    _HAS_LG = True                             # console and the file sink
except ImportError:
    _lg, _HAS_LG = None, False

def _ui(msg):
    if _HAS_LG:
        _lg.info(msg)
    else:
        _ui(msg)

ENABLE = os.environ.get("GS_ENABLE", "0") == "1"
WINDOW = int(os.environ.get("GS_WINDOW", "16384"))
MIN_KEY = int(os.environ.get("GS_MIN_KEY", "4"))
LADDER = tuple(int(x) for x in os.environ.get("GS_LADDER", "16,8,4").split(","))
TOPK = int(os.environ.get("GS_TOPK", "1"))  # RESERVED: propose is always top-1 today
_UNION_MODE = os.environ.get("GS_UNION", "off")
_ACCEP = os.environ.get("GS_ACCEP", "0") == "1"  # engine acceptance telemetry (dev-engine)  # off | on | auto=per-job 50/50
_UTOG = [0]

def _union_for(jid):
    # NOTE: must be called ONCE per hit - a multi-call site round-robin
    # anti-locks the arms (regression 01:12: d_len_n=55, u_len_n=0 despite
    # u_try=52 -> behaved as union-on). Single-draw wiring is a morning task;
    # until then `auto` is honestly documented as on-equivalent.
    if _UNION_MODE == "on":
        return True
    if _UNION_MODE == "auto":
        _UTOG[0] += 1
        return _UTOG[0] % 2 == 0   # round-robin per hit; U captured once upstream
    return False
MODE = os.environ.get("GS_MODE", "top1")        # top1 | ev | tree
TREE_B = int(os.environ.get("GS_TREE_B", "8"))
TREE_BETA = float(os.environ.get("GS_TREE_BETA", "-0.05"))
STORE_PATH = os.environ.get("GS_STORE", "")
DUMP_SECS = int(os.environ.get("GS_DUMP_SECS", "0"))  # 0 = only save at clean exit
ENGINE_TREE_T = int(os.environ.get("GS_ENGINE_T", "0"))  # 0 = flat chain (default, prod-inert); >0 = engine tree budget
MAX_KEYS = int(os.environ.get("GS_MAX_KEYS", "1000000"))  # store SIZE is the first-order monotone lever (draft-map L105); dump waits scale with BATCH not total — proven 41ms worst/41ms worst wait at 3M entries 2026-10-10

_lad = {L: {} for L in LADDER}          # L -> {key_tuple: Counter(continuation)}
_wm = {}                               # job id -> ingested length (watermark)
_lock = threading.Lock()
_calls = Counter()
_HITFLAG = [0]                     # observability: hit/miss/fallback

def _ingest(seq, spec=frozenset(), jid=None, base=0, n_total=None):
    """Append-only causal ingest of a job's current token stream from its
    watermark. Idempotent; stream growth is monotone (append-only jobs).
    jid MUST be a stable per-job key (id(job), not id(seq)) — the caller
    rebuilds seq every step; keying on the fresh list re-ingested the whole
    context per token and pinned the decode thread (prod incident 2026-10-10,
    sub-tok/s). n < w means the id was recycled by a new job: start over.
    base/n_total: seq may be a TAIL slice starting at global index base;
    n_total is the global length. Callers must pass base <= lo - LADDER[0]
    so every needed key/window lies inside the slice."""
    n = len(seq) if n_total is None else n_total
    if jid is None:
        jid = id(seq)
    w = _wm.get(jid, 0)
    if n < w:
        w = 0
    if n == w:
        return
    lo = max(1, w - LADDER[0] + 1)
    if base:
        lo = max(lo, base + LADDER[0])
    for i in range(lo, n):
        jl = i - base
        if spec and spec.intersection(
                seq[max(0, jl - LADDER[0]):jl + 16]):
            continue
        cont = list(seq[jl:min(jl + 16, n - base)])
        if spec:
            cut = next((j for j, t in enumerate(cont) if t in spec), None)
            if cut is not None:
                del cont[cut:]
        cont = tuple(cont)
        if len(cont) < MIN_KEY:
            continue
        for L in LADDER:
            if i < L:
                continue
            k = tuple(seq[jl - L:jl])
            d = _lad[L]
            if k not in d and len(d) >= MAX_KEYS:
                continue
            d.setdefault(k, Counter())[cont] += 1
    _wm[jid] = n

EV = os.environ.get("GS_EV", "0") == "1"
GATE_MIN = int(os.environ.get("GS_GATE", "2"))

def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

def _ev_chain(entry, draft_length):
    top = entry.most_common(8)
    if sum(c for _, c in top) < GATE_MIN:
        return None
    best, best_score = None, 0
    for cand, _ in top:
        for l in range(1, min(len(cand), draft_length) + 1):
            pref = cand[:l]
            score = 0
            for ci, cnt in top:
                score += cnt * _lcp(pref, ci)
            if score > best_score:
                best_score, best = score, pref
    return list(best) if best else None

def _tree_spine(entry, draft_length):
    """Top-B tree (Lemma-1: score = log(count) + beta*depth, beta<=0 =>
    global top-B is prefix-closed optimal) reduced to its best single
    chain: the spine the engine can verify as one flat draft row.
    Deterministic; ties prefer shorter prefixes."""
    B = max(1, TREE_B)
    beta = min(0.0, TREE_BETA)
    nodes = {}
    for cont, cnt in entry.items():
        for d in range(1, min(len(cont), draft_length) + 1):
            p = cont[:d]
            nodes[p] = nodes.get(p, 0) + cnt
    scored = sorted(nodes, key=lambda p: (-(math.log(nodes[p]) + beta * len(p)),
                                          len(p), p))
    top = set(scored[:B])
    spine = []
    for d in range(1, draft_length + 1):
        cands = [p for p in top if len(p) == d and tuple(p[:d - 1]) == tuple(spine)]
        if not cands:
            break
        spine = list(min(cands, key=lambda p: (-nodes[p], p)))
    return spine or None

def propose_tree(seq, draft_length, budget=15):
    """Tree proposal: top-B chains from the deepest ladder hit, deduplicated
    into DFS pre-order nodes. Returns None or dict(nodes=[tok...], parent=[i...],
    depth=[i...], columns=[(start,end)...] per-parent-col) with len(nodes)<=budget.
    budget=1 -> EXACTLY the single chain propose() returns (identity law).
    Pure data: engine wiring lands in milestone M2."""
    if budget < 1:
        return None
    chains = _propose_chains(seq, draft_length, budget if budget > 1 else 1)
    if not chains:
        return None
    if budget == 1:
        c = chains[0]
        return {"nodes": list(c), "parent": [-1] + list(range(len(c) - 1)),
                "depth": list(range(len(c)))}
    root = []
    trie = {}  # prefix tuple -> None sentinel dict
    nodes, parent, depth = [], [], []
    def ins(prefix_chain):
        cur = trie
        pidx = -1
        for tok in prefix_chain:
            nxt = cur.get(tok)
            if nxt is None:
                nxt = {}
                cur[tok] = nxt
                nodes.append(tok)
                parent.append(pidx)
                depth.append(len(prefix_chain) - prefix_chain.index(prefix_chain[0]) if False else 0)
                pidx = len(nodes) - 1
            else:
                pidx = nxt["_i"] if "_i" in nxt else pidx
            # track node index
            if "_i" not in nxt:
                nxt["_i"] = pidx
            cur = nxt
    # rebuild with explicit index bookkeeping (clearer than the above sketch)
    nodes, parent, depth = [], [], []
    root_map = {}
    for chain in chains:
        cur = root_map
        pi = -1
        for d, tok in enumerate(chain):
            nxt = cur.get(tok)
            if nxt is None:
                nodes.append(tok)
                parent.append(pi)
                depth.append(d)
                nxt = {"_i": len(nodes) - 1, "_kids": {}}
                cur[tok] = nxt
            cur = nxt["_kids"]
            pi = nxt["_i"]
            if len(nodes) >= budget:
                break
        if len(nodes) >= budget:
            break
    return {"nodes": nodes, "parent": parent, "depth": depth}


def _propose_chains(seq, draft_length, k):
    """Up to k distinct continuation chains for the deepest ladder hit,
    frequency-ordered (keep-all-branches law: no scoring until verify)."""
    for L in LADDER:
        if L < MIN_KEY or len(seq) < L:
            continue
        e = _lad[L].get(tuple(seq[-L:]))
        if e:
            conts = [c for c, _ in e.most_common(k)]
            return [list(c[:draft_length]) for c in conts if len(c[:draft_length]) >= MIN_KEY]
    return []


def propose(seq, draft_length):
    for L in LADDER:
        if L < MIN_KEY or len(seq) < L:
            continue
        e = _lad[L].get(tuple(seq[-L:]))
        if e:
            if MODE == "tree":
                ch = _tree_spine(e, draft_length)
                if ch and len(ch) >= MIN_KEY:
                    _calls["hit"] += 1
                    return ch
                continue
            if EV:
                ch = _ev_chain(e, draft_length)
                if ch is None:
                    _calls["gate"] += 1
                    continue
                _calls["hit"] += 1
                return ch
            best = e.most_common(1)[0][0]
            if len(best) >= MIN_KEY:
                _calls["hit"] += 1
                return list(best)[:draft_length]
    _calls["miss"] += 1
    return None



_LAST_DUMP_FP = None


def _dump():
    if not STORE_PATH:
        return
    try:
        tmp = STORE_PATH + ".tmp"
        # Chunked dump (wedged-prod incident 2026-10-10 ~20:0x): snapshotting
        # the WHOLE store under one lock hold stalled decode for minutes as
        # the store grew. One level per lock acquisition, released between
        # levels, so the worst decode stall is ONE level copy (~100 ms at
        # cap). Pickling stays outside every lock hold; a counter mutated
        # mid-pickle simply fails THIS cycle (caught), never the thread.
        # heartbeat guard (monitor:CM-19): an unchanged store dumped on a cadence
        # is a heartbeat, not information. Fingerprint = counters + per-level key
        # totals; O(1)-ish, taken outside any long hold.
        fp = (tuple(sorted(_calls.items())),
              tuple(sorted((L, len(_lad[L])) for L in _lad)))
        global _LAST_DUMP_FP
        if _LAST_DUMP_FP == fp and os.path.exists(STORE_PATH):
            return
        snap = {"lad": {}}
        BATCH = 5000
        for L in list(_lad):
            lvl = {}
            snap["lad"][L] = lvl
            with _lock:
                keys = list(_lad[L])          # one short hold: refs only
            for s in range(0, len(keys), BATCH):
                with _lock:
                    for k in keys[s:s + BATCH]:
                        c = _lad[L].get(k)
                        if c:
                            lvl[k] = dict(c)
                time.sleep(0.001)              # between batches: decode wins
        # Thread-only dump. NO FORK: os.fork() of a CUDA-initialized process
        # with 241 GB virtual address space stalls the parent (and corrupts the
        # child) — that is what wedged prod a second time. Pickling this
        # private `snap` copy on the dumper thread measured <=78 ms worst-case
        # heartbeat gap at 2x the store size; the GIL is released on GPU waits,
        # so decode keeps running. The old wedge was holding _lock across the
        # whole-store copy, which this batched loop already removed.
        with open(tmp, "wb") as fh:
            pickle.dump(snap, fh, protocol=4)
        os.replace(tmp, STORE_PATH)
        _LAST_DUMP_FP = fp
    except Exception:
        pass

def _load():
    if STORE_PATH and os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "rb") as f:
                lad = pickle.load(f)["lad"]
            for L in LADDER:
                for k, c in lad.get(str(L), lad.get(L, {})).items():
                    _lad[L].setdefault(tuple(k), Counter()).update(c)
            _ui(f"[GS] loaded store: "
                  f"{ {L: len(d) for L, d in _lad.items()} }")
        except Exception as ex:
            _ui(f"[GS] load failed: {ex}")

def _dumper():
    while True:
        time.sleep(DUMP_SECS)
        with _lock:
            _dump()

def install():
    if not ENABLE:
        return
    import torch
    from exllamav3.generator import job as _job
    orig = _job.Job.get_ngram_draft

    def patched(self, draft_length):
        if sum(_calls.values()) and sum(_calls.values()) % 20 == 0:
            _ui(f"[GS] stats {_calls}")
        _calls["calls"] += 1
        if _ACCEP:
            try:
                self.generator.record_draft_stats = True
                if self.draft_stats:
                    _w, _ac = self.draft_stats[-1][1], self.draft_stats[-1][2]
                    _calls["acc_w"] += _w; _calls["acc_a"] += _ac; _calls["acc_n"] += 1
            except Exception:
                _calls["error"] += 1
        try:
            toks = getattr(patched, "_toks", None)
            if toks is None:
                toks = []
                mdl = self.generator.model
                for c in (mdl, getattr(mdl, "cache", None)):
                    if c is None:
                        continue
                    tj = getattr(c, "tokenizer", None)
                    if tj is None:
                        env = getattr(getattr(c, "jinja_env", None),
                                      "template_ctx", None)
                        env = getattr(env, "environment", None) if env else None
                        tj = getattr(env, "tokenizer", None) if env else None
                    if tj is not None and tj not in toks:
                        toks.append(tj)
                patched._toks = toks
            im = -1
            for t in toks:
                for tt in (t, getattr(t, "tokenizer", None)):
                    if tt is None:
                        continue
                    fn = getattr(tt, "convert_tokens_to_ids", None)
                    if fn:
                        try:
                            v = fn("<|im_start|>")
                        except Exception:
                            continue
                        if isinstance(v, int) and v >= 0:
                            im = v; break
                if im >= 0:
                    break
            spec = set()
            for t in toks:
                tt = getattr(t, "tokenizer", t)
                ids = getattr(tt, "all_special_ids", None)
                if ids:
                    spec.update(int(x) for x in ids)
            spec.add(im)
            patched._spec = frozenset(spec)
            t = self.sequences[0].sequence_ids.torch()[0]
            n = int(t.shape[0])
            jid = id(self)
            w = _wm.get(jid, 0)
            if n < w:
                w = 0
            if n == w:
                # no new tokens (rare): propose straight off the tail window
                seq = t[-LADDER[0]:].tolist() if n else []
                with _lock:
                    cand = propose(seq, draft_length) if seq else None
            else:
                lo = max(1, w - LADDER[0] + 1)
                if w:
                    base = max(0, lo - LADDER[0])
                else:
                    base = max(0, n - WINDOW)
                seq = t[base:].tolist()
                with _lock:
                    _ingest(seq, spec, jid=jid, base=base, n_total=n)
                    cand = propose(seq, draft_length)
                    if ENGINE_TREE_T > 0:
                        try:
                            self._gs_tree = propose_tree(
                                seq, draft_length, budget=ENGINE_TREE_T)
                        except Exception:
                            self._gs_tree = None
            if cand:
                _HITFLAG[0] = 1
                row = torch.tensor([cand], dtype=t.dtype, device=t.device)
                U = _union_for(jid)   # ONE draw per hit: calling the toggle
                _k = ("u" if U else "d") + "_len"   # twice per hit anti-locked
                _calls[_k + "_n"] += 1   # length counted at the RETURN point
                if U and len(cand) < draft_length:
                    _calls["u_try"] += 1
                    # L125: union not dispatch. Store hit but a shorter chain
                    # than the window allows: ask the STOCK session-local
                    # drafter and take the LONGER chain (ties stay with the
                    # store - it has cross-request evidence). Both are greedy
                    # drafts; verify accepts only true prefixes either way.
                    try:
                        loc = orig(self, draft_length)
                        if loc is not None and loc.shape[-1] > row.shape[-1]:
                            _calls["u_win"] += 1
                            _calls[_k] += int(loc.shape[-1])
                            return loc
                    except Exception:
                        pass
                _calls[_k] += len(cand)
                return row
        except Exception as e:
            _calls["error"] += 1
            if _calls["error"] <= 3:
                try:
                    import traceback
                    with open("/tmp/gs_error.log", "a") as fh:
                        fh.write(f"#{_calls['calls']} {type(e).__name__}: {e}\n")
                        fh.write(traceback.format_exc())
                except Exception:
                    pass
        return orig(self, draft_length)   # stock session-local fallback

    _job.Job.get_ngram_draft = patched

    _load()
    atexit.register(_dump)
    import signal as _sig
    def _onterm(_s, _f):
        try:
            _dump()
        except Exception:
            pass
        prev = _sig.getsignal(_sig.SIGTERM)
        if callable(prev):
            prev(_s, _f)          # chain (uvicorn's own graceful shutdown)
        else:
            _sig.signal(_sig.SIGTERM, _sig.SIG_DFL)
            _sig.kill(_sig.os.getpid(), _sig.SIGTERM)  # re-raise default
    try:
        _sig.signal(_sig.SIGTERM, _onterm)
    except ValueError:
        pass  # not main thread - atexit still covers graceful exit
    # Periodic dumps OFF by default (2026-10-10): the launcher takes over with
    # kill -9, which never runs the exit hook — so the periodic file is stale
    # anyway, and the dump machinery itself wedged prod twice. Save happens on
    # any GRACEFUL stop; GS_DUMP_SECS>0 opts back into the periodic thread.
    if DUMP_SECS > 0:
        threading.Thread(target=_dumper, daemon=True).start()

    def _watchdog():
        # Activation assertion (lesson 2026-10-10: prod ran draft_mode=mtp for
        # hours, which bypasses get_ngram_draft -> GS silently idle). Loud log
        # if the hook path is dead after 120s of live traffic.
        time.sleep(120)
        if _calls["calls"] == 0:
            _ui("[GS] WARNING: 0 calls in 120s — the ngram drafter is not "
                  "being used (draft_mode is not 'ngram'). Global store is "
                  "INERT. Set draft_mode: ngram (launcher does this when "
                  "GS_ENABLE=1).")
        else:
            _ui(f"[GS] watchdog: {dict(_calls)}")
    threading.Thread(target=_watchdog, daemon=True).start()
    _ui("[GS] v3 code loaded")
    _ui(f"[GS] global store installed (ladder {LADDER}, min_key {MIN_KEY}, "
          f"topk {TOPK}, dump {DUMP_SECS}s -> {STORE_PATH or 'off'})")
