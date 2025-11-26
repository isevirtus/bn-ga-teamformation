# filter_devs_by_graph.py
from pathlib import Path
import json, re, shutil

DEV_PATH   = Path("dados/data_base.json")
GRAPH_PATH = Path("dados/data_base_graph.json")
BACKUP     = DEV_PATH.with_suffix(".backup.json")

def _as_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x)
    m = re.search(r"(\d+)$", s)  # extrai 558 de "Dev558"
    return int(m.group(1)) if m else None

def carregar_ids_do_grafo(path: Path) -> set[int]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    # aceita {"edges":[...]}, {"links":[...]}, ou lista direta
    if isinstance(data, dict):
        edges = data.get("edges") or data.get("links") or []
    elif isinstance(data, list):
        edges = data
    else:
        edges = []

    ids = set()
    for e in edges:
        u = _as_int(e.get("source_user_id")) or _as_int(e.get("source"))
        v = _as_int(e.get("target_user_id")) or _as_int(e.get("target"))
        if u is not None: ids.add(int(u))
        if v is not None: ids.add(int(v))
    return ids

def carregar_devs(path: Path):
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw, "list"          # raiz é lista
    elif isinstance(raw, dict):
        devs = raw.get("developers", [])
        return devs, "dict"         # raiz é dict com "developers"
    else:
        return [], "unknown"

def salvar_devs(path: Path, devs, formato: str, original_raw=None):
    if formato == "list":
        out = devs
    elif formato == "dict":
        out = dict(original_raw) if isinstance(original_raw, dict) else {}
        out["developers"] = devs
    else:
        out = devs
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

def main():
    if not DEV_PATH.exists():
        raise FileNotFoundError(f"Não encontrei {DEV_PATH}")
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Não encontrei {GRAPH_PATH}")

    valid_ids = carregar_ids_do_grafo(GRAPH_PATH)
    print(f"[grafo] IDs presentes no grafo: {len(valid_ids)}")

    # carrega devs preservando formato original
    with DEV_PATH.open(encoding="utf-8") as f:
        original_raw = json.load(f)
    devs, formato = (original_raw, "list") if isinstance(original_raw, list) else (original_raw.get("developers", []), "dict")

    before = len(devs)
    # filtra: mantém só devs cujo id aparece no grafo
    def _did(d):
        try:    return int(d.get("id"))
        except: return d.get("id")
    devs_filtrados = [d for d in devs if _did(d) in valid_ids]
    after = len(devs_filtrados)

    # backup e grava
    shutil.copyfile(DEV_PATH, BACKUP)
    salvar_devs(DEV_PATH, devs_filtrados, formato, original_raw=original_raw)

    print(f"[ok] Filtrado: {before} -> {after}.")
    print(f"[backup] Cópia salva em: {BACKUP}")

if __name__ == "__main__":
    main()
