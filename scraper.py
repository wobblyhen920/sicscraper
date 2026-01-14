#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scraper ScuolaInChiaro / UNICA (v3)

Cosa produce (oltre a quanto già faceva v2):
- anagrafica_base_wide (sheet + CSV in outdir)
- observations_semantic (sheet(s) + CSV in outdir)

Note:
- Nessuno slug: {NOME_SCUOLA} viene URL-encoded integralmente.
- I JSON completi vengono salvati su disco in outdir/raw/<CODICE_SCUOLA>/<endpoint>.json.
  Le trasformazioni "semantic" leggono SEMPRE da saved_path (quindi non soffrono del truncation per Excel).
"""

import argparse
import asyncio
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import aiohttp
import pandas as pd


SCRIPT_VERSION = "4.1"

# =============== CONFIG ENDPOINTS ===============
# Nota: ho deduplicato "distribuzione votazioni esame" (era ripetuta).
# Nota: nel tuo elenco "assenze ata" puntava a "assenze-docenti": qui la tengo fedele.
ENDPOINTS: List[Tuple[str, str]] = [
    ("anagrafica_base", "https://unica.istruzione.gov.it/services/sic/api/v1.0/ricerche/ricercaRapida?chiaviDiRicerca={CODICE_SCUOLA}&numeroElementiPagina=5000&numeroPagina=1"),

    ("numero_alunni_23_24", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/andamento-alunni/"),

    ("esiti_giugno_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/esiti-giugno"),
    ("esiti_giugno_settembre_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/esiti-giugno-settembre"),

    ("sospesi_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/sospesi/"),

    ("diplomati_esaminati_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/diplomati-esaminati/"),
    ("distribuzione_votazioni_esame_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/distribuzione-votazioni-esame/"),

    ("abbandoni_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/abbandoni"),
    ("trasferimenti_24_25", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/trasferimenti"),
    ("studenti_ripetenti", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/studenti-ripetenti"),

    ("rav_24c5", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/rav-24c5"),
    ("rav_24c1", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/rav-24c1"),
    ("rav_24c2_II", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/rav-24c2?i=II"),
    ("rav_24c3_II", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/rav-24c3?i=II"),

    ("immatricolati_universita", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/immatricolati-universita/"),
    ("immatricolati_universita_area", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/alunni/grafici/immatricolati-universita-area-didattica/"),

    ("docenti_fasce_eta", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/docenti-fasce-eta"),
    ("docenti_trasferiti", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/docenti-trasferiti"),
    ("docenti_pensionati", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/docenti-pensionati"),

    ("assenze_docenti", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/assenze-docenti"),
    ("assenze_ata", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/personale/grafici/assenze-docenti"),

    ("entrate_fonti_finanziamento", "https://unica.istruzione.gov.it/cercalatuascuola/istituti/{CODICE_SCUOLA}/{NOME_SCUOLA}/finanza/grafici/entrate-fonti-finanziamento"),
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Python aiohttp",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "it,en-US;q=0.7,en;q=0.3",
}

EXCEL_CELL_SAFE_MAX = 32000  # margine sotto 32767
EXCEL_ROW_LIMIT = 1_048_576
OBS_SHEET_CHUNK = 900_000


# =============== INPUT CSV (semplice) ===============
def read_schools_csv(path: Path, sep: str) -> pd.DataFrame:
    """
    Lettura semplice: CSV standard con header.
    - Richiede colonne: CODICE_SCUOLA, denominazione
    - Usa sep (default ';')
    - Fallback minimale: se il parser C fallisce, usa engine='python' e salta le righe problematiche.
    """
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    except Exception:
        # fallback minimale per CSV sporchi (non "ripara": salta le righe che rompono il parsing)
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")

    # Normalizza header: rimuove BOM e spazi
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    if "CODICE_SCUOLA" not in df.columns or "denominazione" not in df.columns:
        raise SystemExit("Header senza CODICE_SCUOLA e/o denominazione")

    out = df[["CODICE_SCUOLA", "denominazione"]].copy()
    out = out.rename(columns={"denominazione": "NOME_SCUOLA"})
    out["CODICE_SCUOLA"] = out["CODICE_SCUOLA"].astype(str).str.strip()
    out["NOME_SCUOLA"] = out["NOME_SCUOLA"].astype(str).str.strip()
    out = out[out["CODICE_SCUOLA"] != ""]
    out = out.drop_duplicates(subset=["CODICE_SCUOLA"])
    return out

# =============== FETCH ===============

@dataclass
class ResultRow:
    CODICE_SCUOLA: str
    NOME_SCUOLA: str
    endpoint_key: str
    url: str
    final_url: str
    status: int
    ok: bool
    content_type: str
    is_json: bool
    raw_len: int
    saved_path: str
    raw_excel: str
    error: str
    from_cache: bool

def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (s or "").strip())
    return s[:180] if s else "x"

def looks_like_json(content_type: str, text: str) -> bool:
    ct = (content_type or "").lower()
    if "json" in ct:
        return True
    t = (text or "").lstrip()
    return t.startswith("{") or t.startswith("[")

async def fetch_text(
    session: aiohttp.ClientSession,
    url: str,
    timeout_s: int,
    retries: int,
    backoff_s: float,
) -> Tuple[int, str, str, str]:
    """
    Return: (status, final_url, content_type, text)
    status -1 per eccezioni non HTTP.
    """
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with session.get(url, timeout=timeout, allow_redirects=True) as resp:
                text = await resp.text(errors="replace")
                status = resp.status
                final_url = str(resp.url)
                content_type = resp.headers.get("Content-Type", "")
                if status in (429, 500, 502, 503, 504):
                    await asyncio.sleep(backoff_s * attempt)
                    continue
                return status, final_url, content_type, text
        except Exception as e:
            last_err = str(e)
            await asyncio.sleep(backoff_s * attempt)

    return -1, url, "", f'{{"_error":"failed","_last_exception":{json.dumps(last_err)}}}'

def build_url(template: str, codice: str, nome_scuola: str) -> str:
    if "{NOME_SCUOLA}" not in template:
        return template.format(CODICE_SCUOLA=codice)
    nome_enc = quote((nome_scuola or "").strip(), safe="")
    return template.format(CODICE_SCUOLA=codice, NOME_SCUOLA=nome_enc)

async def fetch_one_endpoint(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    codice: str,
    nome: str,
    endpoint_key: str,
    template: str,
    out_raw_dir: Path,
    timeout_s: int,
    retries: int,
    backoff_s: float,
    skip_existing: bool,
) -> ResultRow:
    async with sem:
        url = build_url(template, codice, nome)

        # Cache su disco: se il payload esiste già, non rifà la richiesta
        school_dir = out_raw_dir / safe_filename(codice)
        school_dir.mkdir(parents=True, exist_ok=True)

        cached_json = school_dir / f"{safe_filename(endpoint_key)}.json"
        cached_txt = school_dir / f"{safe_filename(endpoint_key)}.txt"
        cached_path = cached_json if cached_json.exists() else (cached_txt if cached_txt.exists() else None)

        if skip_existing and cached_path is not None and cached_path.stat().st_size > 0:
            raw_to_save = cached_path.read_text(encoding="utf-8", errors="replace")
            is_json = cached_path.suffix.lower() == ".json" or looks_like_json("application/json", raw_to_save)

            # se è JSON ma non parseabile, ignora cache e prova a riscaricare
            if is_json:
                try:
                    obj = json.loads(raw_to_save)
                    raw_to_save = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    cached_path = None  # forza fetch sotto

            if cached_path is not None:
                content_type = "application/json" if is_json else "text/plain"
                raw_excel = raw_to_save
                if len(raw_excel) > EXCEL_CELL_SAFE_MAX:
                    raw_excel = raw_excel[:EXCEL_CELL_SAFE_MAX] + "…[TRUNCATED]"

                return ResultRow(
                    CODICE_SCUOLA=codice,
                    NOME_SCUOLA=nome,
                    endpoint_key=endpoint_key,
                    url=url,
                    final_url=url,
                    status=200,
                    ok=True,
                    content_type=content_type,
                    is_json=is_json,
                    raw_len=len(raw_to_save),
                    saved_path=str(cached_path),
                    raw_excel=raw_excel,
                    error="cached",
                    from_cache=True,
                )

        status, final_url, content_type, text = await fetch_text(session, url, timeout_s, retries, backoff_s)

        ok = 200 <= status < 300
        is_json = looks_like_json(content_type, text)

        parsed_error = ""
        raw_to_save = text

        if is_json:
            try:
                obj = json.loads(text)
                raw_to_save = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            except Exception as e:
                parsed_error = f"json_parse_error: {e}"

        ext = "json" if is_json else "txt"
        fname = f"{safe_filename(endpoint_key)}.{ext}"
        fpath = school_dir / fname
        fpath.write_text(raw_to_save, encoding="utf-8", errors="replace")

        raw_excel = raw_to_save
        if len(raw_excel) > EXCEL_CELL_SAFE_MAX:
            raw_excel = raw_excel[:EXCEL_CELL_SAFE_MAX] + "…[TRUNCATED]"

        return ResultRow(
            CODICE_SCUOLA=codice,
            NOME_SCUOLA=nome,
            endpoint_key=endpoint_key,
            url=url,
            final_url=final_url,
            status=status,
            ok=ok,
            content_type=content_type,
            is_json=is_json,
            raw_len=len(raw_to_save),
            saved_path=str(fpath),
            raw_excel=raw_excel,
            error=parsed_error,
            from_cache=False,
        )


# =============== SEMANTIC EXTRACTION ===============
def load_json_from_saved_path(saved_path: str) -> Any:
    p = Path(saved_path)
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="replace")
    return json.loads(txt)

def build_anagrafica_base_wide(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Output: 1 riga per CODICE_SCUOLA, colonne = chiavi della prima entry in obj['scuole'].
    """
    rows: List[Dict[str, Any]] = []
    sub = df_res[(df_res["endpoint_key"] == "anagrafica_base") & (df_res["is_json"] == True) & (df_res["ok"] == True)]
    for r in sub.itertuples(index=False):
        obj = None
        try:
            obj = load_json_from_saved_path(r.saved_path)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        scuole = obj.get("scuole") or []
        if not scuole:
            continue
        s0 = scuole[0] or {}
        if not isinstance(s0, dict):
            continue

        row: Dict[str, Any] = {
            "CODICE_SCUOLA": r.CODICE_SCUOLA,
            "NOME_SCUOLA": r.NOME_SCUOLA,
        }
        for k, v in s0.items():
            row[str(k)] = v
        # conservativo: esito e paginazione se presenti
        if "esito" in obj:
            row["esito"] = obj.get("esito")
        if "numeroTotaleElementi" in obj:
            row["numeroTotaleElementi"] = obj.get("numeroTotaleElementi")
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.drop_duplicates(subset=["CODICE_SCUOLA"])
    return df_out

def build_observations_semantic(df_res: pd.DataFrame, include_kind: bool = True) -> pd.DataFrame:
    """
    Scioglie i JSON 'grafici' in dataset long:
    - caso A (categories presenti + series.data list): una riga per (series_name, category, value)
    - caso B (categories assenti + series con name + valore scalare): una riga per (category=series.name, value)
    - caso C (categories assenti + series.data list): una riga per index-only (category_index, value)

    Non inventa label quando non ci sono.
    """
    out_rows: List[Dict[str, Any]] = []
    sub = df_res[(df_res["endpoint_key"] != "anagrafica_base") & (df_res["is_json"] == True) & (df_res["ok"] == True)]

    for r in sub.itertuples(index=False):
        try:
            obj = load_json_from_saved_path(r.saved_path)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        title = obj.get("title")
        info = obj.get("info")
        note = obj.get("note")
        categories = obj.get("categories")
        series = obj.get("series")

        if not isinstance(series, list) or not series:
            continue

        # helper
        def s_data_is_list(s: Any) -> bool:
            return isinstance(s, dict) and isinstance(s.get("data"), list)

        any_list_data = any(s_data_is_list(s) for s in series)
        all_scalar = all(isinstance(s, dict) and not isinstance(s.get("data"), list) for s in series)

        # A) categories presenti + series.data list
        if isinstance(categories, list) and any_list_data:
            kind = "by_categories"
            for si, s in enumerate(series):
                if not isinstance(s, dict):
                    continue
                sname = s.get("name")
                data = s.get("data")
                if not isinstance(data, list):
                    continue
                for ci, val in enumerate(data):
                    cat = categories[ci] if ci < len(categories) else None
                    row = {
                        "CODICE_SCUOLA": r.CODICE_SCUOLA,
                        "NOME_SCUOLA": r.NOME_SCUOLA,
                        "endpoint_key": r.endpoint_key,
                        "title": title,
                        "series_index": si,
                        "series_name": sname,
                        "category_index": ci,
                        "category": cat,
                        "value": val,
                        "info": info,
                        "note": note,
                    }
                    if include_kind:
                        row["kind"] = kind
                    out_rows.append(row)

        # B) categories assenti + pie-like: series[].name + scalar data
        elif categories is None and all_scalar:
            kind = "by_name_scalar"
            for si, s in enumerate(series):
                if not isinstance(s, dict):
                    continue
                cat = s.get("name")
                val = s.get("data") if "data" in s else s.get("y")
                row = {
                    "CODICE_SCUOLA": r.CODICE_SCUOLA,
                    "NOME_SCUOLA": r.NOME_SCUOLA,
                    "endpoint_key": r.endpoint_key,
                    "title": title,
                    "series_index": None,
                    "series_name": None,
                    "category_index": si,
                    "category": cat,
                    "value": val,
                    "info": info,
                    "note": note,
                }
                if include_kind:
                    row["kind"] = kind
                out_rows.append(row)

        # C) categories assenti + series.data list (index-only)
        elif categories is None and any_list_data:
            kind = "index_only"
            for si, s in enumerate(series):
                if not isinstance(s, dict):
                    continue
                sname = s.get("name")
                data = s.get("data")
                if not isinstance(data, list):
                    continue
                for ci, val in enumerate(data):
                    row = {
                        "CODICE_SCUOLA": r.CODICE_SCUOLA,
                        "NOME_SCUOLA": r.NOME_SCUOLA,
                        "endpoint_key": r.endpoint_key,
                        "title": title,
                        "series_index": si,
                        "series_name": sname,
                        "category_index": ci,
                        "category": None,
                        "value": val,
                        "info": info,
                        "note": note,
                    }
                    if include_kind:
                        row["kind"] = kind
                    out_rows.append(row)

        else:
            # Non c'è abbastanza struttura per sciogliere senza inventare semantica.
            continue

    return pd.DataFrame(out_rows)


def write_df_to_excel_chunked(w: pd.ExcelWriter, df: pd.DataFrame, base_sheet: str, chunk_size: int = OBS_SHEET_CHUNK) -> None:
    if df.empty:
        df.to_excel(w, index=False, sheet_name=base_sheet[:31])
        return
    if len(df) <= EXCEL_ROW_LIMIT:
        df.to_excel(w, index=False, sheet_name=base_sheet[:31])
        return

    # chunk
    for i in range(0, len(df), chunk_size):
        part = df.iloc[i:i + chunk_size]
        sheet = f"{base_sheet}_{i//chunk_size+1}"
        part.to_excel(w, index=False, sheet_name=sheet[:31])


# =============== MAIN ===============
async def run(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.rawdir) if args.rawdir else (outdir / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Endpoint selection
    endpoints_to_use = ENDPOINTS
    if args.endpoints:
        wanted = [x.strip() for x in str(args.endpoints).split(",") if x.strip()]
        wanted_set = set(wanted)
        all_keys = [k for k, _ in ENDPOINTS]
        unknown = sorted(wanted_set.difference(all_keys))
        if unknown:
            raise SystemExit(f"Endpoint sconosciuti: {unknown}. Disponibili: {all_keys}")
        endpoints_to_use = [(k, u) for (k, u) in ENDPOINTS if k in wanted_set]

    df_schools = read_schools_csv(inp, sep=args.sep)
    if args.limit and args.limit > 0:
        df_schools = df_schools.head(args.limit)

    schools = df_schools.to_dict(orient="records")
    sem = asyncio.Semaphore(args.concurrency)

    connector = aiohttp.TCPConnector(limit=0, ssl=False)
    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, connector=connector) as session:
        tasks = []
        for s in schools:
            codice = str(s["CODICE_SCUOLA"]).strip()
            nome = str(s["NOME_SCUOLA"]).strip()
            if not codice:
                continue
            for endpoint_key, template in endpoints_to_use:
                tasks.append(
                    fetch_one_endpoint(
                        sem=sem,
                        session=session,
                        codice=codice,
                        nome=nome,
                        endpoint_key=endpoint_key,
                        template=template,
                        out_raw_dir=raw_dir,
                        timeout_s=args.timeout,
                        retries=args.retries,
                        backoff_s=args.backoff,
                        skip_existing=args.skip_existing,
                    )
                )

        results: List[ResultRow] = []
        total = len(tasks)
        done = 0
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            done += 1
            if args.progress_every and done % args.progress_every == 0:
                print(f"[{done}/{total}]")

    # dataframe long
    df_res = pd.DataFrame([r.__dict__ for r in results])

    # summary
    df_sum = (
        df_res.groupby(["endpoint_key", "status", "ok"])
        .size()
        .reset_index(name="n")
        .sort_values(["endpoint_key", "ok", "status"], ascending=[True, False, True])
    )

    # jsonl backend-friendly
    jsonl_path = outdir / "results_long.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    # ===== NEW: semantic outputs =====
    df_anag = pd.DataFrame()
    df_obs = pd.DataFrame()
    if args.semantic:
        print("[semantic] building anagrafica_base_wide + observations_semantic ...")
        df_anag = build_anagrafica_base_wide(df_res)
        df_obs = build_observations_semantic(df_res, include_kind=args.include_kind)

        # scrivi anche CSV (backend-friendly)
        anag_csv = outdir / "anagrafica_base_wide.csv"
        obs_csv = outdir / "observations_semantic.csv"
        df_anag.to_csv(anag_csv, index=False)
        df_obs.to_csv(obs_csv, index=False)
        print("[semantic] CSV:", str(anag_csv))
        print("[semantic] CSV:", str(obs_csv))

    # excel
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df_schools.to_excel(w, index=False, sheet_name="schools")
        df_res.to_excel(w, index=False, sheet_name="responses_long")
        df_sum.to_excel(w, index=False, sheet_name="summary")

        if args.wide:
            wide_raw = df_res.pivot_table(
                index=["CODICE_SCUOLA", "NOME_SCUOLA"],
                columns="endpoint_key",
                values="raw_excel",
                aggfunc="first",
            ).reset_index()

            wide_status = df_res.pivot_table(
                index=["CODICE_SCUOLA", "NOME_SCUOLA"],
                columns="endpoint_key",
                values="status",
                aggfunc="first",
            ).reset_index()

            wide_raw.to_excel(w, index=False, sheet_name="responses_wide_raw")
            wide_status.to_excel(w, index=False, sheet_name="responses_wide_status")

        if args.semantic:
            # anagrafica_base_wide: in genere sta sotto al limite righe/colonne
            write_df_to_excel_chunked(w, df_anag, "anagrafica_base_wide", chunk_size=OBS_SHEET_CHUNK)
            # observations: può crescere, quindi chunk
            write_df_to_excel_chunked(w, df_obs, "observations_semantic", chunk_size=OBS_SHEET_CHUNK)

            # mini stats: utile per QA
            if not df_obs.empty:
                cols = ["endpoint_key"] + (["kind"] if args.include_kind and "kind" in df_obs.columns else [])
                df_obs_stats = df_obs.groupby(cols).size().reset_index(name="n_rows").sort_values("n_rows", ascending=False)
                df_obs_stats.to_excel(w, index=False, sheet_name="observations_stats")

    print("OK")
    print("Excel:", str(out))
    print("JSONL:", str(jsonl_path))
    print("Raw dir:", str(raw_dir))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="input.csv")
    p.add_argument("--sep", default=";", help="separatore CSV input (default ;)")
    p.add_argument("--output", default="scuolainchiaro_dump.xlsx")
    p.add_argument("--outdir", default="out_scuolainchiaro")
    p.add_argument("--rawdir", default="", help="directory cache raw globale; se vuoto usa <outdir>/raw")
    p.add_argument("--endpoints", default="", help="lista endpoint_key separati da virgola; se vuoto usa tutti")
    p.add_argument("--list-endpoints", action="store_true", help="stampa gli endpoint disponibili ed esce")
    p.add_argument("--skip-existing", action="store_true", help="se in outdir/raw esiste già <CODICE_SCUOLA>/<endpoint>.json|.txt, non riscarica")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--backoff", type=float, default=0.6)
    p.add_argument("--wide", action="store_true", help="crea anche fogli wide (raw/status)")
    p.add_argument("--no-semantic", action="store_true", help="disabilita anagrafica_base_wide + observations_semantic")
    p.add_argument("--no-kind", action="store_true", help="rimuove la colonna kind da observations_semantic")
    p.add_argument("--progress-every", type=int, default=200, help="stampa avanzamento ogni N richieste (0 disabilita)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    # Normalizza stringhe vuote
    if not getattr(args, "rawdir", "").strip():
        args.rawdir = None
    if not getattr(args, "endpoints", "").strip():
        args.endpoints = None
    if getattr(args, "list_endpoints", False):
        for k, u in ENDPOINTS:
            print(f"{k}\t{u}")
        raise SystemExit(0)
    args.semantic = (not args.no_semantic)
    args.include_kind = (not args.no_kind)
    if args.progress_every <= 0:
        args.progress_every = None
    asyncio.run(run(args))
