import json
import re
import sys
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# Optional deps
try:
    import altair as alt
except Exception:
    alt = None

try:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium
except Exception:
    folium = None
    Draw = None
    st_folium = None


# ======================
# CONFIG
# ======================
DEFAULT_SCRAPER_SCRIPT = "scuolainchiaro_scrape_v4_patched.py"  # supporta --endpoints e --rawdir
DEFAULT_REGISTRY = "input.csv"
DEFAULT_SEP = ";"

DATA_DIR = Path("data")
RAW_CACHE_DIR = DATA_DIR / "raw"      # cache condivisa
JOBS_DIR = DATA_DIR / "jobs"          # output per job

DATA_DIR.mkdir(exist_ok=True)
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# ENDPOINT CATALOG
# ======================
ENDPOINTS = [
    ("anagrafica_base", "Anagrafica base (ricercaRapida)"),
    ("numero_alunni_23_24", "Numero alunni A.S. 2023/24"),
    ("esiti_giugno_24_25", "Esiti giugno A.S. 2024/25"),
    ("esiti_giugno_settembre_24_25", "Esiti giugno+settembre A.S. 2024/25"),
    ("sospesi_24_25", "Sospesi (sospensione giudizio) A.S. 2024/25"),
    ("diplomati_esaminati_24_25", "Diplomati (% sugli esaminati) A.S. 2024/25"),
    ("distribuzione_votazioni_esame_24_25", "Distribuzione votazioni esame A.S. 2024/25"),
    ("abbandoni_24_25", "Abbandoni in corso d'anno A.S. 2024/25"),
    ("trasferimenti_24_25", "Trasferimenti ingresso/uscita A.S. 2024/25"),
    ("studenti_ripetenti", "Ripetenti (% frequentanti)"),
    ("rav_24c5", "RAV 24c5 (iscritti primo anno per voto esame 1° ciclo)"),
    ("rav_24c1", "RAV 24c1 (consiglio orientativo per tipologia)"),
    ("rav_24c2_II", "RAV 24c2 (corrispondenza consiglio/scelta)"),
    ("rav_24c3_II", "RAV 24c3 (ammessi al 2° anno che hanno seguito consiglio)"),
    ("immatricolati_universita", "Immatricolati università (risultati a distanza)"),
    ("immatricolati_universita_area", "Immatricolati università per area didattica"),
    ("docenti_fasce_eta", "Docenti per fasce d'età"),
    ("docenti_trasferiti", "Docenti trasferiti"),
    ("docenti_pensionati", "Docenti pensionati"),
    ("assenze_docenti", "Assenze docenti (giorni pro capite)"),
    ("assenze_ata", "Assenze ATA (attenzione: endpoint uguale a docenti nel tuo elenco)"),
    ("entrate_fonti_finanziamento", "Entrate per fonti di finanziamento"),
]
ENDPOINT_KEYS = [k for k, _ in ENDPOINTS]
ENDPOINT_LABEL = dict(ENDPOINTS)


# ======================
# UTILS
# ======================
def slug_safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", (s or "").strip())[:180] or "x"


def is_valid_code(c: str) -> bool:
    c = (c or "").strip()
    return bool(re.fullmatch(r"[A-Za-z0-9]{10}", c))


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def detect_col(df: pd.DataFrame, name: str) -> str | None:
    """Trova una colonna case-insensitive."""
    cols = {c.lower(): c for c in df.columns}
    return cols.get(name.lower())


@st.cache_data(show_spinner=False)
def load_registry_from_path(path: str, sep: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
    df = normalize_headers(df)

    if "CODICE_SCUOLA" not in df.columns or "denominazione" not in df.columns:
        raise ValueError("Il registry deve avere colonne CODICE_SCUOLA e denominazione")

    df["CODICE_SCUOLA"] = df["CODICE_SCUOLA"].astype(str).str.strip()
    df["denominazione"] = df["denominazione"].astype(str).str.strip()
    df = df[df["CODICE_SCUOLA"] != ""].drop_duplicates(subset=["CODICE_SCUOLA"])
    return df


@st.cache_data(show_spinner=False)
def load_registry_from_upload(content_bytes: bytes, sep: str) -> pd.DataFrame:
    import io
    data = io.BytesIO(content_bytes)
    try:
        df = pd.read_csv(data, sep=sep, dtype=str, keep_default_na=False)
    except Exception:
        data.seek(0)
        df = pd.read_csv(data, sep=sep, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
    df = normalize_headers(df)

    if "CODICE_SCUOLA" not in df.columns or "denominazione" not in df.columns:
        raise ValueError("Il registry deve avere colonne CODICE_SCUOLA e denominazione")

    df["CODICE_SCUOLA"] = df["CODICE_SCUOLA"].astype(str).str.strip()
    df["denominazione"] = df["denominazione"].astype(str).str.strip()
    df = df[df["CODICE_SCUOLA"] != ""].drop_duplicates(subset=["CODICE_SCUOLA"])
    return df


def parse_prefix_list(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"[,\s;]+", s) if p.strip()]
    return parts[:50]


def filter_registry(df: pd.DataFrame, provincia: str | None, comune: str | None, cap_prefix: str | None, q: str | None) -> pd.DataFrame:
    dfv = df.copy()

    col_prov = detect_col(dfv, "provincia")
    col_com = detect_col(dfv, "comune")
    col_cap = detect_col(dfv, "cap")

    if col_prov and provincia and provincia != "(tutte)":
        dfv = dfv[dfv[col_prov] == provincia]
    if col_com and comune and comune != "(tutti)":
        dfv = dfv[dfv[col_com] == comune]

    if col_cap and cap_prefix:
        prefs = parse_prefix_list(cap_prefix)
        if prefs:
            cap_series = dfv[col_cap].astype(str).str.strip()
            mask = False
            for p in prefs:
                mask = mask | cap_series.str.startswith(p)
            dfv = dfv[mask]

    if q and q.strip():
        qs = q.strip().lower()
        cols = ["denominazione"]
        if col_com:
            cols.append(col_com)
        if col_prov:
            cols.append(col_prov)
        if col_cap:
            cols.append(col_cap)
        mask = pd.Series(False, index=dfv.index)
        for c in cols:
            mask = mask | dfv[c].astype(str).str.lower().str.contains(qs, na=False)
        dfv = dfv[mask]

    return dfv


def find_cache_file(rawdir: Path, codice: str, endpoint_key: str) -> Path | None:
    base = rawdir / codice
    if not base.exists():
        return None
    for ext in ("json", "txt"):
        p = base / f"{endpoint_key}.{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def cache_status_table(rawdir: Path, codes: list[str], endpoints: list[str], stale_days: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc).timestamp()
    rows = []
    for c in codes:
        for ep in endpoints:
            p = find_cache_file(rawdir, c, ep)
            if not p:
                rows.append({"CODICE_SCUOLA": c, "endpoint_key": ep, "cache": "missing", "age_days": None, "path": ""})
                continue
            age_days = int((now - p.stat().st_mtime) / 86400)
            cache = "stale" if (stale_days is not None and age_days >= stale_days) else "cached"
            rows.append({"CODICE_SCUOLA": c, "endpoint_key": ep, "cache": cache, "age_days": age_days, "path": str(p)})
    return pd.DataFrame(rows)


def invalidate_cache(rawdir: Path, df_cache: pd.DataFrame, only_stale: bool, endpoints_scope: set[str], codes_scope: set[str]) -> int:
    n = 0
    for _, r in df_cache.iterrows():
        if r.get("CODICE_SCUOLA") not in codes_scope:
            continue
        if r.get("endpoint_key") not in endpoints_scope:
            continue
        if only_stale and r.get("cache") != "stale":
            continue
        p = r.get("path") or ""
        if not p:
            continue
        try:
            Path(p).unlink(missing_ok=True)
            n += 1
        except Exception:
            pass
    return n


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def run_scraper(
    scraper_script: str,
    input_csv: Path,
    outdir: Path,
    rawdir: Path,
    endpoints: list[str],
    sep: str,
    concurrency: int,
    timeout_s: int,
    retries: int,
    backoff_s: float,
    skip_existing: bool,
    no_kind: bool,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        scraper_script,
        "--input", str(input_csv),
        "--sep", sep,
        "--outdir", str(outdir),
        "--rawdir", str(rawdir),
        "--endpoints", ",".join(endpoints),
        "--wide",
        "--concurrency", str(concurrency),
        "--timeout", str(timeout_s),
        "--retries", str(retries),
        "--backoff", str(backoff_s),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if no_kind:
        cmd.append("--no-kind")

    return subprocess.run(cmd, capture_output=True, text=True)


def zip_job(job_dir: Path, zip_path: Path, include_raw: bool) -> None:
    allowed = {"meta.json", "stdout.txt", "stderr.txt"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in job_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(job_dir).as_posix()
            if rel in allowed:
                z.write(p, rel)
                continue
            if rel.startswith("out/"):
                z.write(p, rel)
                continue
            if rel.startswith("charts/"):
                z.write(p, rel)
                continue
            if include_raw and rel.startswith("raw_subset/"):
                z.write(p, rel)


def build_raw_subset(job_dir: Path, rawdir: Path, codes: list[str], endpoints: list[str]) -> Path:
    subset_dir = job_dir / "raw_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for c in codes:
        (subset_dir / c).mkdir(exist_ok=True)
        for ep in endpoints:
            p = find_cache_file(rawdir, c, ep)
            if not p:
                continue
            target = subset_dir / c / p.name
            try:
                target.write_bytes(p.read_bytes())
            except Exception:
                pass
    return subset_dir


def charts_from_semantic(obs_csv: Path, charts_dir: Path, mode: str, school_code: str | None, agg: str) -> list[Path]:
    if alt is None:
        return []
    if not obs_csv.exists():
        return []
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(obs_csv, dtype=str, keep_default_na=False)
    if df.empty:
        return []

    for col in ("value",):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["value"].notna()]
    if df.empty:
        return []

    out_files = []

    if mode == "single" and school_code:
        dfp = df[df["CODICE_SCUOLA"] == school_code].copy()
        if dfp.empty:
            return []
        title_prefix = f"{school_code} · "
    else:
        # aggregazione su scuole
        group_cols = ["endpoint_key", "title", "series_name", "category"]
        if agg == "sum":
            dfp = df.groupby(group_cols, dropna=False)["value"].sum().reset_index()
            title_prefix = "SUM · "
        else:
            dfp = df.groupby(group_cols, dropna=False)["value"].mean().reset_index()
            title_prefix = "MEAN · "

    # uno chart per endpoint_key
    for ep, dfe in dfp.groupby("endpoint_key"):
        if dfe["value"].nunique() <= 1:
            continue
        # category su x, value su y, series_name come colore se utile
        use_series = dfe["series_name"].nunique() > 1
        base = alt.Chart(dfe).mark_bar().encode(
            x=alt.X("category:N", sort="-y", title="Categoria"),
            y=alt.Y("value:Q", title="Valore"),
            tooltip=["endpoint_key", "title", "series_name", "category", "value"],
        )
        if use_series:
            base = base.encode(color=alt.Color("series_name:N", legend=alt.Legend(title="Serie")))

        chart = base.properties(
            title=f"{title_prefix}{ep} · {ENDPOINT_LABEL.get(ep, ep)}",
            width=900,
            height=420,
        ).interactive()

        out = charts_dir / f"{ep}.html"
        out.write_text(chart.to_html(), encoding="utf-8")
        out_files.append(out)

    return out_files


def get_query_params() -> dict:
    try:
        # streamlit >= 1.30
        return {k: (v if isinstance(v, str) else (v[0] if v else "")) for k, v in dict(st.query_params).items()}
    except Exception:
        qp = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) and v else "") for k, v in qp.items()}


def set_query_params(**kwargs) -> None:
    # Solo campi corti, niente liste lunghe di codici
    clean = {k: str(v) for k, v in kwargs.items() if v is not None and str(v).strip() != ""}
    try:
        st.query_params.clear()
        st.query_params.update(clean)
    except Exception:
        st.experimental_set_query_params(**clean)


# ======================
# PAGE
# ======================

st.set_page_config(page_title="ScuolaInChiaro Scraper", layout="wide")
st.title("ScuolaInChiaro · single-user")

# -----------------------
# Session state
# -----------------------
if "selected_codes" not in st.session_state:
    st.session_state["selected_codes"] = set()
if "var_selection" not in st.session_state:
    st.session_state["var_selection"] = {k: True for k, _ in ENDPOINTS}

# -----------------------
# Sidebar: settings (prudente) + debug in fondo
# -----------------------
with st.sidebar:
    st.subheader("Impostazioni")
    use_cache = st.checkbox("Usa cache globale", value=True, help="Riusa i payload in data/raw se già presenti.")
    drop_kind = st.checkbox("Togli colonna kind", value=False)

    st.markdown("**Runtime (prudente)**")
    concurrency = st.slider("Concurrency", min_value=1, max_value=8, value=4, step=1,
                            help="Troppa concorrenza aumenta 429 e instabilità. 4 è un buon default.")
    timeout_s = st.slider("Timeout (s)", min_value=15, max_value=90, value=45, step=5)
    retries = st.slider("Retries", min_value=0, max_value=4, value=2, step=1)
    backoff_s = st.slider("Backoff (s)", min_value=0.0, max_value=1.0, value=0.4, step=0.1)

    st.markdown("**Cache**")
    stale_days = st.slider("Considera 'stale' dopo (giorni)", min_value=7, max_value=365, value=60, step=1)
    include_raw_in_zip = st.checkbox("Includi raw nel job.zip", value=False)

    st.markdown("**Grafici**")
    make_charts = st.checkbox("Crea grafici (HTML) dalle variabili selezionate", value=False)
    chart_mode = st.selectbox("Modalità grafici", options=["Una scuola", "Media sulle scuole selezionate"], index=0)

    st.divider()

    st.markdown("**[Debug] Script scraper**")
    scraper_script = st.text_input("", value=DEFAULT_SCRAPER_SCRIPT)

    st.markdown("**[Debug] Registro input delle scuole**")
    sep = st.text_input("Separatore CSV", value=DEFAULT_SEP)
    upload = st.file_uploader("Carica CSV (opzionale)", type=["csv"], accept_multiple_files=False)
    registry_path = st.text_input("Path CSV (se non carichi un file)", value=DEFAULT_REGISTRY)

# -----------------------
# Registry load
# -----------------------
reg = None
if upload is not None:
    try:
        reg = load_registry_from_upload(upload.getvalue(), sep)
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    if not Path(registry_path).exists():
        st.error(f"File non trovato: {registry_path}")
        st.stop()
    try:
        reg = load_registry(registry_path, sep)
    except Exception as e:
        st.error(str(e))
        st.stop()

# Columns that may exist
col_prov = detect_col(reg, "provincia")
col_com = detect_col(reg, "comune")
col_cap = detect_col(reg, "cap")
col_lat = detect_col(reg, "lat") or detect_col(reg, "latitude")
col_lon = detect_col(reg, "lon") or detect_col(reg, "lng") or detect_col(reg, "longitude")

# -----------------------
# Tabs: guided journey
# -----------------------
tab_scuole, tab_variabili, tab_run, tab_jobs = st.tabs(
    ["1) Scuole", "2) Variabili", "3) Esecuzione", "4) Job e cache"]
)

# -----------------------
# 1) Scuole
# -----------------------
with tab_scuole:
    st.subheader("Passo 1 · Seleziona le scuole")
    st.caption("Puoi selezionare dalla tabella. La mappa è opzionale e serve per selezione spaziale. "
               "La selezione è condivisa tra tabella e mappa.")

    # Filters
    f1, f2, f3, f4 = st.columns([1, 1, 1, 2], gap="small")
    with f1:
        prov_opt = ["(tutte)"]
        if col_prov:
            prov_opt += sorted([x for x in reg[col_prov].astype(str).unique().tolist() if x.strip() != ""])
        prov = st.selectbox("Provincia", options=prov_opt, index=0, disabled=(col_prov is None))
    with f2:
        com_opt = ["(tutti)"]
        if col_com:
            com_opt += sorted([x for x in reg[col_com].astype(str).unique().tolist() if x.strip() != ""])
        com = st.selectbox("Comune", options=com_opt, index=0, disabled=(col_com is None))
    with f3:
        cap_q = st.text_input("CAP (testo breve)", value="", help="Prefisso o lista separata da virgole. Esempio: 001,004")
    with f4:
        q = st.text_input("Ricerca testo", value="", placeholder="Denominazione / Comune / Provincia")

    df_view = reg.copy()
    if col_prov and prov != "(tutte)":
        df_view = df_view[df_view[col_prov].astype(str) == str(prov)]
    if col_com and com != "(tutti)":
        df_view = df_view[df_view[col_com].astype(str) == str(com)]
    if col_cap and cap_q.strip():
        tokens = [t.strip() for t in cap_q.split(",") if t.strip()]
        cap_series = df_view[col_cap].astype(str).str.replace(r"\D+", "", regex=True)
        mask_cap = False
        for t in tokens:
            t_clean = re.sub(r"\D+", "", t)
            if not t_clean:
                continue
            mask_cap = mask_cap | cap_series.str.startswith(t_clean)
        df_view = df_view[mask_cap] if isinstance(mask_cap, pd.Series) else df_view

    if q.strip():
        qs = q.strip().lower()
        cols = ["denominazione"]
        if col_com:
            cols.append(col_com)
        if col_prov:
            cols.append(col_prov)
        if col_cap:
            cols.append(col_cap)
        mask = pd.Series(False, index=df_view.index)
        for c in cols:
            mask = mask | df_view[c].astype(str).str.lower().str.contains(qs, na=False)
        df_view = df_view[mask]

    # Table selection
    view_cols = ["CODICE_SCUOLA", "denominazione"]
    for c in [col_com, col_prov, col_cap]:
        if c and c not in view_cols:
            view_cols.append(c)

    df_show = df_view[view_cols].copy()
    df_show.insert(0, "Seleziona", df_show["CODICE_SCUOLA"].isin(st.session_state["selected_codes"]))

    colT1, colT2 = st.columns([1, 3], gap="small")
    with colT1:
        if st.button("Seleziona tutte (visibili)"):
            st.session_state["selected_codes"] |= set(df_show["CODICE_SCUOLA"].tolist())
        if st.button("Deseleziona tutte (visibili)"):
            st.session_state["selected_codes"] -= set(df_show["CODICE_SCUOLA"].tolist())

    with colT2:
        st.write(f"Scuole visibili: {len(df_show)} · Selezionate: {len(st.session_state['selected_codes'])}")

    edited = st.data_editor(
        df_show.head(5000),
        use_container_width=True,
        hide_index=True,
        column_config={"Seleziona": st.column_config.CheckboxColumn(required=False)},
        disabled=[c for c in df_show.columns if c != "Seleziona"],
        key="schools_editor",
    )

    # Sync selection from editor
    try:
        selected_now = set(edited.loc[edited["Seleziona"] == True, "CODICE_SCUOLA"].tolist())
        visible_codes = set(edited["CODICE_SCUOLA"].tolist())
        # update for visible
        st.session_state["selected_codes"] -= visible_codes
        st.session_state["selected_codes"] |= selected_now
    except Exception:
        pass

    # Manual paste codes
    manual_codes = st.text_area("Oppure incolla codici (uno per riga)", height=90, placeholder="RMIC....\nRMPS....")
    if manual_codes.strip():
        extra_codes = [x.strip() for x in manual_codes.splitlines() if x.strip()]
        extra_codes = [x for x in extra_codes if is_valid_code(x)]
        st.session_state["selected_codes"] |= set(extra_codes)

    # Optional map
    with st.expander("Mappa (opzionale) · Selezione spaziale", expanded=False):
        if folium is None or st_folium is None:
            st.warning("Mappa non disponibile: installa folium e streamlit-folium.")
        elif not (col_lat and col_lon):
            st.warning("Nel registro mancano lat/lon: la mappa richiede colonne lat e lon.")
        else:
            mdf = df_view.copy()
            # keep only valid coords
            mdf = mdf[mdf[col_lat].astype(str).str.strip() != ""]
            mdf = mdf[mdf[col_lon].astype(str).str.strip() != ""]
            try:
                mdf[col_lat] = mdf[col_lat].astype(float)
                mdf[col_lon] = mdf[col_lon].astype(float)
            except Exception:
                st.warning("lat/lon non numerici nel registro: pulisci le colonne.")
                mdf = mdf.iloc[0:0]

            if len(mdf) == 0:
                st.info("Nessuna scuola con coordinate nel filtro corrente.")
            else:
                # Center
                center_lat = float(mdf[col_lat].mean())
                center_lon = float(mdf[col_lon].mean())
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, control_scale=True)

                # Markers
                for _, r in mdf.iterrows():
                    code = r["CODICE_SCUOLA"]
                    name = r["denominazione"]
                    is_sel = code in st.session_state["selected_codes"]
                    popup = folium.Popup(f"{code}<br>{name}", max_width=280)
                    folium.CircleMarker(
                        location=[float(r[col_lat]), float(r[col_lon])],
                        radius=6 if is_sel else 4,
                        fill=True,
                        popup=popup,
                    ).add_to(m)

                out = st_folium(m, height=520, use_container_width=True)

                # Click selection (best-effort: popups don't return code reliably)
                st.caption("Nota: la selezione via click su marker è limitata da streamlit-folium. "
                           "Usa la tabella per selezioni precise.")

# -----------------------
# 2) Variabili (endpoint)
# -----------------------
with tab_variabili:
    st.subheader("Passo 2 · Scegli le variabili")
    st.caption("Queste variabili determinano quali dati vengono scaricati. "
               "Anagrafica base è obbligatoria.")

    colV1, colV2 = st.columns([1, 3], gap="large")
    with colV1:
        if st.button("Seleziona tutte"):
            for k, _ in ENDPOINTS:
                st.session_state["var_selection"][k] = True
        if st.button("Deseleziona tutte (tranne anagrafica)"):
            for k, _ in ENDPOINTS:
                st.session_state["var_selection"][k] = (k == "anagrafica_base")

    st.markdown("**Variabili**")
    for k, title in ENDPOINTS:
        if k == "anagrafica_base":
            st.checkbox(f"{title} (obbligatoria)", value=True, disabled=True, key=f"var_{k}")
            st.session_state["var_selection"][k] = True
        else:
            v = st.session_state["var_selection"].get(k, True)
            checked = st.checkbox(title, value=v, key=f"var_{k}")
            st.session_state["var_selection"][k] = bool(checked)

    picked_keys = [k for k, _ in ENDPOINTS if st.session_state["var_selection"].get(k, False)]
    st.write(f"Variabili selezionate: {len(picked_keys)}")

# -----------------------
# 3) Run
# -----------------------
with tab_run:
    st.subheader("Passo 3 · Esecuzione")
    selected_codes = sorted(st.session_state["selected_codes"])
    picked_keys = [k for k, _ in ENDPOINTS if st.session_state["var_selection"].get(k, False)]

    if len(selected_codes) == 0:
        st.warning("Seleziona almeno una scuola nel Passo 1.")
    if len(picked_keys) == 0:
        st.warning("Seleziona almeno una variabile nel Passo 2.")

    # Estimate
    est = len(selected_codes) * len(picked_keys)
    st.info(f"Stima richieste: {len(selected_codes)} scuole × {len(picked_keys)} variabili = {est}")

    force_big = False
    if est > 3000:
        st.warning("Run molto grande: aumenta rischio 429/timeout. Riduci scuole o variabili.")
        force_big = st.checkbox("Forza comunque", value=False)

    run_disabled = (len(selected_codes) == 0 or len(picked_keys) == 0 or (est > 3000 and not force_big))

    if st.button("Esegui scraping", type="primary", disabled=run_disabled):
        # progress (stage-based)
        prog = st.progress(0, text="Preparazione job…")

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + slug_safe(f"{len(selected_codes)}_schools")
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # input job minimal
        job_input = job_dir / "input_job.csv"
        reg_map = dict(zip(reg["CODICE_SCUOLA"], reg["denominazione"]))
        rows = [{"CODICE_SCUOLA": c, "denominazione": reg_map.get(c, "")} for c in selected_codes]
        pd.DataFrame(rows).to_csv(job_input, index=False, sep=sep)

        out_xlsx = job_dir / "dump.xlsx"
        outdir = job_dir / "out"
        rawdir = RAW_CACHE_DIR if use_cache else (outdir / "raw")

        prog.progress(15, text="Avvio scraper…")

        cp = run_scraper(
            input_csv=job_input,
            out_xlsx=out_xlsx,
            outdir=outdir,
            rawdir=rawdir,
            sep=sep,
            skip_existing=use_cache,
            no_kind=drop_kind,
            concurrency=concurrency,
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
            endpoints=picked_keys,
        )

        prog.progress(65, text="Post-processing output…")

        # logs
        (job_dir / "stdout.txt").write_text(cp.stdout or "", encoding="utf-8")
        (job_dir / "stderr.txt").write_text(cp.stderr or "", encoding="utf-8")

        anag = outdir / "anagrafica_base_wide.csv"
        obs = outdir / "observations_semantic.csv"

        # Charts
        charts_dir = job_dir / "charts"
        charts_zip = job_dir / "charts.zip"
        if make_charts and obs.exists():
            try:
                charts_dir.mkdir(parents=True, exist_ok=True)
                render_mode = "single" if chart_mode == "Una scuola" else "aggregate"
                charts_from_semantic(
                    obs_csv=obs,
                    charts_dir=charts_dir,
                    mode=render_mode,
                    school_code=(selected_codes[0] if selected_codes else None),
                    agg='mean',
                )
                # crea zip charts
                with zipfile.ZipFile(charts_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                    for fp in charts_dir.rglob('*'):
                        if fp.is_file():
                            z.write(fp, fp.relative_to(charts_dir))
            except Exception:
                pass

        # meta
        meta = {
            "job_id": job_id,
            "n_schools": len(selected_codes),
            "variables_selected": picked_keys,
            "use_cache": use_cache,
            "rawdir": str(rawdir),
            "returncode": cp.returncode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "concurrency": concurrency,
                "timeout_s": timeout_s,
                "retries": retries,
                "backoff_s": backoff_s,
            },
        }
        (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        prog.progress(85, text="Preparazione download…")

        if cp.returncode != 0:
            st.error("Errore nello scraping (returncode != 0).")
            st.code((cp.stderr or "")[-6000:] or "(stderr vuoto)")
        else:
            st.success("OK")

        # Downloads
        d1, d2, d3, d4 = st.columns(4, gap="small")

        with d1:
            if anag.exists():
                st.download_button("Scarica anagrafica_base_wide.csv", data=anag.read_bytes(),
                                   file_name="anagrafica_base_wide.csv", mime="text/csv")
            else:
                st.warning("anagrafica_base_wide.csv non trovato")
        with d2:
            if obs.exists():
                st.download_button("Scarica observations_semantic.csv", data=obs.read_bytes(),
                                   file_name="observations_semantic.csv", mime="text/csv")
            else:
                st.warning("observations_semantic.csv non trovato")
        with d3:
            if charts_zip.exists():
                st.download_button("Scarica charts.zip", data=charts_zip.read_bytes(),
                                   file_name="charts.zip", mime="application/zip")
            else:
                st.caption("charts.zip non disponibile")
        with d4:
            zip_path = job_dir / "job.zip"
            zip_job(job_dir, zip_path, include_raw=include_raw_in_zip)
            st.download_button("Scarica job.zip", data=zip_path.read_bytes(),
                               file_name="job.zip", mime="application/zip")

        prog.progress(100, text="Completato.")
        with st.expander("Log (stdout/stderr)"):
            st.code((cp.stdout or "")[-8000:] or "(stdout vuoto)")
            st.code((cp.stderr or "")[-8000:] or "(stderr vuoto)")

# -----------------------
# 4) Jobs + Cache tools
# -----------------------
with tab_jobs:
    st.subheader("Job e cache")

    selected_codes = sorted(st.session_state["selected_codes"])
    picked_keys = [k for k, _ in ENDPOINTS if st.session_state["var_selection"].get(k, False)]
    rawdir = RAW_CACHE_DIR

    if selected_codes and picked_keys:
        df_cache = cache_status_table(rawdir=rawdir, codes=selected_codes, endpoints=picked_keys, stale_days=stale_days)
        st.write("Stato cache (scuole selezionate × variabili selezionate)")
        st.dataframe(df_cache, use_container_width=True)

        colC1, colC2, colC3 = st.columns([1, 1, 2], gap="small")
        with colC1:
            inv_stale = st.button("Invalida solo stale")
        with colC2:
            inv_all = st.button("Invalida tutto (selezione)")
        with colC3:
            st.caption("L'invalidazione cancella file in data/raw. Il prossimo run li riscarica se necessari.")

        if inv_stale or inv_all:
            endpoints_scope = set(picked_keys)
            codes_scope = set(selected_codes)
            n = invalidate_cache(
                rawdir=rawdir,
                df_cache=df_cache,
                only_stale=bool(inv_stale and not inv_all),
                endpoints_scope=endpoints_scope,
                codes_scope=codes_scope,
            )
            st.success(f"File rimossi: {n}")

    st.divider()

    st.markdown("**Job recenti**")
    jobs = sorted([p for p in JOBS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)[:20]
    if not jobs:
        st.caption("Nessun job.")
    else:
        for p in jobs:
            meta_p = p / "meta.json"
            if meta_p.exists():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            else:
                meta = {}
            title = f"{p.name} · scuole={meta.get('n_schools','?')} · rc={meta.get('returncode','?')}"
            with st.expander(title, expanded=False):
                for fname in ["anagrafica_base_wide.csv", "observations_semantic.csv", "job.zip", "charts.zip", "stdout.txt", "stderr.txt", "meta.json"]:
                    fp = p / fname
                    if fp.exists():
                        st.download_button(f"Scarica {fname}", data=fp.read_bytes(), file_name=fname)
