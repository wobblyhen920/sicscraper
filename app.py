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

# Session state
if "selected_codes" not in st.session_state:
    st.session_state["selected_codes"] = set()
if "last_job_id" not in st.session_state:
    st.session_state["last_job_id"] = ""

# Sidebar settings
with st.sidebar:
    st.subheader("Impostazioni")
    scraper_script = st.text_input("Script scraper", value=DEFAULT_SCRAPER_SCRIPT)
    sep = st.text_input("Separatore CSV", value=DEFAULT_SEP)
    use_cache = st.checkbox("Usa cache globale", value=True)
    drop_kind = st.checkbox("Togli colonna kind", value=False)

    st.markdown("**Runtime**")
    concurrency = st.slider("Concurrency", min_value=1, max_value=16, value=6, step=1)
    timeout_s = st.slider("Timeout (s)", min_value=10, max_value=120, value=40, step=5)
    retries = st.slider("Retries", min_value=1, max_value=6, value=3, step=1)
    backoff_s = st.slider("Backoff (s)", min_value=0.0, max_value=2.0, value=0.4, step=0.1)

    st.markdown("**Cache**")
    stale_days = st.number_input("Considera stantia la cache se più vecchia di (giorni)", min_value=1, max_value=3650, value=180, step=30)
    st.caption("Cache: data/raw. Output job: data/jobs/<job_id>.")


# Registry input
st.subheader("Registry scuole")
colR1, colR2 = st.columns([2, 1], gap="large")
with colR2:
    upload = st.file_uploader("Oppure carica un CSV", type=["csv"], accept_multiple_files=False)
with colR1:
    registry_path = st.text_input("Path registry CSV", value=DEFAULT_REGISTRY)

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
        reg = load_registry_from_path(registry_path, sep)
    except Exception as e:
        st.error(str(e))
        st.stop()

col_prov = detect_col(reg, "provincia")
col_com = detect_col(reg, "comune")
col_cap = detect_col(reg, "cap")
col_lat = detect_col(reg, "latitudine") or detect_col(reg, "lat") or detect_col(reg, "latitude")
col_lon = detect_col(reg, "longitudine") or detect_col(reg, "lon") or detect_col(reg, "longitude")

if not col_lat or not col_lon:
    st.warning("Mappa disattiva: nel registry mancano latitudine/longitudine.")

# Query params defaults
qp = get_query_params()
default_q = qp.get("q", "")
default_cap = qp.get("cap", "")
default_prov = qp.get("prov", "(tutte)")
default_com = qp.get("com", "(tutti)")

# Filters
st.subheader("Filtri")
f1, f2, f3, f4 = st.columns([1, 1, 1, 2], gap="small")
with f1:
    prov_options = ["(tutte)"] + (sorted([x for x in reg[col_prov].unique() if str(x).strip() != ""]) if col_prov else [])
    prov = st.selectbox("Provincia", options=prov_options if col_prov else ["(colonna assente)"], index=(prov_options.index(default_prov) if col_prov and default_prov in prov_options else 0), disabled=(col_prov is None))
with f2:
    com_options = ["(tutti)"] + (sorted([x for x in reg[col_com].unique() if str(x).strip() != ""]) if col_com else [])
    com = st.selectbox("Comune", options=com_options if col_com else ["(colonna assente)"], index=(com_options.index(default_com) if col_com and default_com in com_options else 0), disabled=(col_com is None))
with f3:
    cap_prefix = st.text_input("CAP (prefisso o lista)", value=default_cap, disabled=(col_cap is None), placeholder="001,004 oppure 001")
with f4:
    q = st.text_input("Ricerca testo", value=default_q, placeholder="denominazione, comune, provincia, cap")

# Save filters to URL
colS1, colS2 = st.columns([1, 3], gap="small")
with colS1:
    if st.button("Salva filtri in URL"):
        set_query_params(q=q, cap=cap_prefix, prov=(prov if col_prov else ""), com=(com if col_com else ""))

df_view = filter_registry(reg, provincia=(prov if col_prov else None), comune=(com if col_com else None), cap_prefix=cap_prefix, q=q)
df_view = df_view.head(5000)

st.caption(f"Risultati filtrati: {len(df_view)} (max 5000 in UI)")

# Manual codes
manual_codes = st.text_area("Incolla codici (uno per riga)", height=110, placeholder="RMIC....\nRMPS....")
extra_codes = []
if manual_codes.strip():
    extra_codes = [x.strip() for x in manual_codes.splitlines() if x.strip()]
    extra_codes = [x for x in extra_codes if is_valid_code(x)]

# ======================
# ENDPOINT selection
# ======================
st.subheader("Endpoint")
picked = st.multiselect(
    "Seleziona endpoint",
    options=[f"{k} · {label}" for k, label in ENDPOINTS],
    default=[f"{k} · {label}" for k, label in ENDPOINTS],
)
picked_keys = [p.split(" · ")[0].strip() for p in picked]

# Forza anagrafica_base, altrimenti gli output richiesti diventano contraddittori
if "anagrafica_base" not in picked_keys:
    picked_keys = ["anagrafica_base"] + picked_keys

# ======================
# Selection UI + Map
# ======================
tab_sel, tab_map, tab_run, tab_jobs = st.tabs(["Selezione", "Mappa", "Esecuzione", "Job"])

with tab_sel:
    st.subheader("Selezione scuole")

    view_cols = ["CODICE_SCUOLA", "denominazione"]
    for c in [col_com, col_prov, col_cap]:
        if c and c not in view_cols:
            view_cols.append(c)

    df_show = df_view[view_cols].copy()
    df_show.insert(0, "selected", df_show["CODICE_SCUOLA"].isin(st.session_state["selected_codes"]))

    edited = st.data_editor(
        df_show,
        use_container_width=True,
        hide_index=True,
        column_config={"selected": st.column_config.CheckboxColumn("Seleziona", help="Spunta per includere nel job")},
        disabled=[c for c in df_show.columns if c != "selected"],
        height=420,
    )

    # aggiorna selection da editor
    if edited is not None and "selected" in edited.columns:
        sel_codes = set(edited.loc[edited["selected"] == True, "CODICE_SCUOLA"].astype(str).tolist())
        # conserva anche codici non in vista già selezionati
        keep = st.session_state["selected_codes"] - set(edited["CODICE_SCUOLA"].tolist())
        st.session_state["selected_codes"] = keep | sel_codes

    # aggiungi manual codes validi
    if extra_codes:
        st.session_state["selected_codes"] = set(st.session_state["selected_codes"]) | set(extra_codes)

    b1, b2, b3 = st.columns([1, 1, 2], gap="small")
    with b1:
        if st.button("Seleziona tutte visibili"):
            st.session_state["selected_codes"] = set(st.session_state["selected_codes"]) | set(df_show["CODICE_SCUOLA"].tolist())
    with b2:
        if st.button("Deseleziona tutte visibili"):
            st.session_state["selected_codes"] = set(st.session_state["selected_codes"]) - set(df_show["CODICE_SCUOLA"].tolist())
    with b3:
        st.write(f"Codici selezionati: {len(st.session_state['selected_codes'])}")

    # warning codici invalidi incollati
    invalid = [x.strip() for x in manual_codes.splitlines() if x.strip() and not is_valid_code(x.strip())]
    if invalid:
        st.warning(f"Codici scartati (formato non valido): {', '.join(invalid[:10])}" + (" ..." if len(invalid) > 10 else ""))

with tab_map:
    st.subheader("Mappa")
    if folium is None or st_folium is None or (not col_lat) or (not col_lon):
        st.info("Per la mappa servono: folium + streamlit-folium e colonne latitudine/longitudine nel registry.")
    else:
        dfm = df_view.copy()
        # pulizia numerica
        dfm[col_lat] = pd.to_numeric(dfm[col_lat], errors="coerce")
        dfm[col_lon] = pd.to_numeric(dfm[col_lon], errors="coerce")
        dfm = dfm[dfm[col_lat].notna() & dfm[col_lon].notna()].copy()

        st.caption(f"Punti mappa: {len(dfm)}")

        show_only_selected = st.checkbox("Mostra solo scuole selezionate", value=False)
        if show_only_selected:
            dfm = dfm[dfm["CODICE_SCUOLA"].isin(st.session_state["selected_codes"])]

        if dfm.empty:
            st.warning("Nessun punto da mostrare con i filtri attuali.")
        else:
            lat0 = float(dfm[col_lat].mean())
            lon0 = float(dfm[col_lon].mean())
            m = folium.Map(location=[lat0, lon0], zoom_start=11, control_scale=True)

            if Draw is not None:
                Draw(
                    export=False,
                    draw_options={
                        "polyline": False,
                        "circle": False,
                        "circlemarker": False,
                        "marker": False,
                        "polygon": True,
                        "rectangle": True,
                    },
                    edit_options={"edit": False},
                ).add_to(m)

            # markers
            for _, r in dfm.iterrows():
                code = str(r["CODICE_SCUOLA"])
                denom = str(r.get("denominazione", ""))
                tip = f"{code}"
                pop = folium.Popup(f"<b>{code}</b><br/>{denom}", max_width=360)
                color = "red" if code in st.session_state["selected_codes"] else "blue"
                folium.Marker(
                    location=[float(r[col_lat]), float(r[col_lon])],
                    tooltip=tip,
                    popup=pop,
                    icon=folium.Icon(color=color),
                ).add_to(m)

            map_state = st_folium(m, height=560, use_container_width=True)

            # click marker toggles selection
            clicked_code = map_state.get("last_object_clicked_tooltip")
            if clicked_code and is_valid_code(clicked_code):
                s = set(st.session_state["selected_codes"])
                if clicked_code in s:
                    s.remove(clicked_code)
                else:
                    s.add(clicked_code)
                st.session_state["selected_codes"] = s
                st.rerun()

            # polygon/rectangle selection: aggiunge tutte le scuole nel bbox
            drawing = map_state.get("last_active_drawing")
            if drawing and isinstance(drawing, dict):
                geom = drawing.get("geometry", {})
                if geom.get("type") == "Polygon":
                    coords = geom.get("coordinates", [])
                    if coords and isinstance(coords, list) and coords[0]:
                        ring = coords[0]  # list of [lon,lat]
                        lons = [p[0] for p in ring if isinstance(p, list) and len(p) == 2]
                        lats = [p[1] for p in ring if isinstance(p, list) and len(p) == 2]
                        if lats and lons:
                            min_lat, max_lat = min(lats), max(lats)
                            min_lon, max_lon = min(lons), max(lons)
                            df_sel = dfm[(dfm[col_lat] >= min_lat) & (dfm[col_lat] <= max_lat) & (dfm[col_lon] >= min_lon) & (dfm[col_lon] <= max_lon)]
                            if not df_sel.empty:
                                if st.button(f"Aggiungi selezione area ({len(df_sel)} scuole)"):
                                    st.session_state["selected_codes"] = set(st.session_state["selected_codes"]) | set(df_sel["CODICE_SCUOLA"].astype(str).tolist())
                                    st.rerun()

        st.write(f"Selezionati: {len(st.session_state['selected_codes'])}")

with tab_run:
    st.subheader("Esecuzione job")

    selected_codes = sorted(set(st.session_state["selected_codes"]))
    if not selected_codes:
        st.warning("Seleziona almeno una scuola.")
        st.stop()

    # stima richieste
    est_requests = len(selected_codes) * len(picked_keys)
    st.write(f"Stima richieste: {est_requests} ({len(selected_codes)} scuole × {len(picked_keys)} endpoint)")

    force = False
    if est_requests > 3000:
        st.warning("Volume alto. Aspettati rate limit e tempi lunghi.")
        force = st.checkbox("Forza comunque", value=False)

    # cache status
    st.markdown("### Stato cache")
    df_cache = cache_status_table(RAW_CACHE_DIR, selected_codes, picked_keys, int(stale_days))
    if not df_cache.empty:
        summ = df_cache.groupby(["endpoint_key", "cache"]).size().reset_index(name="n")
        st.dataframe(summ, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns([1, 1, 2], gap="small")
        with c1:
            if st.button("Invalida cache stantia (selezione)"):
                n = invalidate_cache(RAW_CACHE_DIR, df_cache, only_stale=True, endpoints_scope=set(picked_keys), codes_scope=set(selected_codes))
                st.success(f"File rimossi: {n}")
                st.rerun()
        with c2:
            if st.button("Invalida cache endpoint selezionati"):
                n = invalidate_cache(RAW_CACHE_DIR, df_cache, only_stale=False, endpoints_scope=set(picked_keys), codes_scope=set(selected_codes))
                st.success(f"File rimossi: {n}")
                st.rerun()
        with c3:
            st.caption("Invalidazione: cancellazione file raw. Se usi cache, lo scraper riscarica solo i file mancanti.")

    st.markdown("### Avvio")
    run_disabled = (len(picked_keys) == 0) or (est_requests > 3000 and not force)

    create_charts = st.checkbox("Crea grafici HTML dagli endpoint selezionati", value=False, disabled=(alt is None))
    chart_mode = st.radio("Grafici", options=["Una scuola", "Media su scuole"], horizontal=True, disabled=(not create_charts))
    chart_agg = st.selectbox("Aggregazione (solo media)", options=["mean", "sum"], index=0, disabled=(not create_charts or chart_mode != "Media su scuole"))
    chart_school = None
    if create_charts and chart_mode == "Una scuola":
        chart_school = st.selectbox("Scegli scuola per grafici", options=selected_codes)

    include_raw_in_zip = st.checkbox("Includi raw subset nello zip del job", value=False)

    if st.button("Esegui scraping", type="primary", disabled=run_disabled):
        # job
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + slug_safe(f"{len(selected_codes)}_schools")
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # input job
        job_input = job_dir / "input_job.csv"
        reg_map = dict(zip(reg["CODICE_SCUOLA"], reg["denominazione"]))
        rows = [{"CODICE_SCUOLA": c, "denominazione": reg_map.get(c, "")} for c in selected_codes]
        pd.DataFrame(rows).to_csv(job_input, index=False, sep=sep)

        outdir = job_dir / "out"
        outdir.mkdir(parents=True, exist_ok=True)

        # run
        cp = run_scraper(
            scraper_script=scraper_script,
            input_csv=job_input,
            outdir=outdir,
            rawdir=RAW_CACHE_DIR if use_cache else (job_dir / "raw_local"),
            endpoints=picked_keys,
            sep=sep,
            concurrency=int(concurrency),
            timeout_s=int(timeout_s),
            retries=int(retries),
            backoff_s=float(backoff_s),
            skip_existing=bool(use_cache),
            no_kind=bool(drop_kind),
        )

        (job_dir / "stdout.txt").write_text(cp.stdout or "", encoding="utf-8")
        (job_dir / "stderr.txt").write_text(cp.stderr or "", encoding="utf-8")

        # outputs
        anag = outdir / "anagrafica_base_wide.csv"
        obs = outdir / "observations_semantic.csv"
        res_jsonl = outdir / "results_long.jsonl"

        # charts
        chart_files = []
        if create_charts:
            charts_dir = job_dir / "charts"
            if chart_mode == "Una scuola":
                chart_files = charts_from_semantic(obs, charts_dir, mode="single", school_code=chart_school, agg="mean")
            else:
                chart_files = charts_from_semantic(obs, charts_dir, mode="agg", school_code=None, agg=chart_agg)

        # raw subset if requested
        if include_raw_in_zip and use_cache:
            build_raw_subset(job_dir, RAW_CACHE_DIR, selected_codes, picked_keys)

        meta = {
            "job_id": job_id,
            "n_schools": len(selected_codes),
            "endpoints_selected": picked_keys,
            "use_cache": bool(use_cache),
            "rawdir": str(RAW_CACHE_DIR if use_cache else (job_dir / "raw_local")),
            "returncode": cp.returncode,
            "timestamp": datetime.now().isoformat(),
            "charts": [p.name for p in chart_files],
        }
        (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        st.session_state["last_job_id"] = job_id

        if cp.returncode != 0:
            st.error("Errore nello scraping. Guarda stderr.")
            st.code((cp.stderr or "")[-6000:])
        else:
            st.success("OK")

        # Downloads
        st.markdown("### Download")
        d1, d2, d3 = st.columns(3, gap="small")
        with d1:
            if anag.exists():
                st.download_button("Scarica anagrafica_base_wide.csv", anag.read_bytes(), file_name="anagrafica_base_wide.csv", mime="text/csv")
            else:
                st.warning("anagrafica_base_wide.csv assente")
        with d2:
            if obs.exists():
                st.download_button("Scarica observations_semantic.csv", obs.read_bytes(), file_name="observations_semantic.csv", mime="text/csv")
            else:
                st.warning("observations_semantic.csv assente")
        with d3:
            zip_path = job_dir / "job.zip"
            zip_job(job_dir, zip_path, include_raw=include_raw_in_zip)
            st.download_button("Scarica job.zip", zip_path.read_bytes(), file_name="job.zip", mime="application/zip")

        # Charts zip
        if create_charts and (job_dir / "charts").exists():
            charts_zip = job_dir / "charts.zip"
            with zipfile.ZipFile(charts_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in (job_dir / "charts").glob("*.html"):
                    z.write(p, p.name)
            st.download_button("Scarica charts.zip", charts_zip.read_bytes(), file_name="charts.zip", mime="application/zip")

        # Preview
        st.markdown("### Preview output")
        p1, p2 = st.columns(2, gap="large")
        with p1:
            if anag.exists():
                st.write("anagrafica_base_wide.csv (head)")
                try:
                    st.dataframe(pd.read_csv(anag, dtype=str, keep_default_na=False).head(50), use_container_width=True)
                except Exception:
                    st.info("Preview non disponibile (parse).")
        with p2:
            if obs.exists():
                st.write("observations_semantic.csv (head)")
                try:
                    st.dataframe(pd.read_csv(obs, dtype=str, keep_default_na=False).head(100), use_container_width=True)
                except Exception:
                    st.info("Preview non disponibile (parse).")

        # Quick diagnostics + retry
        st.markdown("### Diagnostica e retry")
        df_res = read_jsonl(res_jsonl)
        if not df_res.empty and "ok" in df_res.columns:
            df_res["ok"] = df_res["ok"].astype(bool)
            fail = df_res[df_res["ok"] == False].copy()
            st.write(f"Fallimenti: {len(fail)}")
            if not fail.empty:
                st.dataframe(
                    fail[["CODICE_SCUOLA", "endpoint_key", "status", "error"]].head(50),
                    use_container_width=True,
                    hide_index=True,
                )
                if st.button("Crea job di retry (solo falliti)"):
                    retry_codes = sorted(set(fail["CODICE_SCUOLA"].astype(str).tolist()))
                    retry_eps = sorted(set(fail["endpoint_key"].astype(str).tolist()))
                    st.session_state["selected_codes"] = set(retry_codes)
                    # aggiorna endpoint selection in URL non possibile qui; avviso
                    st.info(f"Impostati in selezione: {len(retry_codes)} scuole. Endpoint falliti: {len(retry_eps)}. Ora ripeti run.")
        else:
            st.caption("results_long.jsonl assente o non leggibile.")

        with st.expander("Log (stdout/stderr)"):
            st.code((cp.stdout or "")[-8000:] or "(stdout vuoto)")
            st.code((cp.stderr or "")[-8000:] or "(stderr vuoto)")


with tab_jobs:
    st.subheader("Job salvati")
    jobs = sorted([p for p in JOBS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    if not jobs:
        st.info("Nessun job.")
    else:
        job_names = [p.name for p in jobs]
        default_idx = 0
        if st.session_state.get("last_job_id") in job_names:
            default_idx = job_names.index(st.session_state["last_job_id"])
        picked_job = st.selectbox("Scegli job", options=job_names, index=default_idx)

        job_dir = JOBS_DIR / picked_job
        meta_path = job_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            st.json(meta)
        else:
            st.warning("meta.json assente")

        outdir = job_dir / "out"
        anag = outdir / "anagrafica_base_wide.csv"
        obs = outdir / "observations_semantic.csv"

        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            if anag.exists():
                st.download_button("Scarica anagrafica_base_wide.csv", anag.read_bytes(), file_name="anagrafica_base_wide.csv", mime="text/csv")
        with c2:
            if obs.exists():
                st.download_button("Scarica observations_semantic.csv", obs.read_bytes(), file_name="observations_semantic.csv", mime="text/csv")
        with c3:
            zip_path = job_dir / "job.zip"
            if zip_path.exists():
                st.download_button("Scarica job.zip", zip_path.read_bytes(), file_name="job.zip", mime="application/zip")
            else:
                st.info("job.zip assente. Lancialo da Esecuzione, oppure ricrea zip a mano.")

        with st.expander("Log"):
            for name in ["stdout.txt", "stderr.txt"]:
                p = job_dir / name
                if p.exists():
                    st.markdown(f"**{name}**")
                    st.code(p.read_text(encoding="utf-8")[-8000:])
