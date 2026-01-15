import io
import base64
from html import escape as html_escape
import streamlit.components.v1 as components
# Bar chart -> render in-app + (optional) embed in HTML
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import plotly.express as px
except Exception:
    px = None

import json
import re
import sys
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# Optional map deps
try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None


# =========================
# Config
# =========================
DEFAULT_SCRAPER = "scraper.py"
DEFAULT_REGISTRY = "data_static/input.csv"
DEFAULT_SEP = ";"

DATA_DIR = Path("data")
RAW_CACHE_DIR = DATA_DIR / "raw"
JOBS_DIR = DATA_DIR / "jobs"

for d in (DATA_DIR, RAW_CACHE_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Variabili: (endpoint_key, title)
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
    ("assenze_ata", "Assenze ATA"),
    ("entrate_fonti_finanziamento", "Entrate per fonti di finanziamento"),
]


# =========================
# Helpers
# =========================
def slug_safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", (s or "").strip())[:180] or "x"


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _find_col_ci(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cols.get(cand.lower())
        if c:
            return c
    return None


def _bar_png_base64(labels: list[str], values: list[float], title: str) -> str | None:
    if plt is None:
        return None

    fig_w = max(8.0, min(16.0, 0.45 * max(1, len(labels))))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    ax.bar(range(len(labels)), values)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Valore")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_charts_from_observations(obs_csv: Path, outdir: Path, picked: list[str]) -> list[Path]:
    """
    Crea HTML per ogni variabile selezionata dentro outdir/charts.
    - Grafici interattivi se Plotly disponibile
    - Fallback a HTML statico minimale se Plotly non disponibile
    - Aggregazione: MEDIA (non somma)
    """
    charts_dir = outdir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if not obs_csv.exists():
        return []

    df = pd.read_csv(obs_csv, dtype=str, keep_default_na=False)

    ep_col = _find_col_ci(df, ["endpoint_key", "endpoint", "variabile", "key"])
    if ep_col is None:
        p = charts_dir / "README.html"
        p.write_text(
            "<h2>Nessun grafico</h2><p>Colonna endpoint_key non trovata in observations_semantic.csv.</p>",
            encoding="utf-8",
        )
        return [p]

    cat_col = _find_col_ci(df, ["category", "categoria", "label", "modalita", "modalità", "name", "x", "voce", "descrizione"])
    val_col = _find_col_ci(df, ["value", "valore", "y", "n", "count", "numero", "percent", "percentage", "pct"])
    title_col = _find_col_ci(df, ["title", "titolo", "nome"])

    created: list[Path] = []
    index_items: list[str] = []

    for k in picked:
        dfk = df[df[ep_col].astype(str) == str(k)].copy()
        if dfk.empty:
            continue

        page_title = str(k)
        if title_col and len(dfk[title_col].unique()) == 1:
            t = str(dfk[title_col].iloc[0]).strip()
            if t:
                page_title = f"{k} · {t}"

        html_path = charts_dir / f"{k}.html"

        if (cat_col is None) or (val_col is None):
            html_path.write_text(
                f"<h2>Nessun grafico</h2><p>Endpoint: <b>{html_escape(k)}</b></p>"
                "<p>Servono una colonna categoria e una colonna valore in observations_semantic.csv.</p>",
                encoding="utf-8",
            )
            created.append(html_path)
            index_items.append(f"<li><a href='{html_escape(html_path.name)}'>{html_escape(page_title)}</a> (solo testo)</li>")
            continue

        dfk[val_col] = pd.to_numeric(dfk[val_col], errors="coerce")
        dfk = dfk.dropna(subset=[val_col])
        if dfk.empty:
            html_path.write_text(
                f"<h2>Nessun grafico</h2><p>Endpoint: <b>{html_escape(k)}</b></p><p>Nessun valore numerico.</p>",
                encoding="utf-8",
            )
            created.append(html_path)
            index_items.append(f"<li><a href='{html_escape(html_path.name)}'>{html_escape(page_title)}</a> (nessun numerico)</li>")
            continue

        agg = (
            dfk.groupby(cat_col, dropna=False)[val_col]
            .mean()  # <-- MEDIA
            .sort_values(ascending=False)
            .head(30)
            .reset_index()
            .rename(columns={cat_col: "categoria", val_col: "valore"})
        )

        # Plotly interattivo se disponibile
        if px is not None and len(agg) > 0:
            # barre orizzontali: leggibilità molto maggiore
            fig = px.bar(
                agg.sort_values("valore", ascending=True),
                x="valore",
                y="categoria",
                orientation="h",
                title=page_title,
            )
            fig.update_layout(
                height=max(420, 22 * len(agg) + 140),
                margin=dict(l=10, r=10, t=70, b=10),
                xaxis_title="Valore (media)",
                yaxis_title="",
            )
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        else:
            # fallback HTML statico minimale
            rows = "\n".join([f"<tr><td>{html_escape(str(r['categoria']))}</td><td>{r['valore']}</td></tr>" for _, r in agg.iterrows()])
            html_path.write_text(
                f"<h2>{html_escape(page_title)}</h2>"
                "<p>Plotly non disponibile: export tabellare.</p>"
                "<p>Aggregazione: media (top 30).</p>"
                f"<table border='1' cellpadding='6' cellspacing='0'><tr><th>Categoria</th><th>Valore</th></tr>{rows}</table>",
                encoding="utf-8",
            )

        created.append(html_path)
        index_items.append(f"<li><a href='{html_escape(html_path.name)}'>{html_escape(page_title)}</a></li>")

    index_html = f"""
    <html>
      <head><meta charset="utf-8" /><title>Grafici</title></head>
      <body>
        <h2>Grafici</h2>
        <p>Aggregazione: <b>media</b> (top 30 per variabile).</p>
        <ul>{''.join(index_items) if index_items else '<li>Nessun grafico creato.</li>'}</ul>
      </body>
    </html>
    """.strip()

    index_path = charts_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    created.insert(0, index_path)

    return created



@st.cache_data(show_spinner=False)
def load_registry_from_path(path: str, sep: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
    df = normalize_headers(df)
    if "CODICE_SCUOLA" not in df.columns or "denominazione" not in df.columns:
        raise ValueError("Il registro deve avere colonne CODICE_SCUOLA e denominazione")
    df["CODICE_SCUOLA"] = df["CODICE_SCUOLA"].astype(str).str.strip()
    df["denominazione"] = df["denominazione"].astype(str).str.strip()
    df = df[df["CODICE_SCUOLA"] != ""].drop_duplicates(subset=["CODICE_SCUOLA"])
    return df


@st.cache_data(show_spinner=False)
def load_registry_from_upload(content_bytes: bytes, sep: str) -> pd.DataFrame:
    data = io.BytesIO(content_bytes)
    try:
        df = pd.read_csv(data, sep=sep, dtype=str, keep_default_na=False)
    except Exception:
        data.seek(0)
        df = pd.read_csv(data, sep=sep, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
    df = normalize_headers(df)
    if "CODICE_SCUOLA" not in df.columns or "denominazione" not in df.columns:
        raise ValueError("Il registro deve avere colonne CODICE_SCUOLA e denominazione")
    df["CODICE_SCUOLA"] = df["CODICE_SCUOLA"].astype(str).str.strip()
    df["denominazione"] = df["denominazione"].astype(str).str.strip()
    df = df[df["CODICE_SCUOLA"] != ""].drop_duplicates(subset=["CODICE_SCUOLA"])
    return df


def run_scraper(
    scraper_script: str,
    input_csv: Path,
    outdir: Path,
    sep: str,
    endpoints: list[str],
    wide: bool,
    skip_existing: bool,
    rawdir: Path | None,
    no_kind: bool,
    concurrency: int,
    timeout: int,
    retries: int,
    backoff: float,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        scraper_script,
        "--input", str(input_csv),
        "--sep", sep,
        "--outdir", str(outdir),
    ]
    if wide:
        cmd.append("--wide")
    if skip_existing:
        cmd.append("--skip-existing")
    if no_kind:
        cmd.append("--no-kind")
    if rawdir is not None:
        cmd += ["--rawdir", str(rawdir)]
    if endpoints:
        cmd += ["--endpoints", ",".join(endpoints)]

    cmd += ["--concurrency", str(concurrency)]
    cmd += ["--timeout", str(timeout)]
    cmd += ["--retries", str(retries)]
    cmd += ["--backoff", str(backoff)]

    return subprocess.run(cmd, capture_output=True, text=True)


def zip_selected(files: list[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in files:
            if fp is None or (not fp.exists()):
                continue

            if fp.is_file():
                # evita di includere lo zip dentro sé stesso
                if fp.resolve() == zip_path.resolve():
                    continue
                z.write(fp, fp.name)
                continue

            if fp.is_dir():
                base = fp.name
                for sub in fp.rglob("*"):
                    if sub.is_file():
                        arcname = str(Path(base) / sub.relative_to(fp))
                        z.write(sub, arcname)


def cache_status_for(reg: pd.DataFrame, rawdir: Path, picked: list[str]) -> pd.DataFrame:
    rows = []
    now = datetime.now(timezone.utc)
    for code in reg["CODICE_SCUOLA"].astype(str):
        cdir = rawdir / code
        for k in picked:
            p_json = cdir / f"{k}.json"
            p_txt = cdir / f"{k}.txt"
            p = p_json if p_json.exists() else (p_txt if p_txt.exists() else None)
            if p is None:
                rows.append({"CODICE_SCUOLA": code, "variabile": k, "status": "missing", "age_days": ""})
                continue
            age = (now - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)).total_seconds() / 86400.0
            rows.append({"CODICE_SCUOLA": code, "variabile": k, "status": "cached", "age_days": round(age, 1)})
    return pd.DataFrame(rows)


def ensure_state():
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("selected_codes", set())
    st.session_state.setdefault("var_selection", {})

    if not st.session_state["var_selection"]:
        for k, _ in ENDPOINTS:
            st.session_state["var_selection"][k] = True

    # anagrafica_base sempre True
    st.session_state["var_selection"]["anagrafica_base"] = True

    for k, _ in ENDPOINTS:
        st.session_state.setdefault(f"var_{k}", bool(st.session_state["var_selection"].get(k, True)))
    st.session_state["var_anagrafica_base"] = True


# =========================
# UI
# =========================
st.set_page_config(page_title="SIC Scraper", layout="wide")
ensure_state()

st.title("ScuolaInChiaro Scraper · v0.1")

with st.sidebar:
    st.subheader("Impostazioni")

    sep = st.text_input("Separatore CSV", value=DEFAULT_SEP)

    use_cache = st.checkbox("Usa cache globale (data/raw)", value=True)
    drop_kind = st.checkbox("Togli colonna kind", value=False)
    make_charts = st.checkbox("Crea grafici (HTML) per variabili selezionate", value=True)

    st.markdown("### [Debug] Limiti runtime")
    concurrency = st.slider("Concurrency", min_value=1, max_value=8, value=4)
    timeout = st.slider("Timeout (secondi)", min_value=15, max_value=90, value=45)
    retries = st.slider("Retries", min_value=0, max_value=4, value=2)
    backoff = st.slider("Backoff", min_value=0.0, max_value=1.0, value=0.4)

    st.divider()
    st.markdown("### [Debug] Script scraper")
    scraper_script = st.text_input("Posizione script scraper", value=DEFAULT_SCRAPER)
    st.caption("Esempio: scraper.py (nella stessa cartella dell'app).")

    st.divider()
    st.markdown("### [Debug] Registro input delle scuole")
    registry_path = st.text_input("Posizione registro input", value=DEFAULT_REGISTRY)
    upload = st.file_uploader("Oppure carica CSV", type=["csv"], label_visibility="collapsed")


# Load registry
reg = None
if upload is not None:
    reg = load_registry_from_upload(upload.getvalue(), sep)
else:
    if Path(registry_path).exists():
        reg = load_registry_from_path(registry_path, sep)
    else:
        st.warning(f"Registro non trovato: {registry_path}. Caricalo dal pannello [Debug].")

if reg is None:
    st.stop()

# detect columns
col_prov = detect_col(reg, ["provincia", "prov"])
col_com = detect_col(reg, ["comune", "citta", "città", "municipio"])
col_cap = detect_col(reg, ["cap", "CAP", "cap_residenza", "cap_zona"])
col_lat = detect_col(reg, ["latitudine", "latitude", "lat", "y", "LATITUDINE", "LAT"])
col_lon = detect_col(reg, ["longitudine", "longitude", "lon", "lng", "long", "x", "LONGITUDINE", "LON", "LNG"])

steps = ["1) Scuole", "2) Variabili", "3) Esecuzione", "[Debug] 4) Job e cache"]
st.caption("Percorso: selezione scuole → selezione variabili → esecuzione → download e cache.")

# Nav
col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1], gap="small")
with col_nav1:
    if st.button("◀ Indietro", disabled=(st.session_state["step"] == 0), key="nav_prev"):
        st.session_state["step"] = max(0, st.session_state["step"] - 1)
        st.rerun()
with col_nav2:
    step_label = st.radio("Passi", options=steps, index=st.session_state["step"], horizontal=True, label_visibility="collapsed")
    st.session_state["step"] = steps.index(step_label)
with col_nav3:
    if st.button("▶ Avanti ▶", disabled=(st.session_state["step"] == len(steps) - 1), key="nav_next"):
        st.session_state["step"] = min(len(steps) - 1, st.session_state["step"] + 1)
        st.rerun()

step = st.session_state["step"]

# -------------------------
# Step 1: Schools
# -------------------------
if step == 0:
    st.subheader("Passo 1 · Seleziona le scuole")
    st.caption("Filtri + tabella. La selezione resta valida nei passi successivi.")

    f1, f2, f3, f4 = st.columns([1, 1, 1, 2], gap="small")
    with f1:
        prov = st.selectbox(
            "Provincia",
            options=["(tutte)"] + (sorted(reg[col_prov].unique()) if col_prov else ["(colonna assente)"]),
            index=0,
            disabled=(col_prov is None),
        )
    with f2:
        com = st.selectbox(
            "Comune",
            options=["(tutti)"] + (sorted(reg[col_com].unique()) if col_com else ["(colonna assente)"]),
            index=0,
            disabled=(col_com is None),
        )
    with f3:
        cap_q = st.text_input("CAP (prefisso o lista, separata da virgole)", value="", help="Esempio: 001,004")
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
        mask = False
        for t in tokens:
            mask = mask | cap_series.str.startswith(t)
        df_view = df_view[mask]
    if q.strip():
        qs = q.strip().lower()
        cols = ["denominazione"]
        if col_com:
            cols.append(col_com)
        if col_prov:
            cols.append(col_prov)
        mask = False
        for c in cols:
            mask = mask | df_view[c].astype(str).str.lower().str.contains(qs, na=False)
        df_view = df_view[mask]

    df_view = df_view.head(5000)
    st.write(f"Risultati (max 5000): {len(df_view)}")

    df_sel = df_view[["CODICE_SCUOLA", "denominazione"]].copy()
    df_sel["Seleziona"] = df_sel["CODICE_SCUOLA"].isin(st.session_state["selected_codes"])

    edited = st.data_editor(
        df_sel,
        hide_index=True,
        column_config={
            "Seleziona": st.column_config.CheckboxColumn("Seleziona"),
            "CODICE_SCUOLA": st.column_config.TextColumn("Codice"),
            "denominazione": st.column_config.TextColumn("Denominazione"),
        },
        disabled=["CODICE_SCUOLA", "denominazione"],
        key="school_table",
        use_container_width=True,
    )

    cbtn1, cbtn2, cbtn3 = st.columns([1, 1, 2], gap="small")
    with cbtn1:
        if st.button("Seleziona tutte visibili", key="sel_all_visible"):
            for code in df_sel["CODICE_SCUOLA"].tolist():
                st.session_state["selected_codes"].add(code)
            st.rerun()
    with cbtn2:
        if st.button("Deseleziona tutte visibili", key="desel_all_visible"):
            for code in df_sel["CODICE_SCUOLA"].tolist():
                st.session_state["selected_codes"].discard(code)
            st.rerun()
    with cbtn3:
        manual_codes = st.text_area(
            "Oppure incolla codici meccanografici (uno per riga)",
            height=90,
            placeholder="RMIC....\nRMPS....",
            key="manual_codes",
        )
        if manual_codes.strip():
            extra = [x.strip() for x in manual_codes.splitlines() if x.strip()]
            for code in extra:
                st.session_state["selected_codes"].add(code)

    for _, row in edited.iterrows():
        code = row["CODICE_SCUOLA"]
        if bool(row["Seleziona"]):
            st.session_state["selected_codes"].add(code)
        else:
            st.session_state["selected_codes"].discard(code)

    st.info(f"Scuole selezionate: {len(st.session_state['selected_codes'])}")

    if (folium is None) or (st_folium is None):
        st.caption("Mappa disattivata (manca folium/streamlit-folium).")
    else:
        st.markdown("### Mappa (opzionale)")
        if (not col_lat) or (not col_lon):
            st.warning("Lat/Lon non trovati nel registro (colonne lat/lon mancanti).")
        else:
            df_map = df_view.copy()
            df_map[col_lat] = pd.to_numeric(df_map[col_lat], errors="coerce")
            df_map[col_lon] = pd.to_numeric(df_map[col_lon], errors="coerce")
            df_map = df_map.dropna(subset=[col_lat, col_lon])

            if len(df_map) == 0:
                st.caption("Nessuna scuola con coordinate nei filtri correnti.")
            else:
                center_lat = float(df_map[col_lat].median())
                center_lon = float(df_map[col_lon].median())
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, control_scale=True)

                for _, r in df_map.iterrows():
                    code = str(r["CODICE_SCUOLA"])
                    name = str(r.get("denominazione", ""))
                    is_sel = code in st.session_state["selected_codes"]
                    popup = folium.Popup(f"<b>{code}</b><br>{name}", max_width=280)
                    folium.CircleMarker(
                        location=[float(r[col_lat]), float(r[col_lon])],
                        radius=6 if is_sel else 4,
                        fill=True,
                        popup=popup,
                    ).add_to(m)

                st_folium(m, height=460, key="map_view")

# -------------------------
# Step 2: Variables
# -------------------------
elif step == 1:
    st.subheader("Passo 2 · Scegli le Variabili")
    st.caption("Queste variabili determinano quali dati vengono scaricati. Anagrafica base è obbligatoria.")

    c1, c2 = st.columns([1, 3], gap="large")

    with c1:
        if st.button("Seleziona tutte", key="vars_all"):
            for k, _ in ENDPOINTS:
                st.session_state["var_selection"][k] = True
                st.session_state[f"var_{k}"] = True
            st.session_state["var_selection"]["anagrafica_base"] = True
            st.session_state["var_anagrafica_base"] = True
            st.rerun()

        if st.button("Deseleziona tutte (tranne anagrafica)", key="vars_none"):
            for k, _ in ENDPOINTS:
                keep = (k == "anagrafica_base")
                st.session_state["var_selection"][k] = keep
                st.session_state[f"var_{k}"] = keep
            st.session_state["var_selection"]["anagrafica_base"] = True
            st.session_state["var_anagrafica_base"] = True
            st.rerun()

    with c2:
        st.markdown("**Variabili**")

        for k, title in ENDPOINTS:
            key = f"var_{k}"

            if k == "anagrafica_base":
                st.session_state[key] = True
                st.checkbox(f"{title} (obbligatoria)", value=True, disabled=True, key=key)
                st.session_state["var_selection"][k] = True
                continue

            if key not in st.session_state:
                st.session_state[key] = True

            checked = st.checkbox(title, key=key)
            st.session_state["var_selection"][k] = bool(checked)

    picked = [k for k, _ in ENDPOINTS if st.session_state["var_selection"].get(k, False)]
    st.info(f"Variabili selezionate: {len(picked)}")

    n_s = len(st.session_state["selected_codes"])
    est = n_s * len(picked)
    st.write(f"Stima richieste: {n_s} scuole × {len(picked)} variabili = {est}")
    if est > 3000:
        st.warning("Run molto grande: rischio 429/timeout. Riduci scuole o variabili.")

# -------------------------
# Step 3: Run
# -------------------------
elif step == 2:
    st.subheader("Passo 3 · Esecuzione")

    selected_codes = sorted(st.session_state["selected_codes"])
    picked = [k for k, _ in ENDPOINTS if st.session_state["var_selection"].get(k, False)]
    if "anagrafica_base" not in picked:
        picked = ["anagrafica_base"] + picked

    if len(selected_codes) == 0:
        st.error("Nessuna scuola selezionata. Torna al Passo 1.")
        st.stop()

    if len(picked) == 0:
        st.error("Nessuna variabile selezionata. Torna al Passo 2.")
        st.stop()

    est = len(selected_codes) * len(picked)
    if est > 3000:
        force = st.checkbox("Forza comunque (rischio alto)", value=False, key="force_big")
        if not force:
            st.stop()

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + slug_safe(f"{len(selected_codes)}_schools")
    job_dir = JOBS_DIR / job_id
    outdir = job_dir / "out"
    job_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    reg_map = dict(zip(reg["CODICE_SCUOLA"], reg["denominazione"]))
    job_input = job_dir / "input_job.csv"
    pd.DataFrame(
        [{"CODICE_SCUOLA": c, "denominazione": reg_map.get(c, "")} for c in selected_codes]
    ).to_csv(job_input, index=False, sep=sep)

    prog = st.progress(0, text="Preparazione job…")
    prog.progress(20, text="Esecuzione scraper…")

    rawdir = RAW_CACHE_DIR if use_cache else None

    cp = run_scraper(
        scraper_script=scraper_script,
        input_csv=job_input,
        outdir=outdir,
        sep=sep,
        endpoints=picked,
        wide=True,
        skip_existing=use_cache,
        rawdir=rawdir,
        no_kind=drop_kind,
        concurrency=concurrency,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )

    (job_dir / "stdout.txt").write_text(cp.stdout or "", encoding="utf-8")
    (job_dir / "stderr.txt").write_text(cp.stderr or "", encoding="utf-8")

    prog.progress(70, text="Post-process…")

    anag = outdir / "anagrafica_base_wide.csv"
    obs = outdir / "observations_semantic.csv"

    meta = {
        "job_id": job_id,
        "n_schools": len(selected_codes),
        "variabili": picked,
        "use_cache": use_cache,
        "returncode": cp.returncode,
        "timestamp": datetime.now().isoformat(),
    }
    (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    prog.progress(100, text="Completato")

    if cp.returncode != 0:
        st.error("Errore nello scraping. Vedi stderr.")
        st.code((cp.stderr or "")[-8000:])
        st.stop()

    st.success("OK")

    # ---------- CHARTS ----------
    charts_dir = outdir / "charts"
    charts_zip = job_dir / "charts.zip"

  # ---------- GRAFICI (IN FONDO, TUTTI) ----------
    if make_charts:
        st.markdown("### Grafici (interattivi)")
    
        charts_dir = outdir / "charts"
        charts_zip = job_dir / "charts.zip"
    
        # genera HTML in outdir/charts (plotly se possibile)
        created_html = build_charts_from_observations(obs, outdir, picked)
    
        # zip charts/
        if charts_dir.exists():
            zip_selected([charts_dir], charts_zip)
    
        # rendering in-app: tutti automaticamente
        if obs.exists():
            df_obs = pd.read_csv(obs, dtype=str, keep_default_na=False)
            ep_col = _find_col_ci(df_obs, ["endpoint_key", "endpoint", "variabile", "key"])
            cat_col = _find_col_ci(df_obs, ["category", "categoria", "label", "modalita", "modalità", "name", "x", "voce", "descrizione"])
            val_col = _find_col_ci(df_obs, ["value", "valore", "y", "n", "count", "numero", "percent", "percentage", "pct"])
            title_col = _find_col_ci(df_obs, ["title", "titolo", "nome"])
    
            if ep_col and cat_col and val_col:
                # per evitare una pagina ingestibile, puoi mettere expander.
                # Se li vuoi proprio tutti “aperti”, togli l’expander.
                for k in picked:
                    dfk = df_obs[df_obs[ep_col].astype(str) == str(k)].copy()
                    if dfk.empty:
                        continue
    
                    page_title = k
                    if title_col and len(dfk[title_col].unique()) == 1:
                        t = str(dfk[title_col].iloc[0]).strip()
                        if t:
                            page_title = f"{k} · {t}"
    
                    dfk[val_col] = pd.to_numeric(dfk[val_col], errors="coerce")
                    dfk = dfk.dropna(subset=[val_col])
                    if dfk.empty:
                        continue
    
                    agg = (
                        dfk.groupby(cat_col, dropna=False)[val_col]
                        .mean()  # MEDIA
                        .sort_values(ascending=False)
                        .head(30)
                        .reset_index()
                        .rename(columns={cat_col: "categoria", val_col: "valore"})
                    )
                    if len(agg) == 0:
                        continue
    
                    # expander per non demolire la UI; vuoi tutti aperti -> expanded=True
                    with st.expander(page_title, expanded=True):
                        if px is not None:
                            fig = px.bar(
                                agg.sort_values("valore", ascending=True),
                                x="valore",
                                y="categoria",
                                orientation="h",
                                title=None,
                            )
                            fig.update_layout(
                                height=max(420, 22 * len(agg) + 120),
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Valore (media)",
                                yaxis_title="",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # fallback non interattivo
                            st.bar_chart(agg.set_index# ---------- GRAFICI (IN FONDO, TUTTI) ----------
    if make_charts:
        st.markdown("### Grafici (interattivi)")
    
        charts_dir = outdir / "charts"
        charts_zip = job_dir / "charts.zip"
    
        # genera HTML in outdir/charts (plotly se possibile)
        created_html = build_charts_from_observations(obs, outdir, picked)
    
        # zip charts/
        if charts_dir.exists():
            zip_selected([charts_dir], charts_zip)
    
        # rendering in-app: tutti automaticamente
        if obs.exists():
            df_obs = pd.read_csv(obs, dtype=str, keep_default_na=False)
            ep_col = _find_col_ci(df_obs, ["endpoint_key", "endpoint", "variabile", "key"])
            cat_col = _find_col_ci(df_obs, ["category", "categoria", "label", "modalita", "modalità", "name", "x", "voce", "descrizione"])
            val_col = _find_col_ci(df_obs, ["value", "valore", "y", "n", "count", "numero", "percent", "percentage", "pct"])
            title_col = _find_col_ci(df_obs, ["title", "titolo", "nome"])
    
            if ep_col and cat_col and val_col:
                # per evitare una pagina ingestibile, puoi mettere expander.
                # Se li vuoi proprio tutti “aperti”, togli l’expander.
                for k in picked:
                    dfk = df_obs[df_obs[ep_col].astype(str) == str(k)].copy()
                    if dfk.empty:
                        continue
    
                    page_title = k
                    if title_col and len(dfk[title_col].unique()) == 1:
                        t = str(dfk[title_col].iloc[0]).strip()
                        if t:
                            page_title = f"{k} · {t}"
    
                    dfk[val_col] = pd.to_numeric(dfk[val_col], errors="coerce")
                    dfk = dfk.dropna(subset=[val_col])
                    if dfk.empty:
                        continue
    
                    agg = (
                        dfk.groupby(cat_col, dropna=False)[val_col]
                        .mean()  # MEDIA
                        .sort_values(ascending=False)
                        .head(30)
                        .reset_index()
                        .rename(columns={cat_col: "categoria", val_col: "valore"})
                    )
                    if len(agg) == 0:
                        continue
    
                    # expander per non demolire la UI; vuoi tutti aperti -> expanded=True
                    with st.expander(page_title, expanded=True):
                        if px is not None:
                            fig = px.bar(
                                agg.sort_values("valore", ascending=True),
                                x="valore",
                                y="categoria",
                                orientation="h",
                                title=None,
                            )
                            fig.update_layout(
                                height=max(420, 22 * len(agg) + 120),
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Valore (media)",
                                yaxis_title="",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # fallback non interattivo
                            st.bar_chart(agg.set_index("categoria")[["valore"]])
            else:
                st.caption("Impossibile costruire grafici: mancano colonne endpoint/categoria/valore in observations_semantic.csv.")
    
        # download charts.zip
        if charts_zip.exists():
            st.download_button(
                "Scarica grafici (charts.zip)",
                data=charts_zip.read_bytes(),
                file_name="charts.zip",
                mime="application/zip",
                key=f"dl_charts_{job_id}",
            )



    # ---------- PREVIEW ----------
    p1, p2 = st.columns(2, gap="small")
    with p1:
        if anag.exists():
            st.caption("Preview anagrafica_base_wide.csv")
            st.dataframe(pd.read_csv(anag, dtype=str, keep_default_na=False).head(30), use_container_width=True)
        else:
            st.caption("anagrafica_base_wide.csv assente.")
    with p2:
        if obs.exists():
            st.caption("Preview observations_semantic.csv")
            st.dataframe(pd.read_csv(obs, dtype=str, keep_default_na=False).head(30), use_container_width=True)
        else:
            st.caption("observations_semantic.csv assente.")

    # ---------- DOWNLOAD ----------
    st.markdown("### Download")
    d1, d2, d3 = st.columns(3, gap="small")

    with d1:
        if anag.exists():
            st.download_button(
                "Scarica anagrafiche",
                data=anag.read_bytes(),
                file_name="anagrafica_base_wide.csv",
                mime="text/csv",
                key=f"dl_anag_{job_id}",
            )

    with d2:
        if obs.exists():
            st.download_button(
                "Scarica dataset (semantic)",
                data=obs.read_bytes(),
                file_name="observations_semantic.csv",
                mime="text/csv",
                key=f"dl_obs_{job_id}",
            )

    with d3:
        zip_path = job_dir / "job.zip"
        files_to_zip = [
            job_dir / "meta.json",
            job_dir / "stdout.txt",
            job_dir / "stderr.txt",
        ]
        if anag.exists():
            files_to_zip.append(anag)
        if obs.exists():
            files_to_zip.append(obs)
        if charts_dir.exists():
            files_to_zip.append(charts_dir)
        if charts_zip.exists():
            files_to_zip.append(charts_zip)

        zip_selected(files_to_zip, zip_path)

        if zip_path.exists():
            st.download_button(
                "Scarica job.zip",
                data=zip_path.read_bytes(),
                file_name="job.zip",
                mime="application/zip",
                key=f"dl_zip_{job_id}",
            )

    with st.expander("Log"):
        st.code((cp.stdout or "")[-8000:])
        st.code((cp.stderr or "")[-8000:])

    # ---------- RENDER CHARTS IN APP (FROM GENERATED HTML) ----------
    if make_charts:
        st.markdown("### Grafici (tutti)")
    
        charts_dir = outdir / "charts"
        index_html = charts_dir / "index.html"
    
        if not charts_dir.exists():
            st.caption("Cartella charts assente.")
        else:
            # prendi tutti gli html dei grafici, escludendo index/readme
            chart_pages = sorted([
                p for p in charts_dir.glob("*.html")
                if p.name not in ("index.html", "README.html")
            ])
    
            if not chart_pages:
                st.caption("Nessun grafico HTML trovato in charts/.")
            else:
                # opzionale: link all'index (se vuoi)
                if index_html.exists():
                    with st.expander("Index (HTML)"):
                        components.html(index_html.read_text(encoding="utf-8"), height=260, scrolling=True)
    
                # mostra tutti i grafici
                for p in chart_pages:
                    with st.expander(p.stem, expanded=False):
                        html = p.read_text(encoding="utf-8")
                        components.html(html, height=620, scrolling=True)
    

# -------------------------
# Step 4: Jobs & cache
# -------------------------
else:
    st.subheader("Passo 4 · Job e cache")

    picked = [k for k, _ in ENDPOINTS if st.session_state["var_selection"].get(k, False)]
    if "anagrafica_base" not in picked:
        picked = ["anagrafica_base"] + picked

    if use_cache:
        st.markdown("### Cache globale (data/raw)")
        if st.button("Calcola stato cache (selezione corrente)", key="cache_status_btn"):
            df_cache = cache_status_for(
                reg[reg["CODICE_SCUOLA"].isin(st.session_state["selected_codes"])]
                if st.session_state["selected_codes"]
                else reg.head(200),
                RAW_CACHE_DIR,
                picked,
            )
            st.session_state["cache_df"] = df_cache

        df_cache = st.session_state.get("cache_df")
        if isinstance(df_cache, pd.DataFrame):
            st.dataframe(df_cache.head(500), use_container_width=True)
            st.caption("Max 500 righe in preview.")
    else:
        st.caption("Cache globale disattivata.")

    st.markdown("### Job recenti")
    jobs = sorted([p for p in JOBS_DIR.iterdir() if p.is_dir()], reverse=True)[:30]
    if not jobs:
        st.caption("Nessun job.")
    else:
        for j in jobs:
            meta_path = j / "meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}

            with st.expander(j.name):
                if meta:
                    st.json(meta)

                out = j / "out"
                an = out / "anagrafica_base_wide.csv"
                ob = out / "observations_semantic.csv"
                ch_zip = j / "charts.zip"
                cols = st.columns(4, gap="small")

                if an.exists():
                    cols[0].download_button(
                        "Scarica anagrafica",
                        data=an.read_bytes(),
                        file_name="anagrafica_base_wide.csv",
                        mime="text/csv",
                        key=f"dl_old_anag_{j.name}",
                    )
                if ob.exists():
                    cols[1].download_button(
                        "Scarica semantic",
                        data=ob.read_bytes(),
                        file_name="observations_semantic.csv",
                        mime="text/csv",
                        key=f"dl_old_obs_{j.name}",
                    )
                if ch_zip.exists():
                    cols[2].download_button(
                        "Scarica charts.zip",
                        data=ch_zip.read_bytes(),
                        file_name="charts.zip",
                        mime="application/zip",
                        key=f"dl_old_charts_{j.name}",
                    )

                logs_zip = j / "job_logs.zip"
                zip_selected([j / "meta.json", j / "stdout.txt", j / "stderr.txt"], logs_zip)
                if logs_zip.exists():
                    cols[3].download_button(
                        "Scarica log.zip",
                        data=logs_zip.read_bytes(),
                        file_name="job_logs.zip",
                        mime="application/zip",
                        key=f"dl_old_logs_{j.name}",
                    )
