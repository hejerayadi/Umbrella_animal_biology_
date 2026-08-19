"""
Ingestion du dataset SciCite (allenai/scicite) dans la collection Qdrant
ghaya_citation_examples.

IMPORTANT : ce dataset a un chargeur `datasets` legacy (script Python) non
supporte par les versions recentes de `datasets`. On contourne le probleme
en recuperant directement les fichiers Parquet auto-generes par HuggingFace
via leur API datasets-server (meme approche que pour Multi-XScience).

Schema reel confirme :
    string: le texte de la phrase de citation
    sectionName: nom de la section (ex. "Introduction", "Discussion")
    label: intention de citation -- Method / Background / Result (entier ou texte selon la version)
    citingPaperId / citedPaperId: identifiants des papiers

Usage (depuis backend/, avec le venv active) :
    python -m agents.Literature_Agent.subagents.writing.kb.ingest_scicite

Prerequis :
    pip install pandas pyarrow requests
"""

import os
import time
import requests
import pandas as pd
import tiktoken

from .ingest_to_qdrant import upsert_batch

COLLECTION = "ghaya_citation_examples"

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def download_parquet(url: str, max_retries: int = 8) -> pd.DataFrame:
    tmp_path = "scicite_download.tmp.parquet"

    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) == 0:
        os.remove(tmp_path)

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}

            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code not in (200, 206):
                    resp.raise_for_status()

                mode = "ab" if downloaded > 0 and resp.status_code == 206 else "wb"
                if mode == "wb":
                    downloaded = 0

                with open(tmp_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

            print(f"    telechargement termine ({downloaded / 1_000_000:.1f} Mo)", flush=True)
            df = pd.read_parquet(tmp_path)
            os.remove(tmp_path)
            return df

        except Exception as e:
            current_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            print(f"    tentative {attempt}/{max_retries} echouee a {current_size / 1_000_000:.1f} Mo : {e}", flush=True)
            if attempt < max_retries:
                time.sleep(3)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    raise RuntimeError(f"Impossible de telecharger {url} apres {max_retries} tentatives")


def get_parquet_urls(dataset_id: str, split: str = "train", prefer_source_config: bool = False) -> list[str]:
    resp = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": dataset_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    files = [f for f in data.get("parquet_files", []) if f.get("split") == split]
    if not files:
        raise RuntimeError(f"Aucun fichier parquet '{split}' trouve pour {dataset_id}")

    configs_found = sorted(set(f.get("config", "?") for f in files))
    print(f"  configs disponibles : {configs_found}", flush=True)

    if prefer_source_config:
        source_files = [f for f in files if "source" in f.get("config", "").lower()]
        if source_files:
            files = source_files
            print(f"  -> on garde uniquement : {sorted(set(f.get('config','?') for f in files))}", flush=True)

    return [f["url"] for f in files]


def normalize_intent(raw_label) -> str:
    label_str = str(raw_label).lower()
    if label_str in ("0", "method"):
        return "Method"
    if label_str in ("2", "result", "resultcomparison", "result_comparison"):
        return "Result Comparison"
    return "Background"  # 1 / "background" / defaut


def build_points_for_df(df: pd.DataFrame, split_name: str) -> tuple[list[str], list[dict], list[str]]:
    texts, payloads, ids = [], [], []

    # noms de colonnes possibles selon la source (allenai vs bigbio)
    text_col = next((c for c in ["string", "text"] if c in df.columns), None)
    section_col = next((c for c in ["sectionName", "section_name"] if c in df.columns), None)
    label_col = next((c for c in ["label", "labels"] if c in df.columns), None)
    citing_col = next((c for c in ["citingPaperId", "citing_paper_id", "document_id"] if c in df.columns), None)
    cited_col = next((c for c in ["citedPaperId", "cited_paper_id"] if c in df.columns), None)

    if text_col is None:
        print(f"  ATTENTION : aucune colonne de texte reconnue parmi {list(df.columns)}", flush=True)
        return texts, payloads, ids

    for i, row in df.iterrows():
        text = row.get(text_col, "") or ""
        if not str(text).strip():
            continue

        raw_label = row.get(label_col) if label_col else None
        # "labels" (bigbio) peut etre une liste -- on prend le premier element
        if isinstance(raw_label, (list, tuple)) or hasattr(raw_label, "__len__") and not isinstance(raw_label, str):
            try:
                raw_label = raw_label[0] if len(raw_label) > 0 else None
            except Exception:
                pass
        intent = normalize_intent(raw_label)

        texts.append(str(text))
        payloads.append({
            "doc_id": str(row.get(citing_col, f"{split_name}_{i}")) if citing_col else f"{split_name}_{i}",
            "source_dataset": "scicite",
            "citing_paper_id": str(row.get(citing_col, "")) if citing_col else "",
            "cited_paper_id": str(row.get(cited_col, "")) if cited_col else "",
            "intent": intent,
            "section_name": str(row.get(section_col, "") or "") if section_col else "",
            "context": str(text),
            "n_tokens": count_tokens(str(text)),
        })
        ids.append(f"scicite_{split_name}_{i}")

    return texts, payloads, ids


def ingest_scicite():
    print("Recuperation des URLs Parquet...", flush=True)
    urls = None
    for dataset_id, prefer_source in [("allenai/scicite", False), ("bigbio/scicite", True)]:
        try:
            print(f"  tentative : {dataset_id}", flush=True)
            urls = get_parquet_urls(dataset_id, split="train", prefer_source_config=prefer_source)
            print(f"  OK, source retenue : {dataset_id}", flush=True)
            break
        except Exception as e:
            print(f"  echec sur {dataset_id} : {e}", flush=True)

    if urls is None:
        print("ERREUR : aucune des sources n'a fonctionne.", flush=True)
        return

    print(f"{len(urls)} fichier(s) parquet a traiter...", flush=True)

    all_texts, all_payloads, all_ids = [], [], []

    for url in urls:
        print(f"  lecture de {url.split('/')[-1]}...", flush=True)
        df = download_parquet(url)
        print(f"  {len(df)} lignes, colonnes : {list(df.columns)}", flush=True)

        texts, payloads, ids = build_points_for_df(df, "train")
        all_texts.extend(texts)
        all_payloads.extend(payloads)
        all_ids.extend(ids)

    print(f"{len(all_texts)} points a inserer dans {COLLECTION}", flush=True)
    if all_texts:
        upsert_batch(COLLECTION, all_texts, all_payloads, all_ids, batch_size=20)
    print("Ingestion SciCite terminee.", flush=True)


if __name__ == "__main__":
    ingest_scicite()