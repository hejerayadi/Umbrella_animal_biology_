"""
Ingestion du dataset Multi-XScience dans la collection Qdrant
ghaya_related_work.

Tentative via la config bigbio_t2t (schema texte-a-texte standard bigbio :
text_1 = entree, text_2 = sortie) puisque la config "source" n'a pas le
champ related_work (verifie precedemment). Detection defensive des noms de
colonnes reelles (affichees au demarrage) pour eviter de re-deviner a l'aveugle.

Pas de filtre thematique : Multi-XScience est un dataset informatique/ML,
cette collection sert a apprendre la STRUCTURE d'un related work, pas le
sujet.

Usage (depuis backend/, avec le venv active) :
    python -m agents.Literature_Agent.subagents.writing.kb.ingest_multixscience
"""

import os
import time
import requests
import pandas as pd
import tiktoken

from .ingest_to_qdrant import upsert_batch

MAX_TOKENS_PER_CHUNK = 200
COLLECTION = "ghaya_related_work"
DOMAIN = "general"

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def chunk_section_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[str]:
    sentences = text.replace("\n", " ").split(". ")
    chunks, current, current_tokens = [], [], 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > max_tokens and current:
            chunks.append(". ".join(current) + ".")
            current, current_tokens = [], 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(". ".join(current) + ".")

    return chunks


def download_parquet(url: str, max_retries: int = 8) -> pd.DataFrame:
    tmp_path = "mxs_t2t_download.tmp.parquet"

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


def get_t2t_parquet_urls(dataset_id: str) -> list[str]:
    resp = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": dataset_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    files = [f for f in data.get("parquet_files", []) if f.get("split") == "train"]
    configs_found = sorted(set(f.get("config", "?") for f in files))
    print(f"  configs disponibles : {configs_found}", flush=True)

    t2t_files = [f for f in files if "t2t" in f.get("config", "").lower()]
    chosen = t2t_files if t2t_files else files
    print(f"  -> on garde : {sorted(set(f.get('config','?') for f in chosen))}", flush=True)

    return [f["url"] for f in chosen]


def build_points_for_row(row, idx: int, input_col: str, output_col: str) -> tuple[list[str], list[dict], list[str]]:
    texts, payloads, ids = [], [], []

    # IMPORTANT : "document_id" n'est PAS unique par ligne dans le schema t2t
    # (c'est l'id du papier source, partage par toutes ses paires de
    # citations -- l'utiliser seul causait des collisions d'ID Qdrant qui
    # ecrasaient les points entre eux). "id" est la vraie cle unique par ligne.
    row_id = str(row.get("id", idx))
    paper_id = str(row.get("document_id", idx))  # garde pour regrouper par papier source dans le payload
    input_text = row.get(input_col, "") or ""
    output_text = row.get(output_col, "") or ""

    if not str(output_text).strip():
        return texts, payloads, ids

    for chunk_idx, chunk in enumerate(chunk_section_text(str(input_text))):
        texts.append(chunk)
        payloads.append({
            "doc_id": paper_id,
            "source_dataset": "multi_xscience",
            "domain": DOMAIN,
            "section_type": "related_work",
            "input_text": chunk,
            "target_text": str(output_text),
            "n_tokens": count_tokens(chunk),
            "language": "en",
        })
        ids.append(f"mxs_t2t_{row_id}_{chunk_idx}")

    return texts, payloads, ids


def ingest_multixscience(target_count: int = 150):
    print("Recuperation des URLs Parquet (config t2t)...", flush=True)
    try:
        urls = get_t2t_parquet_urls("bigbio/multi_xscience")
    except Exception as e:
        print(f"ERREUR : {e}", flush=True)
        return

    all_texts, all_payloads, all_ids = [], [], []

    for url in urls:
        print(f"  lecture de {url.split('/')[-1]}...", flush=True)
        df = download_parquet(url)
        print(f"  {len(df)} lignes, colonnes : {list(df.columns)}", flush=True)

        input_col = next((c for c in ["text_1", "text1", "input"] if c in df.columns), None)
        output_col = next((c for c in ["text_2", "text2", "output", "target"] if c in df.columns), None)

        if input_col is None or output_col is None:
            print(f"  ATTENTION : colonnes text_1/text_2 introuvables parmi {list(df.columns)}", flush=True)
            continue

        print(f"  utilisation : input='{input_col}', output='{output_col}'", flush=True)

        # IMPORTANT : plusieurs lignes partagent le meme document_id (une par
        # paper cite). On regroupe par document_id et on combine tous les
        # abstracts cites (text_2) en UN SEUL texte cible par papier source,
        # pour eviter des embeddings dupliques (meme text_1 sur chaque ligne).
        grouped = df.groupby("document_id")
        n_papers = 0

        for doc_id, group in grouped:
            source_abstract = str(group.iloc[0][input_col] or "")
            combined_related_work = " ".join(str(t) for t in group[output_col] if str(t).strip())

            if not combined_related_work.strip() or not source_abstract.strip():
                continue

            for chunk_idx, chunk in enumerate(chunk_section_text(source_abstract)):
                all_texts.append(chunk)
                all_payloads.append({
                    "doc_id": str(doc_id),
                    "source_dataset": "multi_xscience",
                    "domain": DOMAIN,
                    "section_type": "related_work",
                    "input_text": chunk,
                    "target_text": combined_related_work,
                    "n_tokens": count_tokens(chunk),
                    "language": "en",
                })
                all_ids.append(f"mxs_t2t_{doc_id}_{chunk_idx}")

            n_papers += 1
            if n_papers % 50 == 0:
                print(f"  ... {n_papers} papiers regroupes jusqu'ici", flush=True)

            if n_papers >= target_count:
                break

    print(f"{len(all_texts)} chunks a inserer dans {COLLECTION} ({n_papers} papiers uniques)", flush=True)
    if all_texts:
        upsert_batch(COLLECTION, all_texts, all_payloads, all_ids, batch_size=20)
    print("Ingestion Multi-XScience (t2t, regroupe par papier) terminee.", flush=True)


if __name__ == "__main__":
    ingest_multixscience(target_count=150)