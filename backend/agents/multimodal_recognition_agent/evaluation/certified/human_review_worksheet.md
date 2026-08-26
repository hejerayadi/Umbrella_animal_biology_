# Human review worksheet — Sprint 4 Phase 3 (live)

## ✅ APPROVED by Leith — 2026-08-26T01:01:18+00:00

All **105** proposed scores (35 cases × 3 criteria) accepted **without modification**. These are now recorded human scores.

> Human scores cover **communication quality only**. They never override a deterministic biological verdict: the `det. Top-1` / `det. dec.` columns are unchanged and remain authoritative. Metric totals were compared before and after approval and are identical.

> Answers come from the **supplemental explanation-capture run** (35/35 captured), which is separate from the certified deterministic results and feeds no metric.

## Approved human metrics

| Criterion | Mean | Median | Min | Max | ≥4 | <3 |
|---|---:|---:|---:|---:|---:|---:|
| Relevance | 4.77 | 5 | 4 | 5 | 35/35 | 0 |
| Explanation quality | 4.86 | 5 | 4 | 5 | 35/35 | 0 |
| Human task completion | 4.63 | 5 | 4 | 5 | 35/35 | 0 |

## Cases with a failed deterministic verdict — 7 of 35

These keep their failure. A high human score here means the agent *communicated* well, not that it was right.

| Case | Answer (sanitized) | det. Top-1 | det. dec. | Rel. | Expl. | Task |
|---|---|---|---|---|---|---|
| `REC-PPAR-02` | Remote BioCLIP-2 inference does not conclusively support Neofelis diardi as the highest-ranked taxonomic label… | **fail** | **pass** | 4 | 5 | 4 |
| `REC-UMAR-02` | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | **fail** | **fail** | 4 | 5 | 4 |
| `REC-LAFR-02` | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | **fail** | **fail** | 4 | 5 | 4 |
| `REC-GCAM-01` | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | **fail** | **fail** | 4 | 5 | 4 |
| `REC-GCAM-02` | Remote BioCLIP-2 inference supports Giraffa reticulata as the highest-ranked taxonomic label (classification s… | **fail** | **pass** | 5 | 5 | 5 |
| `REC-AMEL-01` | Remote BioCLIP-2 inference does not conclusively support Ailuropoda melanoleuca as the highest-ranked taxonomi… | **pass** | **fail** | 4 | 5 | 4 |
| `REC-PROS-02` | Remote BioCLIP-2 inference does not conclusively support Phoenicopterus ruber as the highest-ranked taxonomic… | **fail** | **pass** | 4 | 5 | 4 |

## All 35 cases — approved scores

| Case | Instruction | Answer (sanitized) | Rel. | Expl. | Task | Reviewer |
|---|---|---|---|---|---|---|
| `REC-PLEO-01` | What animal species is shown in this photograp… | The agent identified the subject as Panthera leo. The classification was produced by imageomics/bioclip-2 in r… | 5 | 5 | 5 | Leith |
| `REC-PLEO-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Panthera leo as the highest-ranked taxonomic label (classification score 0… | 5 | 5 | 5 | Leith |
| `REC-PTIG-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Panthera tigris as the highest-ranked taxonomic label (classification scor… | 5 | 5 | 5 | Leith |
| `REC-PTIG-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Panthera tigris as the highest-ranked taxonomic label (classification scor… | 5 | 5 | 5 | Leith |
| `REC-PPAR-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference does not conclusively support Panthera pardus as the highest-ranked taxonomic label… | 4 | 5 | 4 | Leith |
| `REC-PPAR-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference does not conclusively support Neofelis diardi as the highest-ranked taxonomic label… | 4 | 5 | 4 | Leith |
| `REC-UMAR-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Ursus maritimus as the highest-ranked taxonomic label (classification scor… | 5 | 5 | 5 | Leith |
| `REC-UMAR-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | 4 | 5 | 4 | Leith |
| `REC-LAFR-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Loxodonta africana as the highest-ranked taxonomic label (classification s… | 5 | 5 | 5 | Leith |
| `REC-LAFR-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | 4 | 5 | 4 | Leith |
| `REC-GCAM-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | 4 | 5 | 4 | Leith |
| `REC-GCAM-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Giraffa reticulata as the highest-ranked taxonomic label (classification s… | 5 | 5 | 5 | Leith |
| `REC-EQUA-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Equus quagga as the highest-ranked taxonomic label (classification score 0… | 5 | 5 | 5 | Leith |
| `REC-EQUA-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference does not conclusively support Equus quagga as the highest-ranked taxonomic label (c… | 4 | 5 | 4 | Leith |
| `REC-AMEL-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference does not conclusively support Ailuropoda melanoleuca as the highest-ranked taxonomi… | 4 | 5 | 4 | Leith |
| `REC-AMEL-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Ailuropoda melanoleuca as the highest-ranked taxonomic label (classificati… | 5 | 5 | 5 | Leith |
| `REC-VLAG-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Vulpes lagopus as the highest-ranked taxonomic label (classification score… | 5 | 5 | 5 | Leith |
| `REC-VLAG-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Vulpes lagopus as the highest-ranked taxonomic label (classification score… | 5 | 5 | 5 | Leith |
| `REC-PROS-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Phoenicopterus roseus as the highest-ranked taxonomic label (classificatio… | 5 | 5 | 5 | Leith |
| `REC-PROS-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference does not conclusively support Phoenicopterus ruber as the highest-ranked taxonomic… | 4 | 5 | 4 | Leith |
| `REC-AFOR-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Aptenodytes forsteri as the highest-ranked taxonomic label (classification… | 5 | 5 | 5 | Leith |
| `REC-AFOR-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Aptenodytes forsteri as the highest-ranked taxonomic label (classification… | 5 | 5 | 5 | Leith |
| `REC-BSCA-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Bubo scandiacus as the highest-ranked taxonomic label (classification scor… | 5 | 5 | 5 | Leith |
| `REC-BSCA-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference supports Bubo scandiacus as the highest-ranked taxonomic label (classification scor… | 5 | 5 | 5 | Leith |
| `AMB-01` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | 5 | 5 | 4 | Leith |
| `AMB-02` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | 5 | 5 | 4 | Leith |
| `AMB-03` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | 5 | 5 | 4 | Leith |
| `AMB-04` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | 5 | 5 | 4 | Leith |
| `AMB-05` | What animal species is shown in this photograp… | Remote BioCLIP-2 inference returned no taxonomic label for this image confident enough to name a species. Spec… | 5 | 5 | 4 | Leith |
| `INV-01` |     | A non-empty text instruction is required. Recognition needs one image and one text instruction in the same req… | 5 | 4 | 5 | Leith |
| `INV-02` | What animal species is shown in this photograp… | No image was provided. Recognition requires one image and one text instruction in the same request. | 5 | 4 | 5 | Leith |
| `INV-03` | What animal species is shown in this photograp… | Only inline base64 data URLs are accepted. Remote URLs and file paths are never fetched. | 5 | 4 | 5 | Leith |
| `DEL-01` | What is the genome of the animal in this photo… | Recognition identified the species in the supplied image as Panthera leo. Provide the genome information the u… | 5 | 4 | 5 | Leith |
| `DEL-02` | What is the evolutionary lineage of the animal… | Recognition identified the species in the supplied image as Panthera leo. Provide the evolution information th… | 5 | 4 | 5 | Leith |
| `DEL-03` | What is the genome of the animal in this photo… | Remote BioCLIP-2 inference supports Panthera leo as the highest-ranked taxonomic label (classification score 0… | 5 | 5 | 5 | Leith |
