# Motif collection

`combined_TF_motifs_jaspar_hocomoco12_cisbp.meme` — 1,503 position weight matrices in MEME
format, the collection every motif scan in this project reads. Scans use FIMO at `--thresh 1e-4`
with `--no-pgc`.

| source | motifs | selected by |
|--------|--------|-------------|
| JASPAR 2022 CORE vertebrates (non-redundant) | 779 | all profiles in the release |
| HOCOMOCO v12 | 440 | factors with no JASPAR profile |
| CIS-BP build 2.00 | 284 | factors with neither |

Motif identifiers carry their origin, so a hit can always be traced back to the collection it
came from and to the matrix version within it:

```
AHR::ARNT|jaspar|MA0006.1
AHRR|hocomoco12|AHRR.H12CORE.0.P.C
AHR|cisbp|M08716_2.00
```

The three sources are layered rather than merged: a factor takes its JASPAR profile where one
exists, and falls back to HOCOMOCO and then CIS-BP only where it does not, so no factor
contributes two competing matrices. No identifier occurs twice.

`sha256  d0b35040ce3f0d2d60eea7ee015bd061db12b259b3eb635085c66f53c26435d0`

## Attribution

The matrices are redistributed unchanged from their sources, and the terms that apply are the
original publishers', not this repository's. Cite the source collections, not this file:

- Castro-Mondragon, J. A. et al. JASPAR 2022. *Nucleic Acids Res.* **50**, D165–D173 (2022).
- Vorontsov, I. E. et al. HOCOMOCO in 2024. *Nucleic Acids Res.* **52**, D116–D123 (2024).
- Weirauch, M. T. et al. Determination and inference of eukaryotic transcription factor
  sequence specificity. *Cell* **158**, 1431–1443 (2014). (CIS-BP)
