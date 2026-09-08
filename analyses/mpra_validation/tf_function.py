"""Curated activator/repressor annotation for partner TFs in K562_all.tsv.

Sources: canonical literature roles. TFs with strongly context-dependent or
unclear primary function are marked "unknown" — the validation falls back to
two-sided |effect| in that case (PLAN.md caveat 3).

Conservative assignments only. When in doubt, "unknown".
"""

TF_FUNCTION: dict[str, str] = {
    # canonical activators
    "ARNT": "activator",
    "ATF1": "activator",
    "ATF4": "activator",
    "CEBPG": "activator",
    "CLOCK": "activator",
    "CREB3L4": "activator",
    "ELK4": "activator",
    "ETS1": "activator",
    "ETV2": "activator",
    "ETV6": "repressor",        # ETS-family ETV6/TEL is a canonical repressor
    "FOS": "activator",
    "FOSL1::JUND": "activator",
    "FOSL2": "activator",
    "GATA5": "activator",
    "GMEB1": "activator",
    "GMEB2": "activator",
    "GRHL2": "activator",
    "HNF1A": "activator",
    "JUN": "activator",
    "KLF15": "activator",
    "MAFK": "activator",
    "MLXIPL": "activator",
    "MYC": "activator",
    "NFE2L2": "activator",
    "NFIC": "activator",
    "NFKB1": "activator",
    "NFYA": "activator",
    "NFYB": "activator",
    "NFYC": "activator",
    "NPAS2": "activator",
    "NRF1": "activator",
    "ONECUT2": "activator",
    "PAX5": "activator",
    "RFX1": "activator",
    "RFX7": "activator",
    "RUNX2": "activator",
    "SMAD5": "activator",
    "SP1": "activator",
    "SRF": "activator",
    "SRY": "activator",
    "STAT1": "activator",
    "STAT4": "activator",
    "STAT6": "activator",
    "TBP": "activator",
    "TCF12": "activator",
    "TEAD2": "activator",
    "TEAD3": "activator",
    "TFAP2C": "activator",
    "TFEB": "activator",
    "THAP11": "activator",
    "USF1": "activator",
    "VEZF1": "activator",
    "YY1": "activator",
    "IRF2": "activator",
    "RREB1": "activator",
    "FOXD3": "activator",
    "EBF3": "activator",
    "HOXA9": "activator",
    "HOXB13": "activator",
    "MEIS1": "activator",
    "PHOX2B": "activator",
    "ISL1": "activator",
    "ZNF740": "activator",

    # canonical repressors
    "BCL6": "repressor",
    "PRDM1": "repressor",       # Blimp1 — terminal differentiation repressor
    "HES1": "repressor",        # Notch-pathway repressor
    "REST": "repressor",
    "ZBTB33": "repressor",      # Kaiso, methyl-CpG repressor
    "ZNF263": "repressor",      # KRAB-zinc finger
    "ZNF354A": "repressor",     # KRAB-zinc finger
    "ZNF93": "repressor",
    "ZKSCAN1": "repressor",
    "ZKSCAN3": "repressor",
    "BCL11B": "repressor",      # mostly repressor at TCR genes
    "ZBED2": "repressor",
    "BHLHE41": "repressor",
    "ERF::SREBF2": "repressor", # ERF is a repressor
    "ZNF211": "repressor",
    "ZNF768": "repressor",
    "PRDM9": "repressor",       # methyltransferase, repressive mark setter

    # explicit unknowns / context-dependent — fall back to |effect|
    # (CTCF, ATF3, ZBTB7A, ZBTB14, NR1H3::RXRA, NR2C1, NR2F2, THRB ...)
}


def lookup(tf: str) -> str:
    """Return 'activator' / 'repressor' / 'unknown' for a TF name."""
    return TF_FUNCTION.get(tf, "unknown")


def expected_sign(tf: str) -> int:
    """+1 if activator (motif+ > motif−), -1 if repressor, 0 if unknown."""
    f = lookup(tf)
    if f == "activator":
        return 1
    if f == "repressor":
        return -1
    return 0
