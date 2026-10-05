#!/usr/bin/env python3
"""Conservative exact-receptor check for the pure GSE253857 marrow gates.

This deliberately does not construct clone families or infer differentiation.
It asks only whether a receptor observed in a pure PC sort is *identical* to
one observed in a pure memory sort from the same donor.  A cell is retained
only with exactly one productive IGH and exactly one productive IGK or IGL
contig; missing or multi-contig chains are excluded rather than resolved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = Path("/data/scratch/projects/punim1236/threadfin_data/gse253857_bmpc")
MANIFEST = HERE / "bone_marrow_sample_manifest.tsv"

# Pure, single-donor GEX/BCR libraries verified in the GEO SOFT sample map.
LIBRARIES = {
    "1681": {"PC_BM": "BM_PCs_2", "Memory_BM": "BM_Bsm_2", "Memory_blood": "Blood_Bsm_2"},
    "1684": {"PC_BM": "BM_PCs_3", "Memory_BM": "BM_Bsm_3", "Memory_blood": "Blood_Bsm_3"},
}


def _productive_unique_receptors(library: str) -> pd.DataFrame:
    """One row per cell with one productive heavy and one productive light chain."""
    path = DATA / f"{library.replace('-', '_')}_BCR.filtered_contig_annotations.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"missing paired contig file for {library}: {path}")
    use = ["barcode", "chain", "productive", "v_gene", "j_gene", "cdr3_nt"]
    contigs = pd.read_csv(path, usecols=use, low_memory=False)
    contigs = contigs[contigs["productive"].astype(str).str.upper().isin(("TRUE", "T", "1"))].copy()
    contigs = contigs.dropna(subset=["barcode", "chain", "v_gene", "j_gene", "cdr3_nt"])
    heavy = contigs[contigs.chain.eq("IGH")].copy()
    light = contigs[contigs.chain.isin(("IGK", "IGL"))].copy()
    heavy_n = heavy.groupby("barcode", observed=True).size()
    light_n = light.groupby("barcode", observed=True).size()
    keep = heavy_n[heavy_n.eq(1)].index.intersection(light_n[light_n.eq(1)].index)
    heavy = heavy[heavy.barcode.isin(keep)].set_index("barcode")
    light = light[light.barcode.isin(keep)].set_index("barcode")
    # The preceding cardinality checks make these joins one-to-one by construction.
    paired = heavy.join(light, lsuffix="_h", rsuffix="_l", validate="one_to_one")
    paired["igh_key"] = (paired.v_gene_h + "|" + paired.j_gene_h + "|" + paired.cdr3_nt_h)
    paired["igh_light_key"] = (paired.igh_key + "|" + paired.chain_l + "|" + paired.v_gene_l + "|"
                               + paired.j_gene_l + "|" + paired.cdr3_nt_l)
    return paired.reset_index()[["barcode", "igh_key", "igh_light_key"]]


def _comparison_rows(donor: str, receptors: dict[str, pd.DataFrame]) -> tuple[list[dict], list[dict]]:
    rows, examples = [], []
    for left, right in (("PC_BM", "Memory_BM"), ("PC_BM", "Memory_blood"), ("Memory_BM", "Memory_blood")):
        for definition, key in (("exact_IGH", "igh_key"), ("exact_IGH_plus_light", "igh_light_key")):
            a = receptors[left][["barcode", key]].assign(source=left)
            b = receptors[right][["barcode", key]].assign(source=right)
            all_cells = pd.concat((a, b), ignore_index=True)
            counts = all_cells.groupby([key, "source"], observed=True).size().unstack("source", fill_value=0)
            counts = counts.reindex(columns=[left, right], fill_value=0)
            total = counts.sum(axis=1)
            eligible = counts[total.ge(2)]
            shared = eligible[(eligible[left] > 0) & (eligible[right] > 0)]
            rows.append({
                "donor": donor, "left_source": left, "right_source": right,
                "receptor_definition": definition,
                "left_unique_heavy_light_cells": int(len(a)), "right_unique_heavy_light_cells": int(len(b)),
                "eligible_groups_ge2_cells": int(len(eligible)), "shared_groups": int(len(shared)),
                "shared_left_cells": int(shared[left].sum()), "shared_right_cells": int(shared[right].sum()),
            })
            # Figure examples require identical heavy *and* light chains, not
            # the heavier-chain-only sensitivity analysis above.
            if definition == "exact_IGH_plus_light":
                for receptor, counts_row in shared.sort_values([left, right], ascending=False).iterrows():
                    examples.append({"donor": donor, "left_source": left, "right_source": right,
                                     "receptor_definition": definition, "receptor_key": receptor,
                                     "left_cells": int(counts_row[left]), "right_cells": int(counts_row[right])})
    return rows, examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=HERE / "results" / "bone_marrow_pure_gate_check")
    args = ap.parse_args()
    manifest = pd.read_csv(MANIFEST, sep="\t", dtype=str).set_index("library", verify_integrity=True)
    rows, examples = [], []
    for donor, libraries in LIBRARIES.items():
        for library in libraries.values():
            if library not in manifest.index:
                raise KeyError(f"{library} is absent from {MANIFEST}")
            design = manifest.loc[library]
            if not (design.design_status == "single_donor" and design.donor == donor):
                raise ValueError(f"{library} is not the expected single-donor {donor} library")
        receptors = {source: _productive_unique_receptors(library) for source, library in libraries.items()}
        donor_rows, donor_examples = _comparison_rows(donor, receptors)
        rows.extend(donor_rows); examples.extend(donor_examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = pd.DataFrame(rows)
    validation.to_csv(args.output_dir / "donor_validation.csv", index=False)
    pd.DataFrame(examples, columns=["donor", "left_source", "right_source", "receptor_definition",
                                    "receptor_key", "left_cells", "right_cells"]).to_csv(
                                        args.output_dir / "shared_receptor_examples.csv", index=False)


if __name__ == "__main__":
    main()
