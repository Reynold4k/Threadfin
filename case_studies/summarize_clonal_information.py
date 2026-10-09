#!/usr/bin/env python3
"""Export cross-study evidence from saved, library-stratified analyses.

This collects results; it never pools subjects, reruns a test or imputes missing
gene-set measurements. Coherence is descriptive expression organisation, not
validation of cell fate. Gene-set contrasts use expression-matched backgrounds.
"""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = {
    'mouse_np': ('NP-OVA reporter', 'GC anchors', 'Division reporter / sort gates'),
    'mouse_rbd': ('RBD reporter', 'GC anchors', 'Division reporter / zone / probe'),
    'gc_np_pc': ('NP-OVA PC/GC', 'GC anchors', 'Measured PC / GC sort gates'),
    'malaria': ('PcAS early', 'GC anchors', 'Same-mouse annotated state occupancy'),
    'malaria_late': ('PcAS late', 'GC anchors', 'State occupancy / anti-malarial treatment'),
    'ln_vaccine': ('Human vaccine GC', 'GC anchors', 'Same-donor repeated GC / binding labels'),
    'bone_marrow_pc': ('Marrow / blood', 'Non-GC anchor', 'Pure PC / memory gates; exact H+L'),
    'flu': ('Influenza blood', 'Tested coverage', 'Exploratory expression states'),
    'flu_lung': ('Influenza lung', 'Tested coverage', 'Exploratory expression states'),
    'ebv': ('EBV organoids', 'Tested coverage', 'Measured GFP status in infected tonsil organoids'),
    'tonsil': ('Tonsil', 'Tested coverage', 'Exploratory expression states'),
    'stephenson': ('COVID blood', 'Tested coverage', 'Exploratory expression states'),
}

def main():
    out = HERE / 'results/clonal_information_summary'
    out.mkdir(exist_ok=True)
    rows, modules = [], []
    for ds, (label, tier, anchor) in MODELS.items():
        path = HERE / 'results' / ds
        s = json.loads((path / 'summary.json').read_text())
        q, c = s['qc'], s['coherence']
        t = pd.read_csv(path / 'clone_table.csv', index_col=0)
        rows.append(dict(dataset=ds, label=label, evidence_tier=tier, biological_anchor=anchor,
                         n_cells=q['n_cells'], n_bcr_cells=q['n_cells_bcr'],
                         n_donors=q['n_donors'], n_samples=q['n_samples'],
                         expanded_clones=c['n_clones'], cells_in_expanded=c['n_cells'],
                         reliable_profiles=int(t.reliability.ge(.5).sum()),
                         observed=c['icc'], shuffled=c['null_mean'], shuffled_q95=c['null_q95'],
                         excess=c['excess'], p_value=c['p_value'],
                         permutation_stratum='sample/library', n_perm=500,
                         programmes=s.get('programmes', {}).get('n', 0)))
        cov = s.get('signature_coverage', {})
        # The older lung output predates the mouse-symbol correction. Missing
        # coverage is not zero biological signal; omit it from module contrasts.
        if ds == 'flu_lung' and not cov:
            continue
        if (path / 'geneset_heritability.csv').exists():
            g = pd.read_csv(path / 'geneset_heritability.csv')
            for r in g.to_dict('records'):
                if r['n_genes'] >= 3:
                    modules.append(dict(dataset=ds, label=label, **r))
    pd.DataFrame(rows).to_csv(out / 'dataset_evidence.csv', index=False)
    pd.DataFrame(modules).to_csv(out / 'module_evidence.csv', index=False)
    (out / 'provenance.json').write_text(json.dumps({
        'source': 'case_studies/results/<dataset>/summary.json and saved clone/gene-set tables',
        'coherence': 'Receptor-excluded expression PCs; context-centred clones; clone identities shuffled within sample/library, preserving each clone size and the cells of each library.',
        'interpretation': 'An excess shows within-library clonal expression resemblance; it does not isolate antigen specificity, establish lineage direction, or predict fate.',
        'modules': 'Median gene-level excess ICC relative to an expression-matched background, not a fate prediction. Mouse symbols are case-corrected, not curated orthologue translations.',
        'excluded_module_output': 'None in the 2026-10-09 rerun; datasets lacking signature coverage are skipped explicitly.',
        'independence': 'malaria and malaria_late are separate experiments in one publication; mouse_np and mouse_rbd are separate cohorts in one publication. Twelve analyses are not twelve independent publications.'
    }, indent=2))

if __name__ == '__main__':
    main()
