# Threadfin discovery benchmark

Runs the v3 discovery layer (antigen specificity, SHM maturation, migration,
gene programmes) on the five public datasets, reusing the loaders and configs
from `benchmarks/biological_validation/`.

```bash
# single dataset (small ones run on a laptop in minutes)
python run_discovery.py ../biological_validation/configs/stephenson2021.json

# full set on a cluster
sbatch run_discovery.sbatch ../biological_validation/configs/ln_vaccine_gse195673.json
```

Outputs land in `results/<dataset>/`: `discovery.json` (machine-readable
summary), one CSV per analysis, and figures where applicable. The LN config
uses the authors' spike-specificity calls, ELISA-validated mAb labels and
per-sequence SHM frequencies as held-out ground truth; Stephenson uses
[CoV-AbDab](https://opig.stats.ox.ac.uk/webapps/covabdab/) reference matching
(8.5 MB CSV, not committed).
