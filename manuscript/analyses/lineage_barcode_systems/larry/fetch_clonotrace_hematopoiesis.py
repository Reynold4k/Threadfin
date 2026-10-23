"""Fetch the public in-vitro LARRY inputs used by Clonotrace (GSE140802).

The expression matrix is author-normalised UMI counts, not raw integer counts.
No clone, state or day labels are inferred from a published plot.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess


OUT = Path('/data/scratch/projects/punim1236/threadfin_data/gse140802_larry')
GEO = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4185nnn/GSM4185642/suppl/'
KLEIN = 'https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/'
NAMES = ['normed_counts.mtx.gz', 'gene_names.txt.gz', 'clone_matrix.mtx.gz',
         'metadata.txt.gz', 'cell_barcodes.txt.gz', 'library_names.txt.gz']


def fetch(item):
    name, url = item
    target = OUT / name
    if not target.exists():
        partial = target.with_suffix(target.suffix + '.part')
        subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                        '--connect-timeout', '30', '--continue-at', '-',
                        '--silent', '--show-error', '--output', str(partial), url], check=True)
        partial.replace(target)
    h = hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    result = {'file': name, 'url': url, 'bytes': target.stat().st_size, 'sha256': h.hexdigest()}
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True)
    items = [('stateFate_inVitro_'+n, GEO+'GSM4185642_stateFate_inVitro_'+n) for n in NAMES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, items))
    (OUT/'sources.json').write_text(json.dumps({'accession':'GSE140802 / GSM4185642',
        'source_description':'Author-normalised expression, row-matched metadata and binary lineage-barcode matrix',
        'files':records}, indent=2)+'\n')
