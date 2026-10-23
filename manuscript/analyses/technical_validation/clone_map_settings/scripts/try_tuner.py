import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表/Threadfin")
from load_feats import features
from threadfin.maptune import tune_clone_map
ds = sys.argv[1]; mc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
f, s, r, ids = features(ds, mc)
res = tune_clone_map(f, random_state=0)
print(res.candidates.round(3).to_string())
