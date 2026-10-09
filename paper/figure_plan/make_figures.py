#!/usr/bin/env python3
"""Regenerate main figures; use --figures 1 for the current Figure 1 only."""
import argparse
import json
import make_biology_figures as biology
from make_biology_figures import figure1, figure2, figure3, figure4, figure5, figure6
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--figures',nargs='+',type=int,choices=range(1,7),default=list(range(1,7)))
    args=parser.parse_args()
    audit_path=biology.HERE/'figure_audit.json'
    if audit_path.exists(): biology.AUDIT=json.loads(audit_path.read_text())
    drawing=[figure1,figure2,figure3,figure4,figure5,figure6]
    for number in args.figures: drawing[number-1]()
    biology.AUDIT['outputs']=list(dict.fromkeys(biology.AUDIT['outputs']))
    audit_path.write_text(json.dumps(biology.AUDIT,indent=2)+'\n')
