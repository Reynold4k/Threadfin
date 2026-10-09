#!/usr/bin/env python3
"""Regenerate supplementary figures; --figures selects specific numbers."""
import argparse
import json
import make_biology_figures as biology
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--figures',nargs='+',type=int,choices=range(1,17),default=list(range(1,17)))
    args=parser.parse_args()
    audit=biology.HERE/'figure_audit.json'
    if audit.exists(): biology.AUDIT=json.loads(audit.read_text())
    for number in args.figures:
        getattr(biology,f'supplementary{number}')()
    biology.AUDIT['outputs']=list(dict.fromkeys(biology.AUDIT['outputs']))
    audit.write_text(json.dumps(biology.AUDIT,indent=2)+'\n')
