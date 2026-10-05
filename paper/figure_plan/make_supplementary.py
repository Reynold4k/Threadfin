#!/usr/bin/env python3
"""Regenerate the seven focused supplementary figures."""
from make_biology_figures import supplementary1, supplementary2, supplementary3, supplementary4, supplementary5, supplementary6, supplementary7
if __name__ == '__main__':
    for draw in [supplementary1, supplementary2, supplementary3, supplementary4, supplementary5, supplementary6, supplementary7]:
        draw()
