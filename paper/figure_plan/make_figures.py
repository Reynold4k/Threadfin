#!/usr/bin/env python3
"""Regenerate the six GC-focused main figures from completed source results."""
from make_biology_figures import figure1, figure2, figure3, figure4, figure5, figure6
if __name__ == '__main__':
    for draw in [figure1, figure2, figure3, figure4, figure5, figure6]:
        draw()
