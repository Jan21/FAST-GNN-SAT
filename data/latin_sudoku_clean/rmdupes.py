#!/usr/bin/env python3
# File:  rmdups.py
# Created on:  Mon Mar 27 12:56:31 CEST 2023
import sys
import os
ll=''
for l in sys.stdin:
    l=l.strip()
    if l and ll:
        os.remove(l)
    ll=l
