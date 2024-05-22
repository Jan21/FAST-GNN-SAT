/*
 * File:  auxiliary.cpp
 * Created on:  Mon Sep 4 15:26:08 WEST 2017
 */
#include "auxiliary.h"
#include <limits.h>

int strtonum(const char* s, int * const n) {
    const int M = INT_MAX;
    int i;
    long res = 0;
    for (i = 0; s[i] != '\0'; ++i) {
        if (s[i] < '0' || s[i] > '9') return 0;
        res = res * 10 + (s[i] - '0');
        if (res > M) return 0;
    }
    if (!i) return 0;
    *n = res;
    return 1;
}
