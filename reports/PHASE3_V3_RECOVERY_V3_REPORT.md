# Phase3 v3 Recovery v3

Status: pilot pending. The detached pipeline updates this report after its automatic gates.

## Previous dataset failure

Recovery v2 used 2,800 dialect samples framed as conversion. Its train split had
2,654 conversion prompts, approximately 1,327 paired targets shown twice, and no
dedicated meaning, ending-selection, or correction tasks. `아이가` appeared in
1,590 targets; `퍼뜩`, `단디`, `뭇나`, `온나`, `천지빼까리`, and `파이다` were absent.
The step-30 pilot retained factual, memory, and instruction scores but dialect
fell from 25.0% to 16.7%.

## v4 design

3,500 dialect skill examples: conversion 1,225; meaning 700; grammar/ending
selection 525; correction 525; contextual dialogue 525. The full mix has 1,909
factual/general and 954 memory/instruction replay examples. Completion-only loss
is used. The automatic run verifies target-token share and exact benchmark prompt
disjointness before training.

## Results

The detached pipeline writes pilot, full gate, and final results here.
