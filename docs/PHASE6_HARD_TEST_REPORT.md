# Phase 6 Hard-Test Report

Status: **not run**. `dataset/hard_test/` is a reserved evaluation-only location and currently has no user-provided labeled scenes. No hard-test results are claimed.

## Required inputs

Place unseen, permission-cleared images in `dataset/hard_test/images/` and corresponding YOLO annotations in `dataset/hard_test/labels/`. Preserve the source URL/owner, license or permission basis, capture date where known, expected object class/count, and a short scene note in `dataset/hard_test/manifest.csv`. Do not place these images in any train or validation directory. Record whether each scene contains multiple plastic bottles, glass bottles, metal cans, paper, cardboard, mixed waste, overlap, occlusion, small targets, or clutter.

The Colab procedure must compare baseline and selected candidate on the exact same fixed image set, save both annotated predictions, and compute matched true positives, false positives, and false negatives from the labels. Human-review columns should record confidence and class confusion per object. Hard-test images remain evaluation-only.

| Scene ID | Source/permission | Expected objects | Baseline TP/FP/FN | Candidate TP/FP/FN | Class confusions / notes |
|---|---|---:|---:|---:|---|
| No labeled hard-test scenes supplied | — | — | Not measured | Not measured | No claims |

## Acceptance caution

This hard set must be collected independently of training and validation and must not be used to tune thresholds, augmentations, or select checkpoints. With no images and labels, hard-test improvement cannot currently be established.
