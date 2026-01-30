# Search and Rescue – Computer Vision Based Assignment System

## Overview

This project implements a complete computer vision pipeline to solve the *Search and Rescue* task as specified in the UAS-DTU Round 2 problem statement. The system processes aerial images to detect rescue pads and casualties, classify them by shape and color, compute priority scores, optimally assign casualties to rescue camps under capacity constraints, and generate the required outputs and visualizations.

The solution is rule-based and deterministic, relying on classical computer vision techniques rather than deep learning to ensure transparency, reproducibility, and robustness.

---

## Problem Summary

Given a set of aerial images containing:
- Rescue Pads (circles of different colors)
- Casualties (stars, triangles, squares of different colors)
- Background (ocean and land)

The task is to:
1. Detect and classify all shapes.
2. Interpret shape and color semantics (age group and emergency level).
3. Assign casualties to rescue pads based on a defined priority function and pad capacity.
4. Compute per-image and per-camp priority scores.
5. Rank images based on overall rescue priority.

---

## Shape and Semantic Mapping

### Casualties

| Shape     | Age Group | Age Priority |
|----------|----------|--------------|
| Star     | Child    | 3 |
| Triangle | Elderly  | 2 |
| Square  | Adult    | 1 |

### Emergency Level

| Color  | Severity | Priority |
|------|----------|----------|
| Red  | Severe   | 3 |
| Yellow | Mild   | 2 |
| Green | Safe    | 1 |

### Rescue Pads

| Color | Capacity |
|------|----------|
| Blue | 4 |
| Pink | 3 |
| Grey | 2 |

---

## Approach

### 1. Pre-processing and Background Separation
- Convert image to HSV color space.
- Segment ocean and land using tuned HSV ranges.
- Invert background mask to isolate objects of interest.
- Apply morphological operations to clean noise while preserving shape geometry.

### 2. Shape Protection Mask
- Detect valid contours.
- Build a dilated protection mask to prevent background recoloring from corrupting detected shapes.

### 3. Shape Detection and Classification
- Extract contours from the shapes mask.
- Compute centroids using spatial moments.
- Approximate contours using `approxPolyDP`.
- Classify shapes based on vertex count, extent, and circularity.
- Determine shape color using median HSV values sampled from shape interiors.

### 4. Priority Computation

For each rescue pad–casualty pair:
- Distance is the Euclidean distance between pad and casualty centroids.
- All possible pairs are ranked globally by priority score.

### 5. Assignment Strategy
- Casualties are assigned greedily in descending priority order.
- Each casualty is assigned only once.
- Rescue pad capacity constraints are strictly enforced.

---

## Outputs Generated

### Output 1 – Casualty Distribution per Image
For each image, casualties assigned to each camp are reported as:

### Output 2 – Camp Priority Scores
For each image:
- Blue camp priority score
- Pink camp priority score
- Grey camp priority score
- Total priority score

### Output 3 – Priority Ratio
Average priority score per casualty for each image.

### Output 4 – Image Ranking
Images sorted in descending order of priority ratio.

---

## Visual Outputs

For each image:
- Color-graded background (enhanced ocean and land).
- Detected shapes outlined with contours.
- Centroids marked.
- Assignment arrows drawn from rescue pads to assigned casualties.

Generated outputs are saved in:
- `color_graded_output/`
- `final_output/`

---

## Project Structure


---

## How to Run

1. Install dependencies:

2. Place input images in `task_images/` named `1.png` to `10.png`. (can change this as well, but would need to modify a few things accordingly)

3. Run:

4. Outputs will be generated automatically.

---

## Key Design Choices

- Classical computer vision over deep learning for deterministic behavior.
- Median HSV sampling to reduce boundary noise.
- Object-level filtering instead of aggressive morphology to preserve geometry.
- Separation between internal data representation and final output formatting.

---

## Author

**Yajat Malik**

This repository contains a complete and self-contained solution for the Search and Rescue task, adhering strictly to the problem specification.

