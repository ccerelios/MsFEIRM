# UT-Group

## Overview

UT-Group is a group segmentation and activity recognition dataset constructed by our laboratory based on the publicly available UT-Data individual behavior dataset. It combines real wrist-worn tri-axial accelerometer and tri-axial gyroscope recordings with controlled subgroup composition and simulated spatial coordinates.

AnyLogic is used to simulate individual positions during dynamic activities such as walking and jogging. Individuals are assumed to move within the first quadrant, so the simulated position coordinates are positive. UT-Group serves as a controlled semi-synthetic benchmark with known subgroup memberships, spatial relations, activity labels, adjacency, subgroup cardinalities, and noise individuals.

## Dataset Statistics

| Property | Value |
| --- | --- |
| Sensor measurements | Tri-axial acceleration and tri-axial angular velocity |
| Sensor placement | Wrist |
| Spatial information | Simulated using AnyLogic |
| Sampling frequency | 50 Hz |
| Activity duration | 4 minutes |
| Individuals per group sample | 10, including one random noise individual |
| Subgroups per group sample | 3-5 |
| Subgroup activity categories | 9 |
| Individual action categories | 10 |
| Subgroup instances reported in the paper | 8,857 |

## Group Construction and Activities

Each group sample contains 3-5 subgroups formed by combining individual recordings from UT-Data. One random noise individual, whose action is independent of the other subgroups, is added to each sample. The total number of individuals is fixed at 10 in both training and test samples.

The subgroup compositions and sample counts reported in Table I are:

| Subgroup activity | Subgroup composition | Training samples | Test samples |
| --- | --- | ---: | ---: |
| Walking | 2-5 individuals walking | 786 | 197 |
| Queuing | 2-5 individuals standing | 786 | 197 |
| Jogging | 2-5 individuals jogging | 786 | 197 |
| Working | 2-5 individuals typing or writing | 786 | 197 |
| Resting | 2-5 individuals sitting | 788 | 197 |
| Speeching | One individual giving a speech, with 1-4 individuals sitting and writing | 788 | 197 |
| Smoking | 2-5 individuals smoking | 788 | 197 |
| Dining | 2-5 individuals eating or drinking | 788 | 197 |
| Examining | One individual standing, with 1-4 individuals writing | 788 | 197 |

## Annotations

UT-Group provides:

- Individual action labels.
- Group activity labels for activity recognition.
- Group adjacency matrix labels for group segmentation.

## Reference

Ruohong Huan, Jian Cui, Gaoxiang Dong, Guodao Sun, Peng Chen, Ronghua Liang, and Chen Chen. *MsFEIRM: A Unified Framework for Group Segmentation and Activity Recognition Based on Multi-Scale Feature Extraction and Interaction Relationship Modeling.*

The original UT-Data reference cited in the paper is:

Shoaib M, Bosch S, Incel O D, et al. "Complex Human Activity Recognition Using Smartphone and Wrist-Worn Motion Sensors." *Sensors*, 2016, 16(4): 426.



## Dataset Download

The raw sensor recordings are available from the GitHub Release:

- [UT-Group Dataset v1.0.0](https://github.com/ccerelios/UT-Group/releases/tag/v1.0.0)

The release contains:

- `smartphoneatpocket.csv`
- `smartphoneatwrist.csv`
