# WBSensor

## Overview

WBSensor is a wearable snesor dataset for group segmentation and activity recognition collected and constructed by our laboratory. It involves 21 participants aged 20-30 years. Each participant wears a laboratory-developed wristband on the right wrist. The wristband contains a tri-axial accelerometer and a tri-axial gyroscope. Individual two-dimensional positions `(x, y)` are collected and computed using the UWB positioning module embedded in the wristband.

## Dataset Statistics

| Property | Value |
| --- | --- |
| Participants | 21 |
| Participant age range | 20-30 years |
| Sensor measurements | Tri-axial acceleration and tri-axial angular velocity |
| Sensor placement | Right wrist |
| Spatial information | Two-dimensional UWB positions |
| Sampling frequency | 50 Hz |
| Action instance duration | 10 minutes |
| Subgroups per group sample | 3-5 |
| Random noise individuals per group sample | 1 |
| Individual action categories | 14 |
| Subgroup activity categories | 9 |
| Window duration | 4 seconds |
| Window overlap | 50% |
| Group activity sample sequences | 3,823 |
| Subgroup instances reported in the paper | 9,135 |

## Group Composition and Activities

Each group sample contains 3-5 subgroups. Individuals in the same subgroup perform the same subgroup activity. When a subgroup activity permits multiple individual actions, participants perform one of the allowed actions. Each group sample also includes one random noise individual whose action is independent of the other subgroup members.

The subgroup compositions and sample counts reported in Table II are:

| Subgroup activity | Subgroup composition | Training samples | Test samples |
| --- | --- | ---: | ---: |
| Exercising | 2-5 individuals walking or jogging | 812 | 203 |
| Smoking | 2-5 individuals smoking | 812 | 203 |
| Playing cards | 2-5 individuals playing cards | 812 | 203 |
| Chessing | 2-5 individuals playing chess | 812 | 203 |
| Dining | 2-5 individuals eating or drinking | 812 | 203 |
| Queuing | 2-5 individuals standing | 812 | 203 |
| Working | 2-5 individuals typing, writing, or reading | 812 | 203 |
| Fighting | 2-5 individuals fighting | 812 | 203 |
| Resting | 2-5 individuals sitting or lying | 812 | 203 |

## Group Sample Counts

Sliding-window segmentation with 4-second windows and 50% overlap produces 3,823 group activity sample sequences under different group-size settings. Table III reports:

| Individuals per group sample | Training samples | Test samples |
| ---: | ---: | ---: |
| 13 | 0 | 154 |
| 14 | 0 | 149 |
| 15 | 2,146 | 536 |
| 16 | 0 | 140 |
| 17 | 0 | 135 |
| 18 | 0 | 142 |
| 19 | 0 | 145 |
| 20 | 0 | 134 |
| 21 | 0 | 142 |

## Annotations

WBSensor provides:

- Individual action labels.
- Group activity labels for activity recognition.
- Group adjacency matrix labels for group segmentation.

## Reference

Ruohong Huan, Jian Cui, Gaoxiang Dong, Guodao Sun, Peng Chen, Ronghua Liang, and Chen Chen. *MsFEIRM: A Unified Framework for Group Segmentation and Activity Recognition Based on Multi-Scale Feature Extraction and Interaction Relationship Modeling.*
