# 3D Ball Trajectory from a Single Camera

This repository contains the implementation of a **cost‑effective single‑camera 3D cricket ball tracking system** developed for the paper *“3D Ball Trajectory from a Single Camera: A Cost-Effective Ball Estimation Technique for Cricket”*. The project reconstructs the 3D flight of a cricket ball using monocular video, standardized pitch geometry, and camera calibration, avoiding the need for expensive multi‑camera infrastructure typically used in professional DRS systems.

## Project Description

The system takes high‑resolution cricket videos from a calibrated camera positioned over the pitch and detects the ball in each frame using a CNN‑based detector (YOLO‑NAS in the paper) trained on a custom annotated dataset. From the detected bounding boxes, the pipeline extracts precise 2D pixel information by ROI cropping, HSV‑based red‑ball segmentation, contour detection, and minimum enclosing circle fitting to obtain the ball center and apparent radius in pixels.

Using the pinhole camera model, camera intrinsics and extrinsics, and the known real‑world ball diameter (71.5 mm), the project recovers the 3D position of the ball in world coordinates for every frame, expressed in millimeters and aligned with standardized pitch dimensions. Because single‑camera depth estimation is noisy, a physics‑based Kalman filter with a ballistic motion model (state \([x,y,z,v_x,v_y,v_z]\) under gravity) is applied to smooth the trajectory, estimate velocities, and enforce physically plausible motion.

On the smoothed 3D trajectory, the system automatically detects the **bounce point** by finding when the ball reaches the pitch plane and its vertical velocity changes sign, and then uses the estimated state to **extrapolate the post‑bounce path** towards the stumps for LBW‑style decision analysis. The final 3D paths are visualized on a virtual pitch model built in Blender using official cricket pitch dimensions, providing an interpretable 3D reconstruction comparable in spirit to commercial DRS graphics but based on a single calibrated camera.

## Key Features

- Single‑camera 3D trajectory reconstruction using camera calibration and known ball size, eliminating the need for multi‑camera rigs.
- CNN‑based ball detection (YOLO‑NAS) with competitive mAP across multiple IoU thresholds on a custom cricket dataset.
- Robust 2D pixel extraction via HSV segmentation and contour‑based minimum‑enclosing‑circle fitting tailored for red cricket balls.
- 3D Kalman filtering with a ballistic motion model to suppress depth jitter and recover consistent velocity estimates.
- Automatic bounce point localization and forward trajectory prediction suitable for LBW decision support scenarios.
- Blender‑based 3D visualization on a physically accurate virtual pitch constructed from standardized cricket pitch geometry.

## Citation

If you use this project or its methodology in academic or applied work, please cite the associated paper:

```bibtex
@article{ghimire2026singlecamera,
  title   = {3D Ball Trajectory from a Single Camera: A Cost-Effective Ball Estimation Technique for Cricket},
  author  = {Ghimire, Bijaya and Mishra, Gunjan K. and Lamichhane, Badri Raj},
  journal = {TechRxiv},
  year    = {2026},
  doi     = {10.36227/techrxiv.176761698.86519755},
  note    = {Preprint},
  url     = {https://www.techrxiv.org/doi/full/10.36227/techrxiv.176761698.86519755/v1}
}

