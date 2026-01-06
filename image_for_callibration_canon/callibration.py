#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json, argparse, math
import numpy as np
import cv2

def parse_args():
    ap = argparse.ArgumentParser(description="Calibrate camera intrinsics from chessboard images.")
    ap.add_argument("--images", required=True,
                    help="Glob or folder with chessboard images (e.g. 'data/*.jpg').")
    ap.add_argument("--pattern_cols", type=int, required=True,
                    help="Number of internal corners along columns (width).")
    ap.add_argument("--pattern_rows", type=int, required=True,
                    help="Number of internal corners along rows (height).")
    ap.add_argument("--square_size", type=float, required=True,
                    help="Square size in meters (or chosen units).")
    ap.add_argument("--fisheye", action="store_true",
                    help="Use fisheye model instead of pinhole.")
    ap.add_argument("--show", action="store_true",
                    help="Visualize detections and an undistorted preview.")
    ap.add_argument("--out", default="intrinsics.json",
                    help="Output JSON path.")
    return ap.parse_args()

def find_images(path_glob):
    if os.path.isdir(path_glob):
        ims = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            ims.extend(glob.glob(os.path.join(path_glob, ext)))
    else:
        ims = glob.glob(path_glob)
    return sorted(ims)

def build_object_points(cols, rows, square_size):
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
    objp *= square_size
    return objp

def calibrate_pinhole(img_size, objpoints, imgpoints):
    flags = cv2.CALIB_RATIONAL_MODEL
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None, flags=flags)
    # reprojection error
    total_err, total_pts = 0, 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        total_err += err**2
        total_pts += len(objpoints[i])
    rms = math.sqrt(total_err/total_pts)
    return K, dist, rms

def main():
    args = parse_args()
    images = find_images(args.images)
    if not images:
        print("No images found!"); return

    pattern_size = (args.pattern_cols, args.pattern_rows)
    objp = build_object_points(*pattern_size, args.square_size)

    objpoints, imgpoints = [], []
    img_size = None

    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        ok, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not ok: continue
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
        objpoints.append(objp)
        imgpoints.append(corners)

        if args.show:
            disp = img.copy()
            cv2.drawChessboardCorners(disp, pattern_size, corners, ok)
            cv2.imshow("corners", disp); cv2.waitKey(200)

    if args.show: cv2.destroyAllWindows()

    K, dist, rms = calibrate_pinhole(img_size, objpoints, imgpoints)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    print("=== Intrinsics ===")
    print("Image size:", img_size)
    print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    print("distortion:", dist.ravel().tolist())
    print("RMS reproj error:", rms)

    out = {
        "image_size": {"width": img_size[0], "height": img_size[1]},
        "fx": float(fx), "fy": float(fy),
        "ox": float(cx), "oy": float(cy),
        "distortion": [dist.ravel().tolist()],
        "rms_reprojection_error": float(rms)
    }
    with open(args.out,"w") as f: json.dump(out,f,indent=2)
    print("Saved to", args.out)

if __name__=="__main__":
    main()
