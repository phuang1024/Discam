import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from features import *
from transform import solve_transform
from video_reader import VideoReader


def visualize_keypoints(img1, img2):
    (key1, des1), (key2, des2), matches = run_orb(img1, img2)
    vis = cv2.drawMatches(img1,key1,img2,key2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
    plt.imshow(vis)
    plt.show()


def visualize_transform(img1, img2, trans):
    """
    trans: Should be on data normalized [0, 1].
        Therefore, this function will multiply translation by the image shape.
    """
    trans = trans.detach().cpu().numpy()

    trans[0, 2] *= img2.shape[1]
    trans[1, 2] *= img2.shape[0]

    new_img = cv2.warpPerspective(img1, trans, img1.shape[1::-1])

    vis = np.zeros_like(img1)
    vis[..., 0] = np.mean(img2, axis=2)
    vis[..., 1] = np.mean(new_img, axis=2)

    plt.imshow(vis)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--keyframe", type=int, default=7)
    args = parser.parse_args()

    video = VideoReader(args.file, 640, args.keyframe)

    while True:
        from_pts, to_pts = get_keypoints(video[-1], video[0])
        trans = solve_transform(from_pts, to_pts)

        #visualize_keypoints(video[-1], video[0])
        visualize_transform(video[-1], video[0], trans)

        try:
            video.next()
        except ValueError:
            break


if __name__ == "__main__":
    main()
