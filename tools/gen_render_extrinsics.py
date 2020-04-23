#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from multiprocessing import Pool
import torch
from detectron2.utils.logger import setup_logger
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from shapenet.utils.binvox_torch import read_binvox_coords
from shapenet.utils.coords import (
    SHAPENET_MAX_ZMAX,
    SHAPENET_MIN_ZMIN,
    compute_extrinsic_matrix,
    get_blender_intrinsic_matrix,
    project_verts,
)

from pytorch3d.renderer import look_at_view_transform

logger = logging.getLogger("preprocess")

IMAGE_EXTS = [".png", ".jpg", ".jpeg"]


def int_list(s):
    return [int(x) for x in s.split(",")]


def validcheck(summary, splits):
    valid = True
    for split_set in splits.keys():
        for sid in splits[split_set].keys():
            for mid in splits[split_set][sid].keys():
                if sid not in summary.keys():
                    logger.info("missing %s" % (sid))
                    valid = False
                else:
                    if mid not in summary[sid].keys():
                        logger.info("missing %s - %s" % (sid, mid))
                        valid = False
                    else:
                        if summary[sid][mid] < len(splits[split_set][sid][mid]):
                            logger.info(
                                "mismatch of images for %s - %s: %d vs %d"
                                % (sid, mid, len(splits[split_set][sid][mid]), summary[sid][mid])
                            )
                            valid = False
    return valid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r2n2_dir",
        default="./datasets/shapenet/ShapeNetRendering",
        help="Path to the Shapenet renderings as provided by R2N2",
    )
    parser.add_argument(
        "--output_dir", default="./datasets/shapenet/ShapeNetRenderingExtrinsics", help="Output directory"
    )
    parser.add_argument(
        "--models_per_synset",
        type=int,
        default=-1,
        help="How many models per synset to process. If -1, all will be processed",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--voxel_sizes", type=int_list, default=[])
    parser.add_argument("--zip_output", action="store_true", help="zips output")

    return parser.parse_args()


def main(args):
    setup_logger(name="preprocess")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_samples > 0:
        assert args.num_workers == 0

    # Maps sids to dicts which map mids to number of images
    summary = defaultdict(dict)

    # Walk the directory tree to find synset IDs and model IDs
    num_skipped = 0
    for sid in os.listdir(args.r2n2_dir):
        sid_dir = os.path.join(args.r2n2_dir, sid)
        if not os.path.isdir(sid_dir):
            continue
        logger.info('Starting synset "%s"' % sid)
        cur_mids = os.listdir(sid_dir)
        N = len(cur_mids)
        if args.models_per_synset > 0:
            N = min(N, args.models_per_synset)
        tasks = []
        for i, mid in enumerate(cur_mids):
            if os.path.exists(os.path.join(args.output_dir,sid,mid)):
                continue
            tasks.append((args, sid, mid, i, N))
        if args.models_per_synset > 0:
            tasks = tasks[: args.models_per_synset]
        if args.num_workers == 0:
            outputs = [handle_model(*task) for task in tasks]
        else:
            with Pool(processes=args.num_workers) as pool:
                outputs = pool.starmap(handle_model, tasks)

        num_skipped = 0
        for out in outputs:
            if out is None:
                num_skipped += 1
            else:
                sid, mid, num_imgs = out
                summary[sid][mid] = num_imgs

    # check that the pre processing completed successfully
    logger.info("Checking validity...")
    splits = json.load(open(args.splits_file, "r"))
    if not validcheck(summary, splits):
        raise ValueError("Pre processing identified missing data points")

    summary_json = os.path.join(args.output_dir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f)

    logger.info("Pre processing succeeded!")


def handle_model(args, sid, mid, i, N):
    r2n2_dir = os.path.join(args.r2n2_dir, sid, mid, "rendering")
    output_dir = os.path.join(args.output_dir, sid, mid)

    os.makedirs(output_dir)

    # Save metadata.pt
    image_list = load_image_list(args, sid, mid)
    extrinsics = load_extrinsics(args, sid, mid)
    metadata = {"image_list": image_list, "extrinsics": extrinsics}
    metadata_path = os.path.join(output_dir, "rendering_metadata.pt")
    torch.save(metadata, metadata_path)

    # Copy the images to the output directory
    output_img_dir = os.path.join(output_dir, "images")
    os.makedirs(output_img_dir)
    for fn in image_list:
        ext = os.path.splitext(fn)[1]
        assert ext in IMAGE_EXTS
        src = os.path.join(r2n2_dir, fn)
        dst = os.path.join(output_img_dir, fn)
        shutil.copy(src, dst)

    num_imgs = len(image_list)
    return (sid, mid, num_imgs)

def load_extrinsics(args, sid, mid):
    path = os.path.join(args.r2n2_dir, sid, mid, "rendering", "rendering_metadata.txt")
    extrinsics = []
    MAX_CAMERA_DISTANCE = 1.75  # Horrible constant from 3D-R2N2
    with open(path, "r") as f:
        for line in f:
            vals = [float(v) for v in line.strip().split(" ")]
            azimuth, elevation, yaw, dist_ratio, fov = vals
            distance = MAX_CAMERA_DISTANCE * dist_ratio

            #R and T matrices for use in rendering the model from specified viewpoints
            #azimuth runs in the opposite direction
            R, T = look_at_view_transform(dist=distance, elev=elevation, azim=-azimuth)

            extrinsics.append(torch.cat([R.squeeze(), T.t()], dim=1))
    extrinsics = torch.stack(extrinsics, dim=0)
    return extrinsics

def load_image_list(args, sid, mid):
    path = os.path.join(args.r2n2_dir, sid, mid, "rendering", "renderings.txt")
    with open(path, "r") as f:
        image_list = [line.strip() for line in f]
    return image_list

if __name__ == "__main__":
    args = parse_args()
    main(args)
    # Zips the output dir. This is needed if the user wants to copy
    # the data over to a cluster or machine for faster io
    if args.zip_output:
        logger.info("Archiving output...")
        shutil.make_archive(args.output_dir, "zip", base_dir=args.output_dir)
    logger.info("Done.")
