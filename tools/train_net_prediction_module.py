#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import shutil
import time
import detectron2.utils.comm as comm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import inference_context
from fvcore.common.file_io import PathManager

from loss_prediction_module import LossPredictionModule

import numpy as np

from shapenet.config import get_shapenet_cfg
from shapenet.data import build_data_loader, register_shapenet
from shapenet.evaluation import evaluate_split, evaluate_test, evaluate_test_p2m

# required so that .register() calls are executed in module scope
from shapenet.modeling import MeshLoss, build_model
from shapenet.solver import build_lr_scheduler, build_optimizer
from shapenet.utils import Checkpoint, Timer, clean_state_dict, default_argument_parser

from shapenet.utils.coords import project_verts

from pytorch3d.structures import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    HardPhongShader,
    BlendParams,
    PointLights
)

#import wandb
logger = logging.getLogger("shapenet")

#dataset = "MeshVox"
dataset = "MeshVoxMulti"


def copy_data(args):
    data_base, data_ext = os.path.splitext(os.path.basename(args.data_dir))
    assert data_ext in [".tar", ".zip"]
    t0 = time.time()
    logger.info("Copying %s to %s ..." % (args.data_dir, args.tmp_dir))
    data_tmp = shutil.copy(args.data_dir, args.tmp_dir)
    t1 = time.time()
    logger.info("Copying took %fs" % (t1 - t0))
    logger.info("Unpacking %s ..." % data_tmp)
    shutil.unpack_archive(data_tmp, args.tmp_dir)
    t2 = time.time()
    logger.info("Unpacking took %f" % (t2 - t1))
    args.data_dir = os.path.join(args.tmp_dir, data_base)
    logger.info("args.data_dir = %s" % args.data_dir)


def main_worker_eval(worker_id, args):

    device = torch.device("cuda:%d" % worker_id)
    cfg = setup(args)

    # build test set
    test_loader = build_data_loader(cfg, dataset, "test", multigpu=False, num_workers=8)
    logger.info("test - %d" % len(test_loader))

    # load checkpoing and build model
    if cfg.MODEL.CHECKPOINT == "":
        raise ValueError("Invalid checkpoing provided")
    logger.info("Loading model from checkpoint: %s" % (cfg.MODEL.CHECKPOINT))
    cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))
    state_dict = clean_state_dict(cp["best_states"]["model"])
    model = build_model(cfg)
    model.load_state_dict(state_dict)
    logger.info("Model loaded")
    model.to(device)

    #wandb.init(project='MeshRCNN', config=cfg, name='meshrcnn-eval')
    if args.eval_p2m:
        evaluate_test_p2m(model, test_loader)
    else:
        evaluate_test(model, test_loader)


def main_worker(worker_id, args):
    distributed = False
    if args.num_gpus > 1:
        distributed = True
        dist.init_process_group(
            backend="NCCL", init_method=args.dist_url, world_size=args.num_gpus, rank=worker_id
        )
        torch.cuda.set_device(worker_id)

    device = torch.device("cuda:%d" % worker_id)

    cfg = setup(args)

    # data loaders
    loaders = setup_loaders(cfg)
    for split_name, loader in loaders.items():
        logger.info("%s - %d" % (split_name, len(loader)))

    # build the model
    model = build_model(cfg)
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[worker_id],
            output_device=worker_id,
            check_reduction=True,
            broadcast_buffers=False,
        )

    optimizer = build_optimizer(cfg, model)
    cfg.SOLVER.COMPUTED_MAX_ITERS = cfg.SOLVER.NUM_EPOCHS * len(loaders["train"])
    scheduler = build_lr_scheduler(cfg, optimizer)

    loss_fn_kwargs = {
        "chamfer_weight": cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "normal_weight": cfg.MODEL.MESH_HEAD.NORMALS_LOSS_WEIGHT,
        "edge_weight": cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT,
        "voxel_weight": cfg.MODEL.VOXEL_HEAD.LOSS_WEIGHT,
        "gt_num_samples": cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES,
        "pred_num_samples": cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES,
    }
    loss_fn = MeshLoss(**loss_fn_kwargs)

    checkpoint_path = "checkpoint.pt"
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, checkpoint_path)
    cp = Checkpoint(checkpoint_path)
    if len(cp.restarts) == 0:
        # We are starting from scratch, so store some initial data in cp
        iter_per_epoch = len(loaders["train"])
        cp.store_data("iter_per_epoch", iter_per_epoch)
    else:
        logger.info("Loading model state from checkpoint")
        model.load_state_dict(cp.latest_states["model"])
        optimizer.load_state_dict(cp.latest_states["optim"])
        scheduler.load_state_dict(cp.latest_states["lr_scheduler"])

    # Use pretrained voxmesh weights if supplied
    if not cfg.MODEL.CHECKPOINT == "":
        saved_weights = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))
        logger.info("Loading model from checkpoint: %s" % (cfg.MODEL.CHECKPOINT))

        state_dict = saved_weights["best_states"]["model"]
        state_dict = clean_state_dict(state_dict) #remove .module from key names
        model.load_state_dict(state_dict)

    training_loop(cfg, cp, model, optimizer, scheduler, loaders, device, loss_fn)


def training_loop(cfg, cp, model, optimizer, scheduler, loaders, device, loss_fn):

    #if comm.is_main_process():
    #    wandb.init(project='MeshRCNN', config=cfg, name='prediction_module')

    Timer.timing = False
    iteration_timer = Timer("Iteration")

    # model.parameters() is surprisingly expensive at 150ms, so cache it
    if hasattr(model, "module"):
        params = list(model.module.parameters())
    else:
        params = list(model.parameters())
    loss_moving_average = cp.data.get("loss_moving_average", None)

    # Zhengyuan modification
    loss_predictor = LossPredictionModule().to(device)
    loss_pred_optim = torch.optim.Adam(loss_predictor.parameters(), lr = 1e-5)

    while cp.epoch < cfg.SOLVER.NUM_EPOCHS:
        if comm.is_main_process():
            logger.info("Starting epoch %d / %d" % (cp.epoch + 1, cfg.SOLVER.NUM_EPOCHS))

        # When using a DistributedSampler we need to manually set the epoch so that
        # the data is shuffled differently at each epoch
        for loader in loaders.values():
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(cp.epoch)

        # Config settings for renderer
        render_image_size = 256
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=50,
        )
        rot_y_90 = torch.tensor([[0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [-1, 0, 0, 0],
                                 [0, 0, 0, 1]]).float().to(device)

        for i, batch in enumerate(loaders["train"]):
            if i == 0:
                iteration_timer.start()
            else:
                iteration_timer.tick()

            batch = loaders["train"].postprocess(batch, device)
            if dataset == 'MeshVoxMulti':
                imgs, meshes_gt, points_gt, normals_gt, voxels_gt, id_strs, _, render_RTs, RTs = batch
            else:
                imgs, meshes_gt, points_gt, normals_gt, voxels_gt = batch

            with inference_context(model):
                # NOTE: _imgs contains all of the other images in belonging to this model
                # We have to select the next-best-view from that list of images

                model_kwargs = {}
                if cfg.MODEL.VOXEL_ON and cp.t < cfg.MODEL.VOXEL_HEAD.VOXEL_ONLY_ITERS:
                    model_kwargs["voxel_only"] = True
                with Timer("Forward"):
                    voxel_scores, meshes_pred = model(imgs, **model_kwargs)

            total_silh_loss = torch.tensor(0.)  # Total silhouette loss, to be added to "loss" below
            # Voxel only training for first few iterations
            if not meshes_gt is None and not model_kwargs.get("voxel_only", False):
                _meshes_pred = meshes_pred[-1].clone()
                _meshes_gt = meshes_gt[-1].clone()

                # Render masks from predicted mesh for each view
                # GT probability map to supervise prediction module
                B = len(meshes_gt)
                probability_map = 0.01 * torch.ones((B, 24)).to(device)  # batch size x 24
                viewgrid = torch.zeros((B,24,render_image_size,render_image_size)).to(device) # batch size x 24 x H x W
                for b, (cur_gt_mesh, cur_pred_mesh) in enumerate(zip(meshes_gt, _meshes_pred)):
                    # Maybe computationally expensive, but need to transform back to world space based on rendered image viewpoint
                    RT = RTs[b]
                    # Rotate 90 degrees about y-axis and invert
                    invRT = torch.inverse(RT.mm(rot_y_90))
                    invRT_no_rot = torch.inverse(RT)  # Just invert

                    cur_pred_mesh._verts_list[0] = project_verts(
                        cur_pred_mesh._verts_list[0], invRT)
                    sid = id_strs[b].split('-')[0]

                    # For some strange reason all classes (expect vehicle class) require a 90 degree rotation about the y-axis
                    if sid == '02958343':
                        cur_gt_mesh._verts_list[0] = project_verts(
                            cur_gt_mesh._verts_list[0], invRT_no_rot)
                    else:
                        cur_gt_mesh._verts_list[0] = project_verts(
                            cur_gt_mesh._verts_list[0], invRT)

                    for iid in range(len(render_RTs[b])):

                        R = render_RTs[b][iid][:3, :3].unsqueeze(0)
                        T = render_RTs[b][iid][:3, 3].unsqueeze(0)
                        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
                        silhouette_renderer = MeshRenderer(
                            rasterizer=MeshRasterizer(
                                cameras=cameras,
                                raster_settings=raster_settings
                            ),
                            shader=SoftSilhouetteShader(blend_params=blend_params)
                        )

                        ref_image = (silhouette_renderer(meshes_world=cur_gt_mesh, R=R, T=T)>0).float()
                        image = (silhouette_renderer(meshes_world=cur_pred_mesh, R=R, T=T)>0).float()

                        #Add image silhouette to viewgrid
                        viewgrid[b,iid] = image[...,-1]
                        '''
                        import matplotlib.pyplot as plt
                        plt.subplot(1,2,1)
                        plt.imshow(ref_image[0,:,:,3].detach().cpu().numpy())
                        plt.subplot(1,2,2)
                        plt.imshow(image[0,:,:,3].detach().cpu().numpy())
                        plt.show()
                        '''

                        # MSE Loss between both silhouettes
                        silh_loss = torch.sum((image[0, :, :, 3] - ref_image[0, :, :, 3]) ** 2)
                        probability_map[b, iid] = silh_loss.detach()

                        total_silh_loss += silh_loss


                probability_map = probability_map/(torch.max(probability_map, dim=1)[0].unsqueeze(1))   # Normalize

                probability_map = torch.nn.functional.softmax(probability_map, dim=1).to(device)  # Softmax across images
                #nbv_idx = torch.argmax(probability_map, dim=1)  # Next-best view indices
                #nbv_imgs = _imgs[torch.arange(B), nbv_idx]  # Next-best view images

                # NOTE: Do a second forward pass through the model? This time for multi-view reconstruction
                # The input should be the first image and the next-best view
                #voxel_scores, meshes_pred = model(nbv_imgs, **model_kwargs)

                # Zhengyuan step loss_prediction
                predictor_loss = loss_predictor.train_batch(viewgrid, probability_map, loss_pred_optim)
                if comm.is_main_process():
                    #wandb.log({'prediction module loss':predictor_loss})
                
                    if cp.t % 50 == 0:
                        print('{} predictor_loss: {}'.format(cp.t, predictor_loss))

                    #Save checkpoint every t iteration
                    if cp.t % 500 == 0:
                        print('Saving loss prediction module at iter {}'.format(cp.t))
                        os.makedirs('./output_prediction_module', exist_ok=True)
                        torch.save(loss_predictor.state_dict(), './output_prediction_module/prediction_module_'+str(cp.t)+'.pth')

            cp.step()

            if cp.t % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                eval_and_save(model, loaders, optimizer, scheduler, cp)
        cp.step_epoch()
    eval_and_save(model, loaders, optimizer, scheduler, cp)

    if comm.is_main_process():
        logger.info("Evaluating on test set:")
        test_loader = build_data_loader(cfg, dataset, "test", multigpu=False)
        evaluate_test(model, test_loader)


def eval_and_save(model, loaders, optimizer, scheduler, cp):
    # NOTE(gkioxari) For now only do evaluation on the main process
    if comm.is_main_process():
        logger.info("Evaluating on training set:")
        train_metrics, train_preds = evaluate_split(
            model, loaders["train_eval"], prefix="train_", max_predictions=1000
        )
        eval_split = "val"
        if eval_split not in loaders:
            logger.info("WARNING: No val set!!! Computing metrics on test set!")
            eval_split = "test"
        logger.info("Evaluating on %s set:" % eval_split)
        test_metrics, test_preds = evaluate_split(
            model, loaders[eval_split], prefix="%s_" % eval_split, max_predictions=1000
        )
        str_out = "Results on train: "
        for k, v in train_metrics.items():
            str_out += "%s %.4f " % (k, v)
        logger.info(str_out)
        str_out = "Results on %s: " % eval_split
        for k, v in test_metrics.items():
            str_out += "%s %.4f " % (k, v)
        logger.info(str_out)

        # The main process is responsible for managing the checkpoint
        # TODO(gkioxari) revisit these stores
        """
        cp.store_metric(**train_preds)
        cp.store_metric(**test_preds)
        """
        cp.store_metric(**train_metrics)
        cp.store_metric(**test_metrics)
        cp.store_state("model", model.state_dict())
        cp.store_state("optim", optimizer.state_dict())
        cp.store_state("lr_scheduler", scheduler.state_dict())
        cp.save()

    # Since evaluation and checkpointing only happens on the main process,
    # make all processes wait
    if comm.get_world_size() > 1:
        dist.barrier()


def setup_loaders(cfg):
    loaders = {}
    loaders["train"] = build_data_loader(
        cfg, dataset, "train", multigpu=comm.get_world_size() > 1, num_workers=8
    )

    # Since sampling the mesh is now coupled with the data loader, we need to
    # make two different Dataset / DataLoaders for the training set: one for
    # training which uses precomputd samples, and one for evaluation which uses
    # more samples and computes them on the fly. This is sort of gross.
    loaders["train_eval"] = build_data_loader(cfg, dataset, "train_eval", multigpu=False)

    loaders["val"] = build_data_loader(cfg, dataset, "val", multigpu=False)
    return loaders


def setup(args):
    """
    Create configs and setup logger from arguments and the given config file.
    """
    cfg = get_shapenet_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # register dataset
    data_dir, splits_file = register_shapenet(cfg.DATASETS.NAME)
    cfg.DATASETS.DATA_DIR = data_dir
    cfg.DATASETS.SPLITS_FILE = splits_file
    # if data was copied the data dir has changed
    if args.copy_data:
        cfg.DATASETS.DATA_DIR = args.data_dir
    cfg.freeze()

    colorful_logging = not args.no_color
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    comm.synchronize()

    logger = setup_logger(
        output_dir, color=colorful_logging, name="shapenet", distributed_rank=comm.get_rank()
    )
    logger.info(
        "Using {} GPUs per machine. Rank of current process: {}".format(
            args.num_gpus, comm.get_rank()
        )
    )
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info(
        "Loaded config file {}:\n{}".format(args.config_file, open(args.config_file, "r").read())
    )
    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))
    return cfg


def shapenet_launch():
    args = default_argument_parser()

    # Note we need this only for pretrained models with torchvision.
    os.environ["TORCH_HOME"] = args.torch_home

    if args.copy_data:
        # if copy data is 1 then you need to provide args.data_dir
        # from which to copy data
        if args.data_dir == "":
            raise ValueError("You need to provide args.data_dir")
        copy_data(args)

    if args.eval_only:
        main_worker_eval(0, args)
        return

    if args.num_gpus > 1:
        mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,), daemon=False)
    else:
        main_worker(0, args)


if __name__ == "__main__":
    shapenet_launch()
