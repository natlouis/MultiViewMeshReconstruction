## Metrics
It's worth noting that we could display model results on the dataset as a whole, or present results per-category, which could potentially provide insight on how the agent makes decisions.
1. __Total Distance Traveled__
* Want to discretize our viewpoint sphere enough so that it reduces redundancy between adjacent views
* Want to provide agent enough viewpoints to choose from to not constrain freedom of movement too much - don't want all Distance-Traveled scores to be very similar values in the end

2. __mIOU__
* Defined for voxels, and (maybe??) meshes
* Seems to be one of the standard metrics for voxel-based objects
* _Characteristics_: Not suitable for point clouds, high values can be misleading for solid-interior objects

3. __F-Score__
* Defined only for point-clouds
* Takes into account accuracy of and completeness of reconstruction - strictness controlled by varying threshold _d_
* https://lmb.informatik.uni-freiburg.de/Publications/2019/TB19/paper-s3d.pdf indicates F-Score much better indicator of prediction quality
* Working example in lines 209-229 in https://github.com/facebookresearch/meshrcnn/blob/1c8cbad1bc9a196f2e4fc0a80be6c12c35f1d1e3/meshrcnn/utils/metrics.py
* Working examples #2 at https://github.com/lmb-freiburg/what3d/blob/master/eval.py
* Mesh R-CNN evaluated F-Score at 3 different thresholds: https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/INSTRUCTIONS_SHAPENET.md

4. __Chamfer Distance__
* Defined only for point-clouds
* Used as a loss function for mesh-based objects in PyTorch3D Tutorial: https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
* Seems easy to implement: Working example in Mesh R-CNN repo (lines 156-205): https://github.com/facebookresearch/meshrcnn/blob/1c8cbad1bc9a196f2e4fc0a80be6c12c35f1d1e3/meshrcnn/utils/metrics.py
* _Characteristics_: Sensitive to outliers

## Conversion to Point-clouds
__Voxel-to-Point-Cloud__:
* Marching Cubes Algorithm (suggested in: https://lmb.informatik.uni-freiburg.de/Publications/2019/TB19/paper-s3d.pdf --- Sec. 5.2)

__Mesh-to-Point-Cloud__:
* Built-in function in PyTorch3D called "sample_points_from_meshes"
* Working example in lines 57-83 in https://github.com/facebookresearch/meshrcnn/blob/1c8cbad1bc9a196f2e4fc0a80be6c12c35f1d1e3/meshrcnn/utils/metrics.py
