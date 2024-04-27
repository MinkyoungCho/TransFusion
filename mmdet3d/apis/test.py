import mmcv
import torch
import numpy as np

# from tools.visualize_results import visualize_bbox

def gaussian_noise(pointcloud, severity=5): # Reference: https://github.com/thu-ml/3D_Corruptions_AD/blob/main/LiDAR_corruptions.py
    pointcloud = pointcloud.cpu().numpy()
    
    N, C = pointcloud.shape # N*3
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity - 1]
    jitter = np.random.normal(size=(N, 3)) * c
    # jitter = np.random.uniform(-c, c, (N, 3))
    pointcloud[:,:3] = (pointcloud[:,:3] + jitter).astype('float32')
    
    pointcloud = torch.from_numpy(pointcloud)
    pointcloud = pointcloud.float()
    
    return pointcloud.cuda()    
    
    return pointcloud

def fov_filter(points, severity=5):
    points = points.cpu().numpy()

    angle1 = [-105, -90, -75, -60, -45][severity-1]
    angle2 = [105, 90, 75, 60, 45][severity-1]
    if isinstance(points, np.ndarray):
        pts_npy = points
    pts_p = (np.arctan(pts_npy[:, 0] / pts_npy[:, 1]) + (
                pts_npy[:, 1] < 0) * np.pi + np.pi * 2) % (np.pi * 2)
    pts_p[pts_p > np.pi] -= np.pi * 2
    pts_p = pts_p / np.pi * 180
    assert np.all(-180 <= pts_p) and np.all(pts_p <= 180)
    filt = np.logical_and(pts_p >= angle1, pts_p <= angle2)
    pointcloud = points[filt] 
    
    pointcloud = torch.from_numpy(pointcloud)
    pointcloud = pointcloud.float()


    return pointcloud

def density_dec_global(pointcloud, severity=5):
    pointcloud = pointcloud.cpu().numpy()

    N, C = pointcloud.shape
    num = int(N * 0.3)
    c = [int(0.2*num), int(0.4*num), int(0.6*num), int(0.8*num), num][severity - 1]
    idx = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx, axis=0)
    pointcloud = torch.from_numpy(pointcloud)
    pointcloud = pointcloud.float().cuda()

    return pointcloud


def cutout_local(pointcloud, severity=3):
    pointcloud = pointcloud.cpu().numpy()
    N, C = pointcloud.shape
    num = int(N*0.02)
    c = [(2,num), (3,num), (5,num), (7,num), (10,num)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        
    pointcloud = torch.from_numpy(pointcloud)
    pointcloud = pointcloud.float().cuda()

    return pointcloud


def voxelize(pts, voxel_size=0.5, num_T=35, seed: float = None):
    """
    Voxelize the input point cloud. Code modified from https://github.com/Yc174/voxelnet
    Voxels are 3D grids that represent occupancy info.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be -1 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param num_T: Number of points in each voxel after sampling/padding
    :param seed: The random seed for fixing the data generation.
    """
    extents = np.array([[-64., 64.], [-64., 64.], [-4, 4]])

    filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                            (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                            (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]
    pts = pts[filter_idx].cpu().numpy()

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    points = pts[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Number of points per voxel, last voxel calculated separately
    num_points_in_voxel = np.diff(unique_indices)
    num_points_in_voxel = np.append(num_points_in_voxel, discrete_pts.shape[0] - unique_indices[-1])

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)
    float_voxel_coords = voxel_coords * voxel_size
    float_voxel_coords = torch.from_numpy(float_voxel_coords)
    float_voxel_coords = float_voxel_coords.float()
    

    return float_voxel_coords.cuda()


def single_gpu_test(model, data_loader, show=False, out_dir=None):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    corruption = "no_corrupt"
    
    print()
    print (f"Corruption: {corruption}")

    for i, data in enumerate(data_loader):
        if corruption == "cam_b" or corruption == "cam_n" or corruption == "cam_z":
            cnt = 6
            if corruption == "cam_b":
                print ("corruption: cam_blackout")
                for tmp_cnt in range(cnt):
                    data['img'][0].data[0][0][tmp_cnt][0] = torch.ones_like(data['img'][0].data[0][0][0][0]) * (0 - 123.675) / 58.395
                    data['img'][0].data[0][0][tmp_cnt][1] = torch.ones_like(data['img'][0].data[0][0][0][1]) * (0 - 116.28) / 57.12
                    data['img'][0].data[0][0][tmp_cnt][2] = torch.ones_like(data['img'][0].data[0][0][0][2]) * (0 - 103.53) / 57.375

                    # data['img'][0].data[0][0][tmp_cnt][0] = torch.ones_like(data['img'][0].data[0][0][0][0]) *  (-103.530) # (0 - 123.675) / 58.395
                    # data['img'][0].data[0][0][tmp_cnt][1] = torch.ones_like(data['img'][0].data[0][0][0][1]) *  (-116.280) #(0 - 116.28) / 57.12
                    # data['img'][0].data[0][0][tmp_cnt][2] = torch.ones_like(data['img'][0].data[0][0][0][2]) *  (-123.675) #(0 - 103.53) / 57.375
            
            if corruption == "cam_z":
                print ("corruption: cam_zero")
                for tmp_cnt in range(cnt):
                    data['img'][0].data[0][0][tmp_cnt][0] = torch.zeros_like(data['img'][0].data[0][0][0][0]) 
                    data['img'][0].data[0][0][tmp_cnt][1] = torch.zeros_like(data['img'][0].data[0][0][0][1]) 
                    data['img'][0].data[0][0][tmp_cnt][2] = torch.zeros_like(data['img'][0].data[0][0][0][2]) 

            elif corruption == "cam_n":        
                print ("corruption: cam_noise")
                for tmp_cnt in range(cnt):
                    original_channel_r = (data['img'][0].data[0][0][tmp_cnt][0] + torch.ones_like(data['img'][0].data[0][0][0][0]) * (103.530)).to(torch.uint8)
                    original_channel_g = (data['img'][0].data[0][0][tmp_cnt][1] + torch.ones_like(data['img'][0].data[0][0][0][1]) * (116.280)).to(torch.uint8)
                    original_channel_b = (data['img'][0].data[0][0][tmp_cnt][2] + torch.ones_like(data['img'][0].data[0][0][0][2]) * (123.675)).to(torch.uint8)
                    
                    # tmp_noise = np.random.normal(0, 1, data['img'][0].data[0][0][0].shape)
                    tmp_noise = np.random.uniform(-1, 1, data['img'][0].data[0][0][0].shape)
                    noise = tmp_noise * 50                
                    noisy_image_r = original_channel_r + noise[0]
                    noisy_image_g = original_channel_g + noise[1]
                    noisy_image_b = original_channel_b + noise[2]
                    
                    noisy_image_r_clipped = np.clip(noisy_image_r, 0, 255).to(torch.uint8)
                    noisy_image_g_clipped = np.clip(noisy_image_g, 0, 255).to(torch.uint8)
                    noisy_image_b_clipped = np.clip(noisy_image_b, 0, 255).to(torch.uint8)
                    
                    data['img'][0].data[0][0][tmp_cnt][0] = noisy_image_r_clipped - torch.ones_like(data['img'][0].data[0][0][0][0]) * (103.530)
                    data['img'][0].data[0][0][tmp_cnt][1] = noisy_image_g_clipped - torch.ones_like(data['img'][0].data[0][0][0][1]) * (116.280)
                    data['img'][0].data[0][0][tmp_cnt][2] = noisy_image_b_clipped - torch.ones_like(data['img'][0].data[0][0][0][2]) * (123.675)
        
        if corruption == "lidar":
            ### LiDAR Corruption        
            # print ("corruption: lidar_noise_me")
            # noise = torch.normal(mean=0, std=0.01, size=(data['points'][0].data[0][0].shape[0], 3))
            # data['points'][0].data[0][0][:, :3] += noise
            
            # # ### LiDAR Voxelization    
            # print ("corruption: lidar_noise_uniform")
            # data['points'][0].data[0][0] = voxelize(data['points'][0].data[0][0])

            # print ("corruption: lidar_noise_gaussian")
            # data['points'][0].data[0][0] = gaussian_noise(data['points'][0].data[0][0])

            print ("corruption: lidar_densitydec")
            data['points'][0].data[0][0] = density_dec_global(data['points'][0].data[0][0])

            # print ("corruption: lidar_fov")
            # data['points'][0].data[0][0] = fov_filter(data['points'][0].data[0][0])

        with torch.no_grad():
            result, top3_img_values, top3_pts_values, top3_img_indices, top3_pts_indices = model(return_loss=False, rescale=True, **data)

        if show:
            model.module.show_results(data, result, out_dir)

        results.extend(result)
        
        topk_top3_values_dict = {}
        topk_top3_indices_dict = {}
        
        topk_top3_values_dict["img0"] = top3_img_values[0].permute(1,0)
        topk_top3_values_dict["pts0"] = top3_pts_values[0].permute(1,0)
        
        topk_top3_indices_dict["img0"] = top3_img_indices[0].permute(1,0)
        topk_top3_indices_dict["pts0"] = top3_pts_indices[0].permute(1,0)
        

        # if True: #visualize:
        #     visualize_bbox(
        #         data=data,
        #         data_name=f"sample_{i}",
        #         outputs=result[0]["pts_bbox"],
        #         cfg=None,
        #         topk_ncscore_dict=None, #topk_ncscore_dict, 
        #         topk_p_val_dict=None, #topk_p_val_dict,
        #         topk_top3_values_dict=topk_top3_values_dict,
        #         topk_top3_indices_dict=topk_top3_indices_dict,
        #     )

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
