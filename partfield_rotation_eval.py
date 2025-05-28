from partfield.config import default_argument_parser, setup
import torch
import glob
import os, sys
import numpy as np
import random
import trimesh
from plyfile import PlyData, PlyElement
import pymeshlab
from partfield_model_isolated.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
from partfield_model_isolated.triplane import TriplaneTransformer
import open3d as o3d
import time
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
# NOTE: This script is for evaluating rotation consistency of PartField features

#########################
## To handle quad inputs
#########################
def quad_to_triangle_mesh(F):
    """
    Converts a quad-dominant mesh into a pure triangle mesh by splitting quads into two triangles.

    Parameters:
        quad_mesh (trimesh.Trimesh): Input mesh with quad faces.

    Returns:
        trimesh.Trimesh: A new mesh with only triangle faces.
    """
    faces = F

    ### If already a triangle mesh -- skip
    if len(faces[0]) == 3:
        return F

    new_faces = []

    for face in faces:
        if len(face) == 4:  # Quad face
            # Split into two triangles
            new_faces.append([face[0], face[1], face[2]])  # Triangle 1
            new_faces.append([face[0], face[2], face[3]])  # Triangle 2
        else:
            print(f"Warning: Skipping non-triangle/non-quad face {face}")

    new_faces = np.array(new_faces)

    return new_faces
#########################


def load_mesh_util(input_fname):
    mesh = trimesh.load(input_fname, force='mesh', process=False)
    return mesh


def load_ply_to_numpy(filename):
    """
    Load a PLY file and extract the point cloud as a (N, 3) NumPy array.

    Parameters:
        filename (str): Path to the PLY file.

    Returns:
        numpy.ndarray: Point cloud array of shape (N, 3).
    """
    ply_data = PlyData.read(filename)

    # Extract vertex data
    vertex_data = ply_data["vertex"]

    # Convert to NumPy array (x, y, z)
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T

    return points


def get_model(ply_file, is_pc, data_path, preprocess_mesh, result_name, pc_num_pts):

    uid = ply_file.split(".")[-2].replace("/", "_")

    ####
    if is_pc:
        ply_file_read = os.path.join(data_path, ply_file)
        pc = load_ply_to_numpy(ply_file_read)

        bbmin = pc.min(0)
        bbmax = pc.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        pc = (pc - center) * scale

    else:
        obj_path = os.path.join(data_path, ply_file)
        mesh = load_mesh_util(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces

        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale
        mesh.vertices = vertices

        ### Make sure it is a triangle mesh -- just convert the quad
        mesh.faces = quad_to_triangle_mesh(faces)

        print("before preprocessing...")
        print(mesh.vertices.shape)
        print(mesh.faces.shape)
        print()

        ### Pre-process mesh
        if preprocess_mesh:
            # Create a PyMeshLab mesh directly from vertices and faces
            ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)

            # Create a MeshSet and add your mesh
            ms = pymeshlab.MeshSet()
            ms.add_mesh(ml_mesh, "from_trimesh")

            # Apply filters
            ms.apply_filter('meshing_remove_duplicate_faces')
            ms.apply_filter('meshing_remove_duplicate_vertices')
            percentageMerge = pymeshlab.PercentageValue(0.5)
            ms.apply_filter('meshing_merge_close_vertices', threshold=percentageMerge)
            ms.apply_filter('meshing_remove_unreferenced_vertices')

            # Save or extract mesh
            processed = ms.current_mesh()
            mesh.vertices = processed.vertex_matrix()
            mesh.faces = processed.face_matrix()               

            print("after preprocessing...")
            print(mesh.vertices.shape)
            print(mesh.faces.shape)

        ### Save input
        save_dir = f"exp_results/{result_name}"
        os.makedirs(save_dir, exist_ok=True)
        view_id = 0            
        mesh.export(f'{save_dir}/input_{uid}_{view_id}.ply')                


        pc, _ = trimesh.sample.sample_surface(mesh, pc_num_pts) 

    result = {
                'uid': uid
            }

    result['pc'] = torch.tensor(pc, dtype=torch.float32)

    if not is_pc:
        result['vertices'] = mesh.vertices
        result['faces'] = mesh.faces

    return result


def sample_points(vertices, faces, n_point_per_face):
    # Generate random barycentric coordinates
    # borrowed from Kaolin https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/mesh/trianglemesh.py#L43
    n_f = faces.shape[0]
    u = torch.sqrt(torch.rand((n_f, n_point_per_face, 1),
                                device=vertices.device,
                                dtype=vertices.dtype))
    v = torch.rand((n_f, n_point_per_face, 1),
                    device=vertices.device,
                    dtype=vertices.dtype)
    w0 = 1 - u
    w1 = u * (1 - v)
    w2 = u * v

    face_v_0 = torch.index_select(vertices, 0, faces[:, 0].reshape(-1))
    face_v_1 = torch.index_select(vertices, 0, faces[:, 1].reshape(-1))
    face_v_2 = torch.index_select(vertices, 0, faces[:, 2].reshape(-1))
    points = w0 * face_v_0.unsqueeze(dim=1) + w1 * face_v_1.unsqueeze(dim=1) + w2 * face_v_2.unsqueeze(dim=1)
    return points


def sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face, n_sample_each):
    n_v = tensor_vertices.shape[1]
    n_sample = n_v // n_sample_each + 1
    all_sample = []
    for i_sample in range(n_sample):
        sampled_feature = sample_triplane_feat(part_planes, tensor_vertices[:, i_sample * n_sample_each: i_sample * n_sample_each + n_sample_each,])
        assert sampled_feature.shape[1] % n_point_per_face == 0
        sampled_feature = sampled_feature.reshape(1, -1, n_point_per_face, sampled_feature.shape[-1])
        sampled_feature = torch.mean(sampled_feature, axis=-2)
        all_sample.append(sampled_feature)
    return torch.cat(all_sample, dim=1)


def predict(cfg):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_path = cfg.dataset.data_path
    is_pc = cfg.is_pc

    all_files = os.listdir(data_path)

    selected = []
    for f in all_files:
        if ".ply" in f and is_pc:
            selected.append(f)
        elif (".obj" in f or ".glb" in f or ".off" in f) and not is_pc:
            selected.append(f)

    file_list = selected
    pc_num_pts = 100000

    # Load model
    model_state_dict = torch.load(cfg.continue_ckpt)['state_dict']

    # Set models
    triplane_transformer = TriplaneTransformer(
        input_dim=cfg.triplane_channels_low * 2,
        transformer_dim=1024,
        transformer_layers=6,
        transformer_heads=8,
        triplane_low_res=32,
        triplane_high_res=128,
        triplane_dim=cfg.triplane_channels_high,
    ).to(device)
    triplane_transformer_dict = {
        k.replace('triplane_transformer.', ''): v for k, v in model_state_dict.items() if 'triplane_transformer' in k
    }
    triplane_transformer.load_state_dict(triplane_transformer_dict)
    for param in triplane_transformer.parameters():
        param.requires_grad = False

    pvcnn = TriPlanePC2Encoder(
        cfg.pvcnn,
        device=device,
        shape_min=-1, 
        shape_length=2,
        use_2d_feat=cfg.use_2d_feat
    ).to(device)
    pvcnn_transformer_dict = {
        k.replace('pvcnn.', ''): v for k, v in model_state_dict.items() if 'pvcnn' in k
    }
    pvcnn.load_state_dict(pvcnn_transformer_dict)
    for param in pvcnn.parameters():
        param.requires_grad = False

    num_rot = 10
    angles = np.stack([
        np.arange(num_rot + 1) / num_rot * np.pi * 2,
        np.zeros([num_rot + 1, ]),
        np.zeros([num_rot + 1, ]),
    ], axis=-1)
    test_rot = R.from_euler('zyx', angles, degrees=False).as_matrix().astype(np.float32)

    for f_idx, file in enumerate(file_list):
        rot_feat_list = []
        for rot_mtx in tqdm(test_rot, desc=f"Rotation Eval ({f_idx + 1} / {len(file_list)})"):
            data = get_model(file, is_pc, data_path, cfg.preprocess_mesh, cfg.result_name, pc_num_pts)

            if is_pc:
                data['pc'] = data['pc'] @ rot_mtx.T
                batch = {}
                batch['pc'] = data['pc'].unsqueeze(0).to(device)
                batch['uid'] = [data['uid']]
            else:
                data['pc'] = data['pc'] @ rot_mtx.T
                data['vertices'] = data['vertices'] @ rot_mtx.T
                batch = {}
                batch['pc'] = data['pc'].unsqueeze(0).to(device)
                batch['uid'] = [data['uid']]
                batch['vertices'] = [torch.from_numpy(data['vertices']).to(device)]
                batch['faces'] = [torch.from_numpy(data['faces']).to(device)]

            save_dir = f"exp_results/{cfg.result_name}"
            os.makedirs(save_dir, exist_ok=True)

            uid = batch['uid'][0]
            view_id = 0
            starttime = time.time()

            with torch.no_grad():
                pc_feat = pvcnn(batch['pc'], batch['pc'])
                planes = pc_feat
                planes = triplane_transformer(planes)
                _, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

            if is_pc:
                tensor_vertices = batch['pc'].reshape(1, -1, 3).cuda().to(torch.float32)
                point_feat = sample_triplane_feat(part_planes, tensor_vertices) # N, M, C
                point_feat = point_feat.cpu().detach().numpy().reshape(-1, 448)

                np.save(f'{save_dir}/part_feat_{uid}_{view_id}.npy', point_feat)
                print(f"Exported part_feat_{uid}_{view_id}.npy")

                ###########
                from sklearn.decomposition import PCA
                data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                pca = PCA(n_components=3)

                data_reduced = pca.fit_transform(data_scaled)
                data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
                colors_255 = (data_reduced * 255).astype(np.uint8)

                points = batch['pc'].squeeze().detach().cpu().numpy()

                if colors_255 is None:
                    colors_255 = np.full_like(points, 255)  # Default to white color (255,255,255)
                else:
                    assert colors_255.shape == points.shape, "Colors must have the same shape as points"
                
                # Convert to structured array for PLY format
                vertex_data = np.array(
                    [(*point, *color) for point, color in zip(points, colors_255)],
                    dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
                )

                # Create PLY element
                el = PlyElement.describe(vertex_data, "vertex")
                # Write to file
                filename = f'{save_dir}/feat_pca_{uid}_{view_id}.ply'
                PlyData([el], text=True).write(filename)
                print(f"Saved PLY file: {filename}")
                ############
            
            else:
                use_cuda_version = True
                if use_cuda_version:
                    if cfg.vertex_feature:
                        tensor_vertices = batch['vertices'][0].reshape(1, -1, 3).to(torch.float32)
                        point_feat = sample_and_mean_memory_save_version(part_planes, tensor_vertices, 1, cfg.n_sample_each)
                    else:
                        n_point_per_face = cfg.n_point_per_face
                        tensor_vertices = sample_points(batch['vertices'][0], batch['faces'][0], n_point_per_face)
                        tensor_vertices = tensor_vertices.reshape(1, -1, 3).to(torch.float32)
                        point_feat = sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face, cfg.n_sample_each)  # N, M, C

                    #### Take mean feature in the triangle
                    print("Time elapsed for feature prediction: " + str(time.time() - starttime))
                    point_feat = point_feat.reshape(-1, 448).cpu().numpy()
                    np.save(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy', point_feat)
                    print(f"Exported part_feat_{uid}_{view_id}.npy")

                    ###########
                    from sklearn.decomposition import PCA
                    data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                    pca = PCA(n_components=3)

                    data_reduced = pca.fit_transform(data_scaled)
                    data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
                    colors_255 = (data_reduced * 255).astype(np.uint8)
                    V = batch['vertices'][0].cpu().numpy()
                    F = batch['faces'][0].cpu().numpy()
                    if cfg.vertex_feature:
                        colored_mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=colors_255, process=False)
                    else:
                        colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)

                    # When saving, use open3d to write meshes visualizable in Cloudcompare
                    colored_mesh_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(V), triangles=o3d.utility.Vector3iVector(F))
                    colored_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colored_mesh.visual.vertex_colors[:, :-1] / 255.)
                    o3d.io.write_triangle_mesh(f'{save_dir}/feat_pca_{uid}_{view_id}.ply', colored_mesh_o3d)
                    ############
                    torch.cuda.empty_cache()

                else:
                    ### Mesh input (obj file)
                    V = batch['vertices'][0].cpu().numpy()
                    F = batch['faces'][0].cpu().numpy()

                    ##### Loop through faces #####
                    num_samples_per_face = cfg.n_point_per_face

                    all_point_feats = []
                    for face in F:
                        # Get the vertices of the current face
                        v0, v1, v2 = V[face]

                        # Generate random barycentric coordinates
                        u = np.random.rand(num_samples_per_face, 1)
                        v = np.random.rand(num_samples_per_face, 1)
                        is_prob = (u+v) >1
                        u[is_prob] = 1 - u[is_prob]
                        v[is_prob] = 1 - v[is_prob]
                        w = 1 - u - v
                        
                        # Calculate points in Cartesian coordinates
                        points = u * v0 + v * v1 + w * v2 

                        tensor_vertices = torch.from_numpy(points.copy()).reshape(1, -1, 3).cuda().to(torch.float32)
                        point_feat = sample_triplane_feat(part_planes, tensor_vertices) # N, M, C 

                        #### Take mean feature in the triangle
                        point_feat = torch.mean(point_feat, axis=1).cpu().detach().numpy()
                        all_point_feats.append(point_feat)                  
                    ##############################
                    
                    all_point_feats = np.array(all_point_feats).reshape(-1, 448)
                    
                    point_feat = all_point_feats

                    np.save(f'{save_dir}/part_feat_{uid}_{view_id}.npy', point_feat)
                    print(f"Exported part_feat_{uid}_{view_id}.npy")
                    
                    ###########
                    from sklearn.decomposition import PCA
                    data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                    pca = PCA(n_components=3)

                    data_reduced = pca.fit_transform(data_scaled)
                    data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
                    colors_255 = (data_reduced * 255).astype(np.uint8)

                    colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)

                    # When saving, use open3d to write meshes visualizable in Cloudcompare
                    colored_mesh_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(V), triangles=o3d.utility.Vector3iVector(F))
                    colored_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colored_mesh.visual.vertex_colors[:, :-1] / 255.)
                    o3d.io.write_triangle_mesh(f'{save_dir}/feat_pca_{uid}_{view_id}.ply', colored_mesh_o3d)
                    ############

            rot_feat_list.append(data_scaled)
        full_rot_feat = np.concatenate(rot_feat_list, axis=0)
        full_rot_feat = full_rot_feat.reshape(num_rot + 1, -1, full_rot_feat.shape[-1])

        per_point_feat_dist = np.linalg.norm(full_rot_feat[0:1] - full_rot_feat, axis=-1)
        feat_dist = per_point_feat_dist.mean(-1)
        plt.plot(feat_dist)
        plt.show()


def main():
    parser = default_argument_parser()
    args = parser.parse_args()
    cfg = setup(args, freeze=False)
    predict(cfg)
    
if __name__ == '__main__':
    main()