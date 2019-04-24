import numpy as np
from numpy.linalg import pinv
from Quaternion import Quat, normalize
import tensorflow as tf
import math
from collections import OrderedDict
import sys, getopt
import json
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns


class TF_PCA:
    def __init__(self, data, target=None, dtype=tf.float32):
        self.data = data
        self.target = target
        self.dtype = dtype

        self.graph = None
        self.X = None
        self.u = None
        self.v = None
        self.singular_values = None
        self.sigma = None

    def fit(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(self.dtype, shape=self.data.shape)

            # Perform SVD
            singular_values, u, v = tf.svd(self.X)

            # Create ∑ matrix
            sigma = tf.diag(singular_values)

        with tf.Session(graph=self.graph) as session:
            self.u, self.singular_values, self.sigma, self.v = session.run(
                [u, singular_values, sigma, v], feed_dict={self.X: self.data})

    def calc_info_and_dims(self, n_dims=None, keep_info=None):
        # Normalize singular values
        normalized_singular_values = self.singular_values / sum(self.singular_values)

        # Create the aggregated ladder of kept information per dimension
        ladder = np.cumsum(normalized_singular_values)

        if keep_info:
            # Get the first index which is above the given information threshold
            index = next(idx for idx, value in enumerate(ladder) if value >= keep_info) + 1
            n_dims = index
        else:
            if n_dims < 0:
                ladder = np.cumsum(np.flip(normalized_singular_values))
            keep_info = ladder[abs(n_dims) - 1]

        return n_dims, keep_info


    def reduce(self, n_dims=None, keep_info=None):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info)
        total_dims = self.data.shape[1]

        if n_dims >= 0:
            start_idx = 0
        else:
            start_idx = total_dims + n_dims

        with self.graph.as_default():
            # Cut out the relevant part from ∑ and U
            sigma = tf.slice(self.sigma, [start_idx, start_idx],
                             [abs(n_dims), abs(n_dims)])
            u = tf.slice(self.u, [0, start_idx], [self.data.shape[0], abs(n_dims)])

            # PCA
            pca = tf.matmul(u, sigma)

        with tf.Session(graph=self.graph) as session:
            return keep_info, n_dims, session.run(pca,
                                                  feed_dict={self.X: self.data})


    def reproject(self, n_dims=None, keep_info=None):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info)
        total_dims = self.data.shape[1]

        if n_dims >= 0:
            start_idx = 0
        else:
            start_idx = total_dims + n_dims

        with self.graph.as_default():
            # Cut out the relevant part from ∑, U and V
            sigma = tf.slice(self.sigma, [start_idx, start_idx],
                             [abs(n_dims), abs(n_dims)])
            u = tf.slice(self.u, [0, start_idx], [self.data.shape[0], abs(n_dims)])
            v = tf.slice(self.v, [0, start_idx], [self.data.shape[1], abs(n_dims)])

            # Reproject on to linear subspace spanned by Principle Components
            reproj = tf.matmul(u, tf.matmul(sigma, v, transpose_b=True))

        with tf.Session(graph=self.graph) as session:
            return keep_info, n_dims, session.run(reproj)


    def basis(self, n_dims=None, keep_info=None):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info)

        if n_dims >= 0:
            b = np.matmul(self.sigma[0:n_dims, 0:n_dims], self.v[:, 0:n_dims].T)
        else:
            b = np.matmul(self.sigma[n_dims:, n_dims:], self.v[:, n_dims:].T)

        return keep_info, n_dims, b

    def pc_mean_std(self, n_dims=None, keep_info=None):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info)

        with self.graph.as_default():
            # Cut out the relevant part from U
            u = tf.slice(self.u, [0, 0], [self.data.shape[0], n_dims])
            # μ
            mu = tf.reduce_mean(u, axis=0, keep_dims=True)
            # (x - μ)^2
            devs_squared = tf.square(u - mu)
            # √∑(x - μ)^2
            std = tf.sqrt(tf.reduce_mean(devs_squared, axis=0, keep_dims=False))
            mean = tf.squeeze(mu)

        with tf.Session(graph=self.graph) as session:
            return session.run([mean, std], feed_dict={self.X: self.data})


def plot(data):
    # 3-D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o')

    ax.set_xlabel('PC-1')
    ax.set_ylabel('PC-2')
    ax.set_zlabel('PC-3')

    plt.show()


def concatenate_trajectories(trajs_dict, key_list = [], include=False,
                             fixed_root_pos=False, fixed_root_rot=False):
    trajs_data = []
    for k, v in trajs_dict.items():
        if include:
            if k in key_list:
                if k == 'root_position' and fixed_root_pos:
                    v = (np.ones_like(v) * v[0]).tolist()
                if k == 'root_rotation' and fixed_root_pos:
                    v = (np.ones_like(v) * v[0]).tolist()
                trajs_data.append(v)
        if not include:
            if not k in key_list:
                trajs_data.append(v)

    return np.column_stack(trajs_data)


def decompose_trajectories(motion_data):
    # Decomposes trajectories into indificual DOFs by joint name
    quat_trajs = OrderedDict()

    quat_trajs['frame_duration'] = np.array(motion_data[:,0:1]) # Time
    quat_trajs['root_position'] = np.array(motion_data[:,1:4])  # Position
    quat_trajs['root_rotation'] = np.array(motion_data[:,4:8])  # Quaternion

    quat_trajs['chest_rotation'] = np.array(motion_data[:,8:12]) # Quaternion
    quat_trajs['neck_rotation'] = np.array(motion_data[:,12:16]) # Quaternion

    quat_trajs['right_hip_rotation'] = np.array(motion_data[:,16:20]) # Quaternion
    quat_trajs['right_knee_rotation'] = np.array(motion_data[:,20:21]) # 1D Joint
    quat_trajs['right_ankle_rotation'] = np.array(motion_data[:,21:25]) # Quaternion
    quat_trajs['right_shoulder_rotation'] = np.array(motion_data[:,25:29]) # Quaternion
    quat_trajs['right_elbow_rotation'] = np.array(motion_data[:,29:30]) # 1D Joint

    quat_trajs['left_hip_rotation'] = np.array(motion_data[:,30:34]) # Quaternion
    quat_trajs['left_knee_rotation'] = np.array(motion_data[:,34:35]) # 1D Joint
    quat_trajs['left_ankle_rotation'] = np.array(motion_data[:,35:39]) # Quaternion
    quat_trajs['left_shoulder_rotation'] = np.array(motion_data[:,39:43]) # Quaternion
    quat_trajs['left_elbow_rotation'] = np.array(motion_data[:,43:44]) # 1D Joint

    return quat_trajs


def normalize_quaternions(quat_dict):
    norm_quat_dict = OrderedDict()

    for k, v in quat_dict.items():
        if v.shape[1] == 4:
            norm_quats = []
            for r in v:
                q = np.array([r[1], r[2], r[3], r[0]]) # [x, y, z, w]
                nq = Quat(normalize(q))
                nq_v = nq._get_q()
                norm_quats.append([nq_v[3], nq_v[0], nq_v[1], nq_v[2]]) # [w, x, y, z]
            norm_quat_dict[k] = np.array(norm_quats)
        else:
            norm_quat_dict[k] = v

    return norm_quat_dict


def convert_to_axis_angle(quaternion_dict):
    axis_angle_dict = OrderedDict()

    for k, v in quaternion_dict.items():
        if v.shape[1] == 4:
            axis_angle = []
            for r in v:
                q = np.array([r[1], r[2], r[3], r[0]]) # [x, y, z, w]
                quat = Quat(q)
                a = quat._get_angle_axis()
                axis_angle.append(np.array([a[0], a[1][0], a[1][1], a[1][2]])) # [Ө, x, y, z]
            axis_angle_dict[k] = np.array(axis_angle)
        else:
            axis_angle_dict[k] = v

    return axis_angle_dict


def convert_to_quaternion(axis_angle_dict, k_list):
    quat_dict = OrderedDict()

    for k, v in axis_angle_dict.items():
        if v.shape[1] == 4 and k not in k_list:
            quaternions = []
            for r in v:
                x = r[1] * math.sin(r[0]/2.0)
                y = r[2] * math.sin(r[0]/2.0)
                z = r[3] * math.sin(r[0]/2.0)
                w = math.cos(r[0]/2.0)

                q = np.array([w, x, y, z])
                quaternions.append(q)
            quat_dict[k] = np.array(quaternions)
        else:
            quat_dict[k] = v

    return quat_dict


def check_orthogonality(c_vecs, n_dims, decimal=1e-6):
    num_vecs = c_vecs.shape[0]
    vec_list = [c_vecs[i, :] for i in range(num_vecs)]

    orthogonal = True

    mat_mul = np.matmul(c_vecs, c_vecs.T)
    if np.allclose(mat_mul, np.eye(abs(n_dims)), rtol=1e-05, atol=1e-05):
        print("WARNING: Basis vectors are Normal!")

    for i in range(num_vecs):
        for j in range(i+1, num_vecs):
            dot_prod = np.matmul(vec_list[i], vec_list[j].T)
            if dot_prod > decimal:
                print(dot_prod)
                orthogonal = False
                break

    return orthogonal


def pca_extract(tf_pca, pca_traj_dict, trajectory_dict, key_list, n_dims,
                keep_info=0.9, pca=True, reproj=True, basis=True, vinv=False,
                axisangle=False, fixed_root_pos=False, fixed_root_rot=False):
    if pca:
        # Project the trajectories on to a reduced lower-dimensional space: U ∑
        info_retained, num_dims_retained, reduced = \
            tf_pca.reduce(keep_info=keep_info, n_dims=n_dims)
        pca_traj_dict['Reduced'] = reduced.tolist()
        #plot(reduced)

    if reproj:
        # Reproject the trajectories on to a linear sub-space in the full space: U ∑ V^T
        info_retained, num_dims_retained, reproj_traj = \
            tf_pca.reproject(keep_info=keep_info, n_dims=n_dims)

        # Replace original trajectories, in the controlable DOFs, with
        # reprojected trajectories
        unchanged_traj = concatenate_trajectories(trajectory_dict, key_list,
                                                  include=True,
                                                  fixed_root_pos=fixed_root_pos,
                                                  fixed_root_rot=fixed_root_rot)
        # Remove non-controlable DOFs from reprojected trajectories, if exists
        reproj_traj = reproj_traj[:, -36:reproj_traj.shape[1]]
        pca_traj_dict['Frames'] = np.column_stack((unchanged_traj,
                                              reproj_traj)).tolist()

        if axisangle:
            # Convert back to Quaternions
            mixed_dict = decompose_trajectories(np.array(pca_traj_dict['Frames']))
            quat_dict = convert_to_quaternion(mixed_dict, key_list)
            concat_quat_trajs = concatenate_trajectories(quat_dict)
            pca_traj_dict['Frames'] = concat_quat_trajs.tolist()

    if basis:
        # Get full-rank basis vectors of the linear sub-space: ∑ V^T
        info_retained, num_dims_retained, basis_v = \
            tf_pca.basis(keep_info=keep_info, n_dims=n_dims)

        if not check_orthogonality(basis_v, n_dims, decimal=1e-6):
            if os.path.exists('pca_traj.txt'):
                os.remove('pca_traj.txt')
            print("Error: Basis Vectors not Orthogonal!")
            sys.exit()

        pca_traj_dict['Basis'] = basis_v.tolist()

    if vinv:
        # Get pseudo-inverse of matrix V: (V^-1)^T
        v_pinv = pinv(tf_pca.v[:, 0:n_dims]).T
        pca_traj_dict['V_Inv'] = v_pinv.tolist()

        _, _, u_sigma = tf_pca.reduce(keep_info=keep_info, n_dims=n_dims)
        np.testing.assert_array_almost_equal(np.matmul(tf_pca.data, v_pinv),
                                             u_sigma, decimal=5)

        if not check_orthogonality(v_pinv, n_dims=36, decimal=1e-6):
            print("Warning: V_Inv Vectors not Orthogonal!")

    return info_retained, num_dims_retained, pca_traj_dict


def main(argv):
    input_file = 'humanoid3d_run.txt'

    pca = False
    reproj = False
    basis = False
    vinv = False
    axisangle = False
    normalise = False
    fixed_root_pos = False
    fixed_root_rot = False

    keep_info = None
    n_dims = None

    info_retained = None
    num_dims_retained = None

    try:
        opts, args = getopt.getopt(argv,"hprbvanfi:k:d:",
            ["pca", "reproj", "basis", "vinv", "axisangle", "normalize", "fixed", "ifile=","keep=", "dims="])
    except getopt.GetoptError:
        print("pca.py -i <inputfile> -k <keep_info> -d <num_dims>/'all' -p -r -b -v -a -n -f")
        sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
           print("pca.py -i <inputfile> -k <keep_info> -d <num_dims>/'all' -p -r -b -v -a -n -f")
           sys.exit()
       elif opt in ("-p", "--pca"):
           pca = True
       elif opt in ("-r", "--reproj"):
           reproj = True
       elif opt in ("-b", "--basis"):
           basis = True
       elif opt in ("-v", "--vinv"):
           vinv = True
       elif opt in ("-a", "--axisangle"):
           axisangle = True
           if os.path.exists('pca_traj.txt'):
               os.remove('pca_traj.txt')
           print("PCA in Axis-Angle currently not supposted!")
           sys.exit()
       elif opt in ("-n", "--normalize"):
           normalise = True
       elif opt in ("-i", "--ifile"):
           input_file = arg
       elif opt in ("-k", "--keep"):
           keep_info = float(arg)
       elif opt in ("-d", "--dims"):
           if arg.lower() == 'all':
               n_dims = 36
           else:
               n_dims = int(arg)
       elif opt in ("-f", "--fixed"):
           fixed_root_pos = True
           fixed_root_rot = True

    if keep_info is None and n_dims is None:
        keep_info = 0.9

    with open(input_file) as f:
        data = json.load(f)

    motion_data = np.array(data['Frames'])

    quat_trajectory_dict = decompose_trajectories(motion_data)
    if normalise:
        norm_quat_trajectory_dict = normalize_quaternions(quat_trajectory_dict)
    else:
        norm_quat_trajectory_dict = quat_trajectory_dict

    if axisangle:
        axis_angle_traj_dict = convert_to_axis_angle(norm_quat_trajectory_dict)

    key_list = ['frame_duration', 'root_position', 'root_rotation']
    quat_traj_matrix = concatenate_trajectories(norm_quat_trajectory_dict,
                                                key_list, include=False)
    if axisangle:
        axisangle_traj_matrix = concatenate_trajectories(axis_angle_traj_dict,
                                                         key_list, include=False)

    # Create a TF PCA object
    if axisangle:
        tf_pca = TF_PCA(axisangle_traj_matrix)
    else:
        tf_pca = TF_PCA(quat_traj_matrix)

    # Compute U, ∑ and V
    tf_pca.fit()

    # Create a clone of the input file dictionary
    pca_traj_dict = data.copy()

    key_list = ['frame_duration', 'root_position', 'root_rotation']

    info_retained, num_dims_retained, pca_traj_dict = \
        pca_extract(tf_pca, pca_traj_dict, norm_quat_trajectory_dict,
                    key_list, n_dims=n_dims, keep_info=keep_info, pca=pca,
                    reproj=reproj, basis=basis, vinv=vinv, axisangle=axisangle,
                    fixed_root_pos=fixed_root_pos, fixed_root_rot=fixed_root_rot)

    print("No. of dimensions: ", num_dims_retained)
    print("Keept info: ", info_retained)

    # Create output path and file
    output_file_path = "/home/nash/DeepMimic/data/reduced_motion/pca_"
    output_file = input_file.split("/")[-1]
    output_file = output_file.split(".")[0]
    output_file = output_file_path + output_file + "_" + str(info_retained) + \
                  "_" + str(num_dims_retained) + ".txt"

    # Save pca trajectories and basis dictionary on to the created output file
    with open(output_file, 'w') as fp:
        json.dump(pca_traj_dict, fp, indent=4)

    with open('pca_traj.txt', 'w') as fp:
        json.dump(pca_traj_dict, fp, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])
