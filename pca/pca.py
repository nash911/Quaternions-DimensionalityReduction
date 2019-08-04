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
from scipy.stats import ortho_group


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

    def calc_info_and_dims(self, n_dims=None, keep_info=None, single_pca=False):
        total_dims = self.data.shape[1]

        # Normalize singular values
        normalized_singular_values = self.singular_values / sum(self.singular_values)

        # Create the aggregated ladder of kept information per dimension
        ladder = np.cumsum(normalized_singular_values)

        if keep_info:
            # Get the first index which is above the given information threshold
            index = next(idx for idx, value in enumerate(ladder) if value >= keep_info) + 1
            n_dims = index
        else:
            if single_pca:
                if n_dims < 0:
                    keep_info = normalized_singular_values[total_dims+n_dims]
                else:
                    keep_info = normalized_singular_values[n_dims-1]
            else:
                if n_dims < 0:
                    ladder = np.cumsum(np.flip(normalized_singular_values))
                keep_info = ladder[abs(n_dims)-1]

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


    def reproject(self, n_dims=None, keep_info=None, single_pca=False):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info, single_pca)
        total_dims = self.data.shape[1]

        with self.graph.as_default():
            if single_pca:
                if n_dims >= 0:
                    start_idx = n_dims - 1
                else:
                    start_idx = total_dims + n_dims

                # Cut out the relevant part from ∑, U and V
                sigma = tf.slice(self.sigma, [start_idx, start_idx],
                                 [1, 1])
                u = tf.slice(self.u, [0, start_idx], [self.data.shape[0], 1])
                v = tf.slice(self.v, [0, start_idx], [self.data.shape[1], 1])
            else:
                if n_dims >= 0:
                    start_idx = 0
                else:
                    start_idx = total_dims + n_dims
                # Cut out the relevant part from ∑, U and V
                sigma = tf.slice(self.sigma, [start_idx, start_idx],
                                 [abs(n_dims), abs(n_dims)])
                u = tf.slice(self.u, [0, start_idx], [self.data.shape[0], abs(n_dims)])
                v = tf.slice(self.v, [0, start_idx], [self.data.shape[1], abs(n_dims)])

            # Reproject on to linear subspace spanned by Principle Components
            reproj = tf.matmul(u, tf.matmul(sigma, v, transpose_b=True))

        with tf.Session(graph=self.graph) as session:
            return keep_info, n_dims, session.run(reproj)


    def basis(self, n_dims=None, keep_info=None, single_pca=False):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info, single_pca)

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

    def save_pca(self):
        if os.path.exists('U.dat'):
            os.remove('U.dat')
        np.savetxt('U.dat', self.u, delimiter=',')



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
                if k == 'root_rotation' and fixed_root_rot:
                    v = (np.zeros_like(v) + np.array([1, 0, 0, 0])).tolist()
                trajs_data.append(v)
        if not include:
            if not k in key_list:
                trajs_data.append(v)
    return np.column_stack(trajs_data)


def decompose_quat_trajectories(motion_data):
    # Decomposes trajectories into individual joints (by name)
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


def decompose_euler_trajectories(motion_data):
    # Decomposes trajectories into individual joints (by name)
    quat_trajs = OrderedDict()

    quat_trajs['frame_duration'] = np.array(motion_data[:,0:1]) # Time
    quat_trajs['root_position'] = np.array(motion_data[:,1:4])  # Position
    quat_trajs['root_rotation'] = np.array(motion_data[:,4:8])  # Quaternion

    quat_trajs['chest_rotation'] = np.array(motion_data[:,8:11]) # Quaternion
    quat_trajs['neck_rotation'] = np.array(motion_data[:,11:14]) # Quaternion

    quat_trajs['right_hip_rotation'] = np.array(motion_data[:,14:17]) # EulerAngle
    quat_trajs['right_knee_rotation'] = np.array(motion_data[:,17:18]) # 1D Joint
    quat_trajs['right_ankle_rotation'] = np.array(motion_data[:,18:21]) # EulerAngle
    quat_trajs['right_shoulder_rotation'] = np.array(motion_data[:,21:24]) # EulerAngle
    quat_trajs['right_elbow_rotation'] = np.array(motion_data[:,24:25]) # 1D Joint

    quat_trajs['left_hip_rotation'] = np.array(motion_data[:,25:28]) # EulerAngle
    quat_trajs['left_knee_rotation'] = np.array(motion_data[:,28:29]) # 1D Joint
    quat_trajs['left_ankle_rotation'] = np.array(motion_data[:,29:32]) # EulerAngle
    quat_trajs['left_shoulder_rotation'] = np.array(motion_data[:,32:35]) # EulerAngle
    quat_trajs['left_elbow_rotation'] = np.array(motion_data[:,35:36]) # 1D Joint

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

def convert_quat_to_euler(quat_dict, k_list):
    euler_dict = OrderedDict()

    for k, v in quat_dict.items():
        if v.shape[1] == 4 and k not in k_list:
            euler_angles = []
            for r in v:
                q = np.array([r[1], r[2], r[3], r[0]]) # [x, y, z, w]
                nq = Quat(normalize(q))
                nq_v = nq._get_q()
                w = nq_v[3]
                x = nq_v[0]
                y = nq_v[1]
                z = nq_v[2]

                # roll (x-axis rotation)
                t0 = +2.0 * (w * x + y * z)
                t1 = +1.0 - 2.0 * (x * x + y * y)
                roll = math.atan2(t0, t1)

                # pitch (y-axis rotation)
                t2 = +2.0 * (w * y - z * x)
                t2 = +1.0 if t2 > +1.0 else t2
                t2 = -1.0 if t2 < -1.0 else t2
                pitch = math.asin(t2)

                # yaw (z-axis rotation)
                t3 = +2.0 * (w * z + x * y)
                t4 = +1.0 - 2.0 * (y * y + z * z)
                yaw = math.atan2(t3, t4)

                euler_angles.append([roll, pitch, yaw])
            euler_dict[k] = np.array(euler_angles)
        else:
            euler_dict[k] = v

    return euler_dict


def convert_euler_to_quat(euler_dict, key_list):
    quat_dict = OrderedDict()

    for k, v in euler_dict.items():
        if v.shape[1] == 3 and k not in key_list:
            quats = []
            for r in v:
                roll = r[0]
                pitch = r[1]
                yaw = r[2]

                qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
                qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
                qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

                quats.append([qw, qx, qy, qz])
            quat_dict[k] = np.array(quats)
        else:
            quat_dict[k] = v

    return quat_dict


def check_orthogonality(c_vecs, n_dims, orth_tol=1e-06):
    num_vecs = c_vecs.shape[0]
    vec_list = [c_vecs[i, :] for i in range(num_vecs)]

    orthogonal = True

    mat_mul = np.matmul(c_vecs, c_vecs.T)
    if np.allclose(mat_mul, np.eye(abs(n_dims)), rtol=1e-05, atol=1e-05):
        print("WARNING: Basis vectors are Normal!")

    for i in range(num_vecs):
        for j in range(i+1, num_vecs):
            dot_prod = np.matmul(vec_list[i], vec_list[j].T)
            if dot_prod > orth_tol:
                print(dot_prod)
                orthogonal = False
                break

    return orthogonal


def pca_extract(tf_pca, pca_traj_dict, trajectory_dict, key_list, n_dims,
                keep_info=0.9, pca=True, reproj=True, basis=True, u_matrix=False,
                sigma_matrix=False, v_matrix=False, inverse=False, axisangle=False,
                eulerangle=False, fixed_root_pos=False, fixed_root_rot=False,
                single_pca=False, graph=False, orth_tol=1e-06):
    if pca:
        # Project the trajectories on to a reduced lower-dimensional space: U ∑
        info_retained, num_dims_retained, reduced = \
            tf_pca.reduce(keep_info=keep_info, n_dims=n_dims)
        pca_traj_dict['Reduced'] = reduced.tolist()
        if graph:
            plot(reduced)

    if reproj:
        # Reproject the trajectories on to a linear sub-space in the full space: U ∑ V^T
        info_retained, num_dims_retained, reproj_traj = \
            tf_pca.reproject(keep_info=keep_info, n_dims=n_dims,
                             single_pca=single_pca)

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
            mixed_dict = decompose_quat_trajectories(np.array(pca_traj_dict['Frames']))
            quat_dict = convert_to_quaternion(mixed_dict, key_list)
            concat_quat_trajs = concatenate_trajectories(quat_dict)
            pca_traj_dict['Frames'] = concat_quat_trajs.tolist()

        if eulerangle:
            # Convert back to Quaternions
            mixed_dict = decompose_euler_trajectories(np.array(pca_traj_dict['Frames']))
            quat_dict = convert_euler_to_quat(mixed_dict, key_list)
            concat_quat_trajs = concatenate_trajectories(quat_dict)
            pca_traj_dict['Frames'] = concat_quat_trajs.tolist()

    if basis:
        # Get full-rank basis vectors of the linear sub-space: ∑ V^T
        info_retained, num_dims_retained, basis_v = \
            tf_pca.basis(keep_info=keep_info, n_dims=n_dims, single_pca=single_pca)

        if not check_orthogonality(basis_v, n_dims, orth_tol=orth_tol):
            if os.path.exists('pca_traj.txt'):
                os.remove('pca_traj.txt')
            print("Error: Basis Vectors not Orthogonal!")
            sys.exit()

        pca_traj_dict['Basis'] = basis_v.tolist()

    if u_matrix:
        U = tf_pca.u[:, 0:n_dims]
        pca_traj_dict['U'] = U.tolist()

    if sigma_matrix:
        Sigma = tf.slice(tf_pca.sigma, [0, 0], [n_dims, n_dims])
        pca_traj_dict['Sigma'] = Sigma.tolist()

    if v_matrix:
        V = tf_pca.v[:, 0:n_dims]
        pca_traj_dict['V'] = V.tolist()

    if inverse:
        # Get pseudo-inverse of matrix V: (V^-1)^T
        v_pinv = pinv(tf_pca.v[:, 0:n_dims]).T
        pca_traj_dict['V_Inv'] = v_pinv.tolist()

        _, _, u_sigma = tf_pca.reduce(keep_info=keep_info, n_dims=n_dims)
        np.testing.assert_array_almost_equal(np.matmul(tf_pca.data, v_pinv),
                                             u_sigma, decimal=5)

        # if not check_orthogonality(v_pinv, n_dims=(28 if eulerangle else 36),
        #                            decimal=1e-06):
        #     print("Warning: V_Inv Vectors not Orthogonal!")

        # Get pseudo-inverse of matrix (∑ V^T): (∑ V^T)^-1
        _, _, sigma_v = \
            tf_pca.basis(keep_info=keep_info, n_dims=n_dims, single_pca=single_pca)
        sigma_v_pinv = pinv(sigma_v)
        pca_traj_dict['Basis_Inv'] = sigma_v_pinv.tolist()

        # u = tf_pca.u[:, 0:n_dims]
        # np.testing.assert_array_almost_equal(np.matmul(tf_pca.data, sigma_v_pinv),
        #                                      u, decimal=5)

        # if not check_orthogonality(sigma_v_pinv, n_dims=(28 if eulerangle else 36),
        #                            decimal=1e-06):
        #     print("Warning: Basis_Inv Vectors not Orthogonal!")

    return info_retained, num_dims_retained, pca_traj_dict


def usage():
    print("Usage: pca.py [-a | --axisangle] \n"
          "              [-b | --basis] \n"
          "              [-d | --dims] <no. of dims>/'all' \n"
          "              [-e | --eulerangle] \n"
          "              [-f | --fixed] \n"
          "              [-g | --graph] \n"
          "              [-h | --help] \n"
          "              [-i | --inv] \n"
          "              [-k | --keep] <% of info. to be retained> \n"
          "              [-m | --mfile] <input motion file> \n"
          "              [-n | --normalize] \n"
          "              [-p | --pca] \n"
          "              [-r | --reproj] \n"
          "              [-s | --single] \n"
          "              [-t | --tol] <orthogonal tolerance> \n"
          "              [-u | --U] \n"
          "              [-v | --V] \n"
          "              [-z | --Sigma] \n"
          )


def main(argv):
    motion_file = None
    pca = False
    reproj = False
    basis = False
    u_matrix = False
    sigma_matrix = False
    v_matrix = False
    inverse = False
    axisangle = False
    eulerangle = False
    normalise = False
    fixed_root_pos = False
    fixed_root_rot = False
    single_pca = False
    graph = False
    orth_tol = 1e-06

    keep_info = None
    n_dims = None

    info_retained = None
    num_dims_retained = None

    try:
        opts, args = getopt.getopt(argv,"hprbuzviaenfsgm:k:d:t:",
            ["help", "pca", "reproj", "basis", "U", "Sigma", "V", "inv", "axisangle",
            "eulerangle", "normalize", "fixed", "single", "graph", "mfile=",
             "keep=", "dims=", "tol="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
       if opt in ("-h", "--help"):
           usage()
           sys.exit()
       elif opt in ("-p", "--pca"):
           pca = True
       elif opt in ("-r", "--reproj"):
           reproj = True
       elif opt in ("-b", "--basis"):
           basis = True
       elif opt in ("-u", "--U"):
           u_matrix = True
       elif opt in ("-z", "--Sigma"):
           sigma_matrix = True
       elif opt in ("-v", "--V"):
           v_matrix = True
       elif opt in ("-i", "--inv"):
           inverse = True
       elif opt in ("-a", "--axisangle"):
           axisangle = True
           if os.path.exists('pca_traj.txt'):
               os.remove('pca_traj.txt')
           print("PCA in Axis-Angle currently not supposted!")
           sys.exit()
       elif opt in ("-e", "--eulerangle"):
           eulerangle = True
       elif opt in ("-n", "--normalize"):
           normalise = True
       elif opt in ("-m", "--mfile"):
           motion_file = arg
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
       elif opt in ("-s", "--single"):
           single_pca = True
       elif opt in ("-g", "--graph"):
           graph = True
       elif opt in ("-t", "--tol"):
           orth_tol = float(arg)

    if keep_info is None and n_dims is None:
        keep_info = 0.9

    with open(motion_file) as f:
        data = json.load(f)

    motion_data = np.array(data['Frames'])

    key_list = ['frame_duration', 'root_position', 'root_rotation']

    quat_trajectory_dict = decompose_quat_trajectories(motion_data)
    if normalise:
        norm_trajectory_dict = normalize_quaternions(quat_trajectory_dict)
    else:
        norm_trajectory_dict = quat_trajectory_dict

    if axisangle:
        axis_angle_traj_dict = convert_to_axis_angle(norm_trajectory_dict)

    if eulerangle:
        norm_trajectory_dict = convert_quat_to_euler(norm_trajectory_dict,
                                                          key_list)
    quat_traj_matrix = concatenate_trajectories(norm_trajectory_dict,
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

    # Save PCA on file
    #tf_pca.save_pca()

    # Create a clone of the input file dictionary
    pca_traj_dict = data.copy()

    # Set the domain of the coordination space (Basis-Vectors - ∑ V^T)
    if eulerangle:
        pca_traj_dict['Domain'] = "Eulerangle"
    else:
        pca_traj_dict['Domain'] = "Quaternion"

    key_list = ['frame_duration', 'root_position', 'root_rotation']

    info_retained, num_dims_retained, pca_traj_dict = \
        pca_extract(tf_pca, pca_traj_dict, norm_trajectory_dict,
                    key_list, n_dims=n_dims, keep_info=keep_info, pca=pca,
                    reproj=reproj, basis=basis, u_matrix=u_matrix,
                    sigma_matrix=sigma_matrix, v_matrix=v_matrix, inverse=inverse,
                    axisangle=axisangle, eulerangle=eulerangle,
                    fixed_root_pos=fixed_root_pos, fixed_root_rot=fixed_root_rot,
                    single_pca=single_pca, graph=graph, orth_tol=orth_tol)

    print("No. of dimensions: ", num_dims_retained)
    print("Keept info: ", info_retained)

    # Create output path and file
    output_file_path = "/home/nash/DeepMimic/data/reduced_motion/pca_"
    output_file = motion_file.split("/")[-1]
    output_file = output_file.split(".")[0]
    domain = "euler_" if eulerangle else "quat_"
    output_file = output_file_path + domain + output_file + "_" + \
                  str(info_retained) + "_" + str(num_dims_retained) + ".txt"

    # Save pca trajectories and basis dictionary on to the created output file
    with open(output_file, 'w') as fp:
        json.dump(pca_traj_dict, fp, indent=4)

    with open('pca_traj.txt', 'w') as fp:
        json.dump(pca_traj_dict, fp, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])
