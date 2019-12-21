import numpy as np
from numpy.linalg import pinv
from Quaternion import Quat, normalize
import tensorflow as tf
import math
from collections import OrderedDict
import sys, getopt
import json
import os
import glob
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
        self.X_mean = None
        self.u = None
        self.v = None
        self.singular_values = None
        self.sigma = None

    def fit(self, normalise=False):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(self.dtype, shape=self.data.shape)

            if normalise:
                # Calculate input mean
                X_mean = tf.reduce_mean(self.X, axis=0)
            else:
                X_mean = tf.zeros(shape=self.data.shape[1], dtype=self.dtype)

            # Perform SVD
            singular_values, u, v = tf.svd(self.X - X_mean)

            # Create ∑ matrix
            sigma = tf.diag(singular_values)

        with tf.Session(graph=self.graph) as session:
            self.u, self.singular_values, self.sigma, self.v, self.X_mean = \
                session.run([u, singular_values, sigma, v, X_mean],
                            feed_dict={self.X: self.data})

    def calc_info_and_dims(self, n_dims=None, keep_info=None, single_pca=False):
        total_dims = self.data.shape[1]

        # Normalize singular values
        normalised_singular_values = self.singular_values / sum(self.singular_values)

        # Create the aggregated ladder of kept information per dimension
        ladder = np.cumsum(normalised_singular_values)

        if keep_info:
            # Get the first index which is above the given information threshold
            index = next(idx for idx, value in enumerate(ladder) if value >= keep_info) + 1
            n_dims = index
        else:
            if single_pca:
                if n_dims < 0:
                    keep_info = normalised_singular_values[total_dims+n_dims]
                else:
                    keep_info = normalised_singular_values[n_dims-1]
            else:
                if n_dims < 0:
                    ladder = np.cumsum(np.flip(normalised_singular_values))
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
            return keep_info, n_dims, session.run(pca)


    def reproject(self, n_dims=None, keep_info=None, sine=False, single_pca=False,
                  frame_dur=0.03, sine_amp=[1.0], sine_freq=[1.0], sine_period=1.0,
                  sine_offset=[0], normal_basis=False):
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
                if sine:
                    u = self.sine_fn(frame_dur=frame_dur, amp=sine_amp, freq=sine_freq,
                                     period=sine_period, offset=sine_offset)
                else:
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
                if sine:
                    u = self.sine_fn(frame_dur=frame_dur, amp=sine_amp, freq=sine_freq,
                                     period=sine_period, offset=sine_offset)
                else:
                    u = tf.slice(self.u, [0, start_idx], [self.data.shape[0], abs(n_dims)])
                v = tf.slice(self.v, [0, start_idx], [self.data.shape[1], abs(n_dims)])

            # Reproject on to linear subspace spanned by Principle Components
            if normal_basis:
                reproj = tf.matmul(u, v, transpose_b=True) + self.X_mean
            else:
                reproj = tf.matmul(u, tf.matmul(sigma, v, transpose_b=True)) + self.X_mean

        with tf.Session(graph=self.graph) as session:
            return keep_info, n_dims, session.run(reproj)


    def basis(self, n_dims=None, keep_info=None, single_pca=False,
              normal_basis=False):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info, single_pca)

        if n_dims >= 0:
            if normal_basis:
                b = self.v[:, 0:n_dims].T
            else:
                b = np.matmul(self.sigma[0:n_dims, 0:n_dims], self.v[:, 0:n_dims].T)
        else:
            if normal_basis:
                b = self.v[:, n_dims:].T
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

    def sine_fn(self, frame_dur=0.03, period=1.0, amp=[1.0], freq=[1.0], offset=[0]):
        t = np.expand_dims(np.arange(0, period, frame_dur), axis=-1)
        angular_freq = 2.0 * np.pi * (np.array(freq)/period)
        sine_wave = (np.array(amp) * np.sin(angular_freq*t)) + np.array(offset)
        return np.array(sine_wave, dtype=np.float32)


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
                    #v = (np.ones_like(v) * v[0]).tolist()
                    v = (np.ones_like(v) * 1.0).tolist()
                if k == 'root_rotation' and fixed_root_rot:
                    v = (np.zeros_like(v) + np.array([1, 0, 0, 0])).tolist()
                trajs_data.append(v)
        if not include:
            if not k in key_list:
                trajs_data.append(v)
    return np.column_stack(trajs_data)


def decompose_quat_trajectories(motion_data, motion_control):
    # Decomposes trajectories into individual joints (by name)
    quat_trajs = OrderedDict()
    '''
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
    '''
    quat_trajs['frame_duration']=np.array(motion_data[:, 0:1])
    quat_trajs['root_position'] = np.array(motion_data[:,1:4])  # Position
    quat_trajs['root_rotation'] = np.array(motion_data[:,4:8])  # Quaternion
    
    for key, val in motion_control['pose'].items():
        if key != 'root':
            quat_trajs[key] = np.array(motion_data[:, val[0]:val[1] ])


    return quat_trajs


def decompose_euler_trajectories(motion_data, motion_control):
    # Decomposes trajectories into individual joints (by name)
    euler_trajs = OrderedDict()

    euler_trajs['frame_duration'] = np.array(motion_data[:,0:1]) # Time
    euler_trajs['root_position'] = np.array(motion_data[:,1:4])  # Position
    euler_trajs['root_rotation'] = np.array(motion_data[:,4:8])  # Quaternion
    '''
    euler_trajs['chest_rotation'] = np.array(motion_data[:,8:11]) # EulerAngle
    euler_trajs['neck_rotation'] = np.array(motion_data[:,11:14]) # EulerAngle

    euler_trajs['right_hip_rotation'] = np.array(motion_data[:,14:17]) # EulerAngle
    euler_trajs['right_knee_rotation'] = np.array(motion_data[:,17:18]) # 1D Joint
    euler_trajs['right_ankle_rotation'] = np.array(motion_data[:,18:21]) # EulerAngle
    euler_trajs['right_shoulder_rotation'] = np.array(motion_data[:,21:24]) # EulerAngle
    euler_trajs['right_elbow_rotation'] = np.array(motion_data[:,24:25]) # 1D Joint

    euler_trajs['left_hip_rotation'] = np.array(motion_data[:,25:28]) # EulerAngle
    euler_trajs['left_knee_rotation'] = np.array(motion_data[:,28:29]) # 1D Joint
    euler_trajs['left_ankle_rotation'] = np.array(motion_data[:,29:32]) # EulerAngle
    euler_trajs['left_shoulder_rotation'] = np.array(motion_data[:,32:35]) # EulerAngle
    euler_trajs['left_elbow_rotation'] = np.array(motion_data[:,35:36]) # 1D Joint
    '''
    index = 8
    for key, val in motion_control['pose'].items():
        if key != 'root':
            if (val[1] - val[0] == 4 ):
                euler_trajs[key] = np.array(motion_data[:, index:index+3])
                index = index + 3
            else:
                euler_trajs[key] = np.array(motion_data[:, index:index+1])
                index = index + 1

    return euler_trajs

def normalise_quaternions(quat_dict):
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


def check_orthogonality(c_vecs, n_dims, orth_tol=1e-06, normal_basis=False):
    num_vecs = c_vecs.shape[0]
    vec_list = [c_vecs[i, :] for i in range(num_vecs)]
    orthogonal = True

    mat_mul = np.matmul(c_vecs, c_vecs.T)
    if normal_basis:
        if not np.allclose(mat_mul, np.eye(abs(n_dims)), rtol=1e-05, atol=1e-05):
            return False
    else:
        for i in range(num_vecs):
            for j in range(i+1, num_vecs):
                dot_prod = np.matmul(vec_list[i], vec_list[j].T)
                if dot_prod > orth_tol:
                    print(dot_prod)
                    orthogonal = False
                    break

    return orthogonal

def convert_to_json(pose_file):
    with open(pose_file, 'r') as pf:
        pose_arr = np.loadtxt(pf)

    motion_dict = OrderedDict()
    motion_dict['Loop'] = 'wrap'
    motion_dict['Frames'] = pose_arr.tolist()

    with open(pose_file, 'w') as jf:
        json.dump(motion_dict, jf, indent=4)

    return motion_dict

def pca_extract(tf_pca, pca_traj_dict, trajectory_dict, key_list, n_dims, motion_control,
                keep_info=0.9, pca=True, reproj=True, basis=True, u_matrix=False,
                sigma_matrix=False, v_matrix=False, inverse=False, eulerangle=False,
                fixed_root_pos=False, fixed_root_rot=False, normalise=False,
                single_pca=False, graph=False, activ_stat=False, orth_tol=1e-06,
                sine=False, frame_dur=0.0333, sine_amp=[1.0], sine_freq=[1.0],
                sine_period=1.0, sine_offset=[0], normal_basis=False):
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
            tf_pca.reproject(keep_info=keep_info, n_dims=n_dims, sine=sine,
                             single_pca=single_pca,frame_dur=frame_dur,
                             normal_basis=normal_basis, sine_amp=sine_amp,
                             sine_freq=sine_freq, sine_period=sine_period,
                             sine_offset=sine_offset)

        # Replace original trajectories, in the controlable DOFs, with
        # reprojected trajectories
        unchanged_traj = concatenate_trajectories(trajectory_dict, key_list,
                                                  include=True,
                                                  fixed_root_pos=fixed_root_pos,
                                                  fixed_root_rot=fixed_root_rot)

        m = reproj_traj.shape[0]
        if m != unchanged_traj.shape[0]:
            if fixed_root_pos and fixed_root_rot:
                fixed_dur_root = np.array([frame_dur] + [1, 1, 1] + [1, 0, 0, 0])
                unchanged_traj = np.ones((m, 8), dtype=reproj_traj.dtype) * fixed_dur_root
            else:
                print("Error: Number of frames in the reprojected trajectories: %d != %d"
                      "(number of frames in input file)" % (m, unchanged_traj.shape[0]))
                print("       Use flag: [-f | --fixed], to fix root-position and rotation\n")
                sys.exit()
        # Remove non-controlable DOFs from reprojected trajectories, if exists
        reproj_traj = reproj_traj[:, (-28 if eulerangle else -36):reproj_traj.shape[1]]
        pca_traj_dict['Frames'] = np.column_stack((unchanged_traj,
                                              reproj_traj)).tolist()

        # if axisangle:
        #     # Convert back to Quaternions
        #     mixed_dict = decompose_quat_trajectories(np.array(pca_traj_dict['Frames']))
        #     quat_dict = convert_to_quaternion(mixed_dict, key_list)
        #     concat_quat_trajs = concatenate_trajectories(quat_dict)
        #     pca_traj_dict['Frames'] = concat_quat_trajs.tolist()

        if eulerangle:
            # Convert back to Quaternions
            mixed_dict = decompose_euler_trajectories(np.array(pca_traj_dict['Frames']), motion_control)
            quat_dict = convert_euler_to_quat(mixed_dict, key_list)
            concat_quat_trajs = concatenate_trajectories(quat_dict)
            pca_traj_dict['Frames'] = concat_quat_trajs.tolist()

    if basis:
        # Get full-rank basis vectors of the linear sub-space: ∑ V^T
        info_retained, num_dims_retained, basis_v = \
            tf_pca.basis(keep_info=keep_info, n_dims=n_dims, single_pca=single_pca,
                         normal_basis=normal_basis)

        if not check_orthogonality(basis_v, n_dims, orth_tol=orth_tol,
                                   normal_basis=normal_basis):
            if os.path.exists('Output/pca_traj.txt'):
                os.remove('Output/pca_traj.txt')

            if normal_basis:
                print("Error: Basis Vectors not Orthonormal!")
            else:
                print("Error: Basis Vectors not Orthogonal!")
            sys.exit()

        pca_traj_dict['Basis'] = basis_v.tolist()

    if u_matrix:
        U = tf_pca.u[:, 0:n_dims]
        pca_traj_dict['U'] = U.tolist()

    if sigma_matrix:
        Sigma = tf_pca.sigma[:n_dims, :n_dims]
        Singular_values = tf_pca.singular_values[:n_dims]
        pca_traj_dict['Sigma'] = Sigma.tolist()
        pca_traj_dict['Singular_Values'] = Singular_values.tolist()

    if v_matrix:
        V = tf_pca.v[:, 0:n_dims]
        pca_traj_dict['V'] = V.tolist()

    if inverse:
        # Get pseudo-inverse of matrix V: (V^-1)^T
        v_pinv = pinv(tf_pca.v[:, 0:n_dims]).T
        pca_traj_dict['V_Inv'] = v_pinv.tolist()

        # Check if the pseudo-inverse is consistant
        _, _, u_sigma = tf_pca.reduce(keep_info=keep_info, n_dims=n_dims)
        np.testing.assert_array_almost_equal(
            np.matmul((tf_pca.data - tf_pca.X_mean), v_pinv), u_sigma, decimal=4)

        # if not check_orthogonality(v_pinv, n_dims=(28 if eulerangle else 36),
        #                            decimal=1e-06):
        #     print("Warning: V_Inv Vectors not Orthogonal!")

        # Get pseudo-inverse of the Basis matrix
        _, _, basis_mat = \
            tf_pca.basis(keep_info=keep_info, n_dims=n_dims, single_pca=single_pca,
                         normal_basis=normal_basis)
        basis_mat_pinv = pinv(basis_mat)
        pca_traj_dict['Basis_Inv'] = basis_mat_pinv.tolist()

        # NOTE: The below check does not work when k ≈ N
        # # Check if the pseudo-inverse is consistant
        # if normal_basis:
        #     _, _, RHS = tf_pca.reduce(keep_info=keep_info, n_dims=n_dims) # U ∑
        # else:
        #     RHS = tf_pca.u[:, 0:n_dims] # U
        #
        # np.testing.assert_array_almost_equal(
        #     np.matmul((tf_pca.data - tf_pca.X_mean), basis_mat_pinv), RHS, decimal=4)

    if normalise and eulerangle:
        X_mean = tf_pca.X_mean
        pca_traj_dict['Reference_Mean'] = X_mean.tolist()

    if activ_stat:
        pca_traj_dict['Excite_Min'] = np.min(tf_pca.u[:, 0:n_dims], axis=0).tolist()
        pca_traj_dict['Excite_Max'] = np.max(tf_pca.u[:, 0:n_dims], axis=0).tolist()

    return info_retained, num_dims_retained, pca_traj_dict


def usage():
    print("Usage: pca.py [-a | --activ_stat] \n"
          "              [-A | --sine_amp] <list of sine-excitation amplitudes> \n"
          "              [-b | --basis] \n"
          "              [-d | --dims] <no. of dims>/'all' \n"
          "              [-D | --frame_duration] <frame duration in seconds>/'all' \n"
          "              [-e | --eulerangle] \n"
          "              [-f | --fixed] \n"
          "              [-F | --sine_freq] <list of sine-excitation frequencies> \n"
          "              [-g | --graph] \n"
          "              [-h | --help] \n"
          "              [-i | --inv] \n"
          "              [-k | --keep] <% of info. to be retained> \n"
          "              [-m | --mfile] <input motion file(s) or directory> \n"
          "              [-n | --normalise] \n"
          "              [-N | --normal_basis] \n"
          "              [-O | --sine_offset] <list of sine-excitation offsets> \n"
          "              [-p | --pca] \n"
          "              [-P | --sine_period] <sine-excitation period (in seconds)> \n"
          "              [-r | --reproj] \n"
          "              [-s | --single] \n"
          "              [-S | --sine] \n"
          "              [-t | --tol] <orthogonal tolerance> \n"
          "              [-u | --U] \n"
          "              [-v | --V] \n"
          "              [-z | --Sigma] \n"
          )


def main(argv):
    motion_files = list()
    control_file = None
    all_motions = False
    pca = False
    reproj = False
    basis = False
    u_matrix = False
    sigma_matrix = False
    v_matrix = False
    inverse = False
    activ_stat = False
    eulerangle = False
    normalise = False
    fixed_root_pos = False
    fixed_root_rot = False
    single_pca = False
    graph = False
    orth_tol = 1e-06
    sine = False
    sine_amp = [1.0]
    sine_freq = [1.0]
    sine_period = None
    sine_offset = [0]
    frame_dur = None
    normal_basis = False
    character = "humanoid"
    keep_info = None
    n_dims = None

    info_retained = None
    num_dims_retained = None

    duration_warning = False
    sine_period_warning = False

    try:
        opts, args = getopt.getopt(argv,"haprbuzviaenfsgSNm:k:d:t:A:F:P:O:D:c:C:",
            ["help", "activ_stat", "pca", "reproj", "basis", "U", "Sigma",
             "V", "inv", "eulerangle", "normalise", "fixed", "single", "graph",
             "sine", "normal_basis", "mfile=", "keep=", "dims=", "tol=", "sine_amp=",
             "sine_period=", "sine_freq=", "sine_offset=", "frame_duration=","control=","character="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
       if opt in ("-h", "--help"):
           usage()
           sys.exit()
       elif opt in ("-a", "--activ_stat"):
           activ_stat = True
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
       elif opt in ("-e", "--eulerangle"):
           eulerangle = True
       elif opt in ("-n", "--normalise"):
           normalise = True
       elif opt in ("-m", "--mfile"):
           motion_files.append(arg)
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
       elif opt in ("-S", "--sine"):
           sine = True
       elif opt in ("-A", "--sine_amp"):
           sine_amp = list(map(float, arg.strip('[]').split(',')))
       elif opt in ("-F", "--sine_freq"):
           sine_freq = list(map(float, arg.strip('[]').split(',')))
       elif opt in ("-P", "--sine_period"):
           sine_period = float(arg)
       elif opt in ("-O", "--sine_offset"):
           sine_offset = list(map(float, arg.strip('[]').split(',')))
       elif opt in ("-D", "--frame_duration"):
           frame_dur = float(arg)
       elif opt in ("-N", "--normal_basis"):
           normal_basis = True
       elif opt in ("-c", "--control"):
           control_file=arg
       elif opt in ("-C", "--character"):
           character=arg
 
    if keep_info is None and n_dims is None:
        keep_info = 0.9

    if eulerangle and n_dims == 36:
        n_dims = 28

    if sine:
        if not single_pca:
            sine_amp = (np.ones(n_dims) * sine_amp[0]).tolist()
            sine_freq = (np.ones(n_dims) * sine_freq[0]).tolist()
            sine_offset = (np.ones(n_dims) * sine_offset[0]).tolist()

    if os.path.isdir(motion_files[0]):
        all_motions = True
        motion_files = glob.glob(motion_files[0] + "{}3d_*.txt".format(character))
        motion_files.sort()

    motion_data = list()
    motion_dict_list = list()
    for m_file in motion_files:
        try:
            with open(m_file) as mf:
                motion_dict = json.load(mf)
        except:
            motion_dict = convert_to_json(m_file)
        motion_data.append(np.array(motion_dict['Frames']))
        motion_dict_list.append(motion_dict)
    motion_data = np.vstack(motion_data)

    # Random shuffle data-points (Won't make sense to shuffle when root NOT fixed)
    # np.random.shuffle(motion_data)

    print("Frames count: ", motion_data.shape[0])

    key_list = ['frame_duration', 'root_position', 'root_rotation']
    
    with open(control_file) as cf:
        motion_control = json.load(cf)
    
    print(motion_control)
    
    quat_trajectory_dict = decompose_quat_trajectories(motion_data, motion_control)
    if normalise and not eulerangle:
        norm_trajectory_dict = normalise_quaternions(quat_trajectory_dict)
    else:
        norm_trajectory_dict = quat_trajectory_dict

    if eulerangle:
        norm_trajectory_dict = convert_quat_to_euler(norm_trajectory_dict, key_list)
    ref_traj_matrix = concatenate_trajectories(norm_trajectory_dict, key_list,
                                               include=False)

    if frame_dur is None:
        durations = np.squeeze(quat_trajectory_dict['frame_duration'].tolist())
        if all(elem in [durations[0], 0.0] for elem in durations):
            frame_dur = durations[0]
            if sine_period is None:
                sine_period = motion_data.shape[0] * frame_dur
        else:
            duration_warning = True
            frame_dur = 1.0/30.0
            if sine_period is None:
                sine_period_warning = True
                sine_period = 1.0

    # Create a TF PCA object
    tf_pca = TF_PCA(ref_traj_matrix)

    # Compute U, ∑ and V
    tf_pca.fit(normalise)

    # Create a new reduced-motion-dictionary
    pca_traj_dict = OrderedDict()
    pca_traj_dict["Loop"] = "wrap"

    # Set the domain of the coordination space (Basis-Vectors - ∑ V^T)
    if eulerangle:
        pca_traj_dict['Domain'] = "Eulerangle"
    else:
        pca_traj_dict['Domain'] = "Quaternion"

    if all_motions:
        pca_traj_dict['all_motions'] = "True"
    else:
        pca_traj_dict['all_motions'] = "False"

    pca_traj_dict['mirrored_motion'] = "False"
    for mFile in motion_files:
        if 'mirrored' in mFile:
            pca_traj_dict['mirrored_motion'] = "True"
            break

    if normal_basis:
        pca_traj_dict['normal_basis'] = "True"
    else:
        pca_traj_dict['normal_basis'] = "False"

    key_list = ['frame_duration', 'root_position', 'root_rotation']

    info_retained, num_dims_retained, pca_traj_dict = \
        pca_extract(tf_pca, pca_traj_dict, norm_trajectory_dict, key_list, motion_control = motion_control,
                    n_dims=n_dims, keep_info=keep_info, pca=pca, reproj=reproj,
                    basis=basis, u_matrix=u_matrix, sigma_matrix=sigma_matrix,
                    v_matrix=v_matrix, inverse=inverse, eulerangle=eulerangle,
                    fixed_root_pos=fixed_root_pos, fixed_root_rot=fixed_root_rot,
                    normalise=normalise, single_pca=single_pca, graph=graph,
                    activ_stat=activ_stat, orth_tol=orth_tol, sine_amp=sine_amp,
                    sine=sine, sine_freq=sine_freq, sine_period=sine_period,
                    sine_offset=sine_offset, frame_dur=frame_dur,
                    normal_basis=normal_basis)

    print("No. of dimensions: ", num_dims_retained)
    print("Keept info: ", info_retained)

    # Create output path and file
    output_file_path = "/home/avbiswas/RLTCA/reduced_motion/pca_"
    output_file = '{}3d_'.format(character)
    print("OUTPUT: ", output_file)
    if all_motions:
        output_file += 'all-motions_'
        if pca_traj_dict['mirrored_motion'] == "True":
            output_file += 'mirrored'
    else:
        for m_file in motion_files:
            motion_name = m_file.split("/")[-1]
            motion_name = motion_name.split(".")[0]
            motion_name = motion_name.split("{}3d_".format(character))[-1]
            motion_name = motion_name.split("mirrored_")[-1]
            if not motion_name in output_file:
                output_file += motion_name + '-'
        # Removing the last '-'
        output_file = output_file[:-1]
    domain = "euler_" if eulerangle else "quat_"
    output_file = output_file_path + domain + output_file + "_" + \
                  str(info_retained) + "_" + str(num_dims_retained) + ".txt"

    # Save pca trajectories and basis dictionary on to the created output file
    with open(output_file, 'w') as fp:
        json.dump(pca_traj_dict, fp, indent=4)

    with open('Output/pca_traj.txt', 'w') as fp:
        json.dump(pca_traj_dict, fp, indent=4)

    if not eulerangle:
        print("WARNING: PCA Not in Euler Angle! Use flag: [-e | --eulerangle]")

    if eulerangle and not normalise:
        print("WARNING: Data Not Normalised! Use flag: [-n | --normalise]")

    if not u_matrix:
        print("WARNING: U Not Saved! Use flag: [-u | --U]")

    if not sigma_matrix:
        print("WARNING: Sigma Not Saved! Use flag: [-z | --Sigma]")

    if not inverse:
        print("WARNING: Inverse Not Saved! Use flag: [-i | --inv]")

    if duration_warning:
        print("WARNING: Multiple frame durations found in input-file!\n",
              "        Setting frame_duration to default value 0.03333.\n",
              "        Use flag: [-D | --frame_duration] to set a different value.\n")

    if sine and sine_period_warning:
        print("WARNING: Sine-Period undefined! Use flag: [-P | --sine_period]\n")

if __name__ == "__main__":
    main(sys.argv[1:])
