import numpy as np
from numpy.linalg import pinv
import math
from collections import OrderedDict
import sys, getopt
import json
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ortho_group
from sklearn.decomposition import FastICA


sys.path.insert(0, '/home/nash/Dropbox/Clemson/Projects/quat_conversions/pca')
from Quaternion import Quat, normalize


class ICA:
    def __init__(self, data):
        self.signals = data
        self.ica = None
        self.reconstructed_signals = None
        self.mixing_matrix = None
        self.ica_mean = None

    def fit(self, num_dims):
        self.ica = FastICA(n_components=num_dims)
        self.reconstructed_signals = self.ica.fit_transform(self.signals)
        self.mixing_matrix = self.ica.mixing_  # Get estimated mixing matrix
        self.ica_mean = self.ica.mean_

        print("self.reconstructed_signals.shape: ", self.reconstructed_signals.shape)
        d = np.dot(self.reconstructed_signals, self.mixing_matrix.T)
        print("d.shape: ", d.shape)


    def reproject(self, num_dims=None, single_ica=False):
        #return self.reconstructed_signals
        return np.dot(self.reconstructed_signals, self.mixing_matrix.T) + self.ica.mean_

    def basis(self, num_dims=None, single_ica=False):
        return self.mixing_matrix.T

    def mean(self, num_dims=None, single_ica=False):
        return self.ica_mean

    def pc_mean_std(self, num_dims=None):
        num_dims, keep_info = self.calc_info_and_dims(num_dims)

        with self.graph.as_default():
            # Cut out the relevant part from U
            u = tf.slice(self.u, [0, 0], [self.data.shape[0], num_dims])
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


def decompose_quat_trajectories(motion_data):
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


def decompose_euler_trajectories(motion_data):
    # Decomposes trajectories into indificual DOFs by joint name
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


def check_orthogonality(c_vecs, num_dims, decimal=1e-6):
    num_vecs = c_vecs.shape[0]
    vec_list = [c_vecs[i, :] for i in range(num_vecs)]

    orthogonal = True

    mat_mul = np.matmul(c_vecs, c_vecs.T)
    if np.allclose(mat_mul, np.eye(abs(num_dims)), rtol=1e-05, atol=1e-05):
        print("WARNING: Basis vectors are Normal!")

    for i in range(num_vecs):
        for j in range(i+1, num_vecs):
            dot_prod = np.matmul(vec_list[i], vec_list[j].T)
            if dot_prod > decimal:
                print(dot_prod)
                orthogonal = False
                break

    return orthogonal


def ica_extract(ica, ica_traj_dict, trajectory_dict, key_list, num_dims, reproj=True,
                basis=True, axisangle=False, eulerangle=False, graph=False,
                fixed_root_pos=False, fixed_root_rot=False, single_ica=False):
    if reproj:
        # Reproject the trajectories on to a linear sub-space in the full space: U ∑ V^T
        reproj_traj = ica.reproject(num_dims=num_dims, single_ica=single_ica)

        # Replace original trajectories, in the controlable DOFs, with
        # reprojected trajectories
        unchanged_traj = concatenate_trajectories(trajectory_dict, key_list,
                                                  include=True,
                                                  fixed_root_pos=fixed_root_pos,
                                                  fixed_root_rot=fixed_root_rot)
        # Remove non-controlable DOFs from reprojected trajectories, if exists
        reproj_traj = reproj_traj[:, -36:reproj_traj.shape[1]]
        ica_traj_dict['Frames'] = np.column_stack((unchanged_traj,
                                              reproj_traj)).tolist()

        if axisangle:
            # Convert back to Quaternions
            mixed_dict = decompose_quat_trajectories(np.array(ica_traj_dict['Frames']))
            quat_dict = convert_to_quaternion(mixed_dict, key_list)
            concat_quat_trajs = concatenate_trajectories(quat_dict)
            ica_traj_dict['Frames'] = concat_quat_trajs.tolist()

        if eulerangle:
            # Convert back to Quaternions
            mixed_dict = decompose_euler_trajectories(np.array(ica_traj_dict['Frames']))
            quat_dict = convert_euler_to_quat(mixed_dict, key_list)
            concat_quat_trajs = concatenate_trajectories(quat_dict)
            ica_traj_dict['Frames'] = concat_quat_trajs.tolist()

    if basis:
        # Get full-rank basis vectors of the linear sub-space: ∑ V^T
        basis_v = ica.basis(num_dims=num_dims, single_ica=single_ica)
        mean = ica.mean(num_dims=num_dims, single_ica=single_ica)

        # if not check_orthogonality(basis_v, num_dims, decimal=1e-5):
        #     if os.path.exists('ica_traj.txt'):
        #         os.remove('ica_traj.txt')
        #     print("Error: Basis Vectors not Orthogonal!")
        #     sys.exit()

        print("basis_v.shape: ", basis_v.shape)

        ica_traj_dict['Basis'] = basis_v.tolist()
        ica_traj_dict['Reference_Mean'] = mean.tolist()

    return ica_traj_dict


def main(argv):
    input_file = 'humanoid3d_run.txt'
    num_dims = None

    reproj = False
    basis = False
    axisangle = False
    eulerangle = False
    normalise = False
    fixed_root_pos = False
    fixed_root_rot = False
    single_ica = False
    graph = False



    try:
        opts, args = getopt.getopt(argv,"hrmaenfsgi:d:",
            ["help", "reproj", "mixture", "axisangle", "eulerangle", "normalize",
             "fixed", "single", "graph", "ifile=", "dims="])
    except getopt.GetoptError:
        print("Usage: ica.py [-i  | --ifile] <inputfile> [-k | --keep] <keep_info>\n",
              "             [-d | --dims] <num_dims>/'all' [-p | --ica] [-r | --reproj] \n",
              "             [-b | --basis] [-u | --U] [-z | --Sigma] [-v | --V]\n",
              "             [-j | --inv] [-a | --axisangle] [-e | --eulerangle] \n",
              "             [-n | --normalize] [-f | --fixed] [-s | --single] \n",
              "             [-g | --graph], [-h | --help]")
        sys.exit(2)

    for opt, arg in opts:
       if opt in ("-h", "--help"):
           print("Usage: ica.py [-i  | --ifile] <inputfile> [-k | --keep] <keep_info>\n",
                 "             [-d | --dims] <num_dims>/'all' [-p | --ica] [-r | --reproj] \n",
                 "             [-b | --basis] [-u | --U] [-z | --Sigma] [-v | --V]\n",
                 "             [-j | --inv] [-a | --axisangle] [-e | --eulerangle] \n",
                 "             [-n | --normalize] [-f | --fixed] [-s | --single] \n",
                 "             [-g | --graph], [-h | --help]")
           sys.exit()
       elif opt in ("-r", "--reproj"):
           reproj = True
       elif opt in ("-m", "--mixture"):
           basis = True
       elif opt in ("-a", "--axisangle"):
           axisangle = True
           if os.path.exists('ica_traj.txt'):
               os.remove('ica_traj.txt')
           print("ICA in Axis-Angle currently not supposted!")
           sys.exit()
       elif opt in ("-e", "--eulerangle"):
           eulerangle = True
       elif opt in ("-n", "--normalize"):
           normalise = True
       elif opt in ("-i", "--ifile"):
           input_file = arg
       elif opt in ("-d", "--dims"):
           if arg.lower() == 'all':
               num_dims = 36
           else:
               num_dims = int(arg)
       elif opt in ("-f", "--fixed"):
           fixed_root_pos = True
           fixed_root_rot = True
       elif opt in ("-s", "--single"):
           single_ica = True
       elif opt in ("-g", "--graph"):
           graph = True

    with open(input_file) as f:
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

    # Create a ICA object
    if axisangle:
        ica = ICA(axisangle_traj_matrix)
    else:
        ica = ICA(quat_traj_matrix)

    # Compute U, ∑ and V
    ica.fit(num_dims)

    # Create a clone of the input file dictionary
    ica_traj_dict = data.copy()

    # Set the domain of the coordination space (Basis-Vectors - ∑ V^T)
    if eulerangle:
        ica_traj_dict['Domain'] = "Eulerangle"
    else:
        ica_traj_dict['Domain'] = "Quaternion"

    key_list = ['frame_duration', 'root_position', 'root_rotation']

    ica_traj_dict = ica_extract(ica, ica_traj_dict, norm_trajectory_dict,
                    key_list, num_dims=num_dims, reproj=reproj, basis=basis,
                    axisangle=axisangle, eulerangle=eulerangle, graph=graph,
                    fixed_root_pos=fixed_root_pos, fixed_root_rot=fixed_root_rot,
                    single_ica=single_ica)

    print("No. of dimensions: ", num_dims)

    # Create output path and file
    output_file_path = "/home/nash/DeepMimic/data/reduced_motion/ica_"
    output_file = input_file.split("/")[-1]
    output_file = output_file.split(".")[0]
    domain = "euler_" if eulerangle else "quat_"
    output_file = output_file_path + domain + output_file + "_" + \
                  str(num_dims) + ".txt"

    # Save ica trajectories and basis dictionary on to the created output file
    with open(output_file, 'w') as fp:
        json.dump(ica_traj_dict, fp, indent=4)

    with open('ica_traj.txt', 'w') as fp:
        json.dump(ica_traj_dict, fp, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])
