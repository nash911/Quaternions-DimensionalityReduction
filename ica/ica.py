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
        self.data = data
        self.ica = None
        self.reconstructed_signals = None
        self.mixing_matrix = None
        self.ica_mean = None

    def fit(self, num_dims):
        self.ica = FastICA(n_components=num_dims)
        self.reconstructed_signals = self.ica.fit_transform(self.data)
        self.mixing_matrix = self.ica.mixing_  # Get estimated mixing matrix
        self.ica_mean = self.ica.mean_

    def reproject(self, comp=None, single_component=False):
        if single_component:
            signal = self.reconstructed_signals[:,[comp]]
            mixing_component = self.mixing_matrix.T[[comp],:]
            return np.dot(signal, mixing_component) + self.ica.mean_
        else:
            return np.dot(self.reconstructed_signals, self.mixing_matrix.T) + \
                self.ica.mean_

    def signals(self):
        return self.reconstructed_signals#[:,:3] #self.reconstructed_signals

    def basis(self):
        return self.mixing_matrix.T

    def mean(self):
        return self.ica_mean

def plot(data):
    # 3-D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o')

    ax.set_xlabel('PC-1')
    ax.set_ylabel('PC-2')
    ax.set_zlabel('PC-3')

    plt.show()


def concatenate_trajectories(trajs_dict, key_list = [], include=False, tilt=False,
                             fixed_root_pos=False, fixed_root_rot=False):
    trajs_data = []
    for k, v in trajs_dict.items():
        if include:
            if k in key_list:
                if k == 'root_position' and fixed_root_pos:
                    v = (np.ones_like(v) * v[0]).tolist()
                if k == 'root_rotation' and fixed_root_rot:
                    if tilt:
                        v = (np.ones_like(v) * v[0]).tolist()
                    else:
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


def ica_extract(ica, ica_traj_dict, trajectory_dict, key_list, comp=None,
                reproj=True, basis=True, eulerangle=False, fixed_root_pos=False,
                fixed_root_rot=False, tilt=False, single_component=False,
                graph=False, inverse=True):
    if reproj:
        # Reproject the trajectories on to a linear sub-space in the full space:
        # X = SW
        reproj_traj = ica.reproject(comp=comp, single_component=single_component)

        # Replace original trajectories, in the controlable DOFs, with
        # reprojected trajectories
        unchanged_traj = concatenate_trajectories(trajectory_dict, key_list,
                                                  include=True, tilt=tilt,
                                                  fixed_root_pos=fixed_root_pos,
                                                  fixed_root_rot=fixed_root_rot)
        # Remove non-controlable DOFs from reprojected trajectories, if exists
        reproj_traj = reproj_traj[:, -28:reproj_traj.shape[1]]
        ica_traj_dict['Frames'] = \
            np.column_stack((unchanged_traj, reproj_traj)).tolist()

        if eulerangle:
            # Convert back to Quaternions
            mixed_dict = decompose_euler_trajectories(np.array(ica_traj_dict['Frames']))
            quat_dict = convert_euler_to_quat(mixed_dict, key_list)
            concat_quat_trajs = concatenate_trajectories(quat_dict)
            ica_traj_dict['Frames'] = concat_quat_trajs.tolist()

    if basis:
        # Get full-rank basis vectors of the linear sub-space: ∑ V^T
        basis_v = ica.basis()
        mean = ica.mean()

        ica_traj_dict['Basis'] = basis_v.tolist()
        ica_traj_dict['Reference_Mean'] = mean.tolist()

        if inverse:
            basis_v_pinv = pinv(ica.basis())
            ica_traj_dict['Basis_Inv'] = basis_v_pinv.tolist()

    if graph:
        plot(ica.signals())

    return ica_traj_dict


def usage():
    print("Usage: ica.py [-b | --basis] \n"
          "              [-d | --dims] <no. of dims>/'all' \n"
          "              [-e | --eulerangle] \n"
          "              [-f | --fixed] \n"
          "              [-g | --graph] \n"
          "              [-h | --help] \n"
          "              [-i | --inv] \n"
          "              [-m | --mfile] <input motion file> \n"
          "              [-n | --normalize] \n"
          "              [-r | --reproj] \n"
          "              [-s | --single] \n"
          "              [-t | --tilt] \n"
          )


def main(argv):
    basis = False
    num_dims = None
    eulerangle = False
    fixed_root_pos = False
    fixed_root_rot = False
    graph = False
    inverse = False
    motion_file = None
    normalise = False
    reproj = False
    single_component = False
    tilt = False


    try:
        opts, args = getopt.getopt(argv,"befghinrstd:m:",
            ["basis", "eulerangle", "fixed", "graph", "help", "inv", "normalize",
             "reproj", "single", "tilt", "dims=", "mfile="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
       if opt in ("-b", "--basis"):
           basis = True
       elif opt in ("-d", "--dims"):
           if arg.lower() == 'all':
               num_dims = 28
           else:
               num_dims = int(arg)
       elif opt in ("-e", "--eulerangle"):
           eulerangle = True
       elif opt in ("-f", "--fixed"):
           fixed_root_pos = True
           fixed_root_rot = True
       elif opt in ("-g", "--graph"):
           graph = True
       elif opt in ("-h", "--help"):
           usage()
           sys.exit()
       elif opt in ("-i", "--inv"):
           inverse = True
       elif opt in ("-m", "--mfile"):
           motion_file = arg
       elif opt in ("-n", "--normalize"):
           normalise = True
       elif opt in ("-r", "--reproj"):
           reproj = True
       elif opt in ("-s", "--single"):
           single_component = True
       elif opt in ("-t", "--tilt"):
           tilt = True


    with open(motion_file) as f:
        data = json.load(f)

    motion_data = np.array(data['Frames'])

    key_list = ['frame_duration', 'root_position', 'root_rotation']

    quat_trajectory_dict = decompose_quat_trajectories(motion_data)
    if normalise:
        norm_trajectory_dict = normalize_quaternions(quat_trajectory_dict)
    else:
        norm_trajectory_dict = quat_trajectory_dict

    if eulerangle:
        norm_trajectory_dict = convert_quat_to_euler(norm_trajectory_dict,
                                                          key_list)
    quat_traj_matrix = concatenate_trajectories(norm_trajectory_dict,
                                                key_list, include=False)

    # Create a ICA object
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

    if single_component:
        for dim in range(num_dims):
            ica_traj_dict = ica_extract(ica, ica_traj_dict, norm_trajectory_dict,
                            key_list, comp=dim, reproj=reproj, basis=basis,
                            eulerangle=eulerangle, fixed_root_pos=fixed_root_pos,
                            fixed_root_rot=fixed_root_rot, tilt=tilt, graph=graph,
                            single_component=True, inverse=False)

            # Create output path and file
            output_file = 'ica_traj' + '_comp-' + str(dim+1) + ".txt"

            # Save ica trajectories and basis dictionary on to the created output file
            with open(output_file, 'w') as fp:
                json.dump(ica_traj_dict, fp, indent=4)
    else:
    #if True:
        ica_traj_dict = ica_extract(ica, ica_traj_dict, norm_trajectory_dict,
                        key_list, comp=None, reproj=reproj, basis=basis,
                        eulerangle=eulerangle, fixed_root_pos=fixed_root_pos,
                        fixed_root_rot=fixed_root_rot, tilt=tilt, graph=graph,
                        single_component=False, inverse=inverse)

        print("No. of components: ", num_dims)

        # Create output path and file
        output_file_path = "/home/nash/DeepMimic/data/reduced_motion/ica_"
        output_file = motion_file.split("/")[-1]
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
