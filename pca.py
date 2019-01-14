import numpy as np
import tensorflow as tf
import sys, getopt
import json
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
            keep_info = ladder[n_dims - 1]

        return n_dims, keep_info


    def reduce(self, n_dims=None, keep_info=None):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info)

        with self.graph.as_default():
            # Cut out the relevant part from ∑ and U
            sigma = tf.slice(self.sigma, [0, 0], [n_dims, n_dims])
            u = tf.slice(self.u, [0, 0], [self.data.shape[0], n_dims])

            # PCA
            pca = tf.matmul(u, sigma)

        with tf.Session(graph=self.graph) as session:
            return keep_info, n_dims, session.run(pca,
                                                  feed_dict={self.X: self.data})


    def reproject(self, n_dims=None, keep_info=None):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info)

        with self.graph.as_default():
            # Cut out the relevant part from ∑, U and V
            sigma = tf.slice(self.sigma, [0, 0], [n_dims, n_dims])
            u = tf.slice(self.u, [0, 0], [self.data.shape[0], n_dims])
            v = tf.slice(self.v, [0, 0], [self.data.shape[1], n_dims])

            # Reproject on to linear subspace spanned by Principle Components
            reproj = tf.matmul(u, tf.matmul(sigma, v, transpose_b=True))

        with tf.Session(graph=self.graph) as session:
            return keep_info, n_dims, session.run(reproj,
                                                  feed_dict={self.X: self.data})


    def basis(self, n_dims=None, keep_info=None):
        n_dims, keep_info = self.calc_info_and_dims(n_dims, keep_info)
        return keep_info, n_dims, self.v[:, 0:n_dims]

def plot(data):
    # 3-D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o')

    ax.set_xlabel('PC-1')
    ax.set_ylabel('PC-2')
    ax.set_zlabel('PC-3')

    plt.show()

def decompose_trajectories(motion_data):
    # Decomposes trajectories into indificual DOFs by joint name
    quat_trajs = dict()

    quat_trajs['frame_duration'] = motion_data[:,0:1]
    quat_trajs['root_position'] = motion_data[:,1:4]
    quat_trajs['root_rotation'] = motion_data[:,4:8]

    quat_trajs['chest_rotation'] = motion_data[:,8:12]
    quat_trajs['neck_rotation'] = motion_data[:,12:16]

    quat_trajs['right_hip_rotation'] = motion_data[:,16:20]
    quat_trajs['right_knee_rotation'] = motion_data[:,20:21]
    quat_trajs['right_ankle_rotation'] = motion_data[:,21:25]
    quat_trajs['right_shoulder_rotation'] = motion_data[:,25:29]
    quat_trajs['right_elbow_rotation'] = motion_data[:,29:30]

    quat_trajs['left_hip_rotation'] = motion_data[:,30:34]
    quat_trajs['left_knee_rotation'] = motion_data[:,34:35]
    quat_trajs['left_ankle_rotation'] = motion_data[:,35:39]
    quat_trajs['left_shoulder_rotation'] = motion_data[:,39:43]
    quat_trajs['left_elbow_rotation'] = motion_data[:,43:44]

    return quat_trajs


def main(argv):
    input_file = 'humanoid3d_run.txt'

    pca = False
    reproj = False
    basis = False

    keep_info = None
    n_dims = None

    info_retained = None
    num_dims_retained = None

    try:
        opts, args = getopt.getopt(argv,"hprbi:k:d:",
            ["pca", "reproj", "basis", "ifile=","keep=", "dims="])
    except getopt.GetoptError:
        print("pca.py -i <inputfile> -k <keep_info> -d <num_dims>/'all' -p -r -b")
        sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
           print("pca.py -i <inputfile> -k <keep_info> -d <num_dims>/'all' -p -r -b")
           sys.exit()
       elif opt in ("-p", "--pca"):
           pca = True
       elif opt in ("-r", "--reproj"):
           reproj = True
       elif opt in ("-b", "--basis"):
           basis = True
       elif opt in ("-i", "--ifile"):
           input_file = arg
       elif opt in ("-k", "--keep"):
           keep_info = float(arg)
       elif opt in ("-d", "--dims"):
           if arg.lower() == 'all':
               n_dims = 36
           else:
               n_dims = int(arg)

    if keep_info is None and n_dims is None:
        keep_info = 0.9

    with open(input_file) as f:
        data = json.load(f)

    motion_data = np.array(data['Frames'])

    quat_trajectory_dict = decompose_trajectories(motion_data)

    starting_joint = 8
    original_traj = motion_data[:, starting_joint:44]

    # Create a TF PCA object
    tf_pca = TF_PCA(original_traj)

    # Compute U, ∑ and V
    tf_pca.fit()

    # Create a clone of the input file dictionary
    pca_traj = data.copy()

    if pca:
        # Project the trajectories on to a reduced lower-dimensional space
        info_retained, num_dims_retained, reduced = \
            tf_pca.reduce(keep_info=keep_info, n_dims=n_dims)
        pca_traj['Reduced'] = reduced.tolist()
        #plot(reduced)

    if reproj:
        # Reproject the trajectories on to a linear sub-space
        info_retained, num_dims_retained, reproj_traj = \
            tf_pca.reproject(keep_info=keep_info, n_dims=n_dims)

        # Replace original trajectories, in the controlable DOFs, with
        # reprojected trajectories
        pca_traj['Frames'] = np.column_stack((motion_data[:, 0:starting_joint],
                                                 reproj_traj)).tolist()

    if basis:
        # Get full-rank basis vectors of the linear sub-space
        info_retained, num_dims_retained, basis_v = \
            tf_pca.basis(keep_info=keep_info, n_dims=n_dims)
        pca_traj['Basis'] = basis_v.tolist()


    print("No. of dimensions: ", num_dims_retained)
    print("Keept info: ", info_retained)

    if pca and reproj and basis:
        # Check if U (∑ V^T) = (U ∑) V^T
        re_proj = np.matmul(reduced, np.transpose(basis_v))
        np.testing.assert_array_almost_equal(reproj_traj, re_proj, decimal=6)

    # Create output path and file
    output_file_path = "/home/nash/DeepMimic/data/reduced_motion/pca_"
    output_file = input_file.split("/")[-1]
    output_file = output_file.split(".")[0]
    output_file = output_file_path + output_file + "_" + str(info_retained) + \
                  "_" + str(num_dims_retained) + ".txt"

    # Save pca trajectories and basis dictionary on to the created output file
    with open(output_file, 'w') as fp:
        json.dump(pca_traj, fp, indent=4)

    with open('pca_traj.txt', 'w') as fp:
        json.dump(pca_traj, fp, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])
