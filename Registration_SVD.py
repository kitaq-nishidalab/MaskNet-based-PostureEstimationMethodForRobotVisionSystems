import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation
from numpy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
matplotlib.use('Agg')  # 非GUIモードのバックエンドを使用

###################
#Registration
###################
# ICP registration module.
class SVD:
	def __init__(self, threshold=0.1, max_iteration=100):
		# threshold: 			Threshold for correspondences. (scalar)
		# max_iterations:		Number of allowed iterations. (scalar)
		self.threshold = threshold
		self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)

	def upsampling(self, pcd, number=1000):
		tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
		# 点群をメッシュ化（Convex Hullを使用）
		mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.01, tetra_mesh=tetra_mesh, pt_map=pt_map)
		# メッシュをサンプルして、点群をアップサンプリング
		pcd_upsampled = mesh.sample_points_uniformly(number_of_points=number)  # 1000点にアップサンプリング

		return pcd_upsampled

	# Preprocess template, source point clouds.
	def preprocess(self, template, source, target):
		if self.is_tensor: template, source = template.detach().cpu().numpy(), source.detach().cpu().numpy()	# Convert to ndarray if tensors.
		if len(template.shape) > 2: 						# Reduce dimension to [N, 3]
			template, source = template[0], source[0]

		template_pcd = o3d.geometry.PointCloud()
		source_pcd = o3d.geometry.PointCloud()

		#model_pcd.points = o3d.utility.Vector3dVector(model)
		template_pcd.points = o3d.utility.Vector3dVector(template)
		source_pcd.points = o3d.utility.Vector3dVector(source)

		# Find mean of template & source.
		self.template_mean = np.mean(np.array(template_pcd.points), axis=0, keepdims=True)
		self.source_mean = np.mean(np.array(source_pcd.points), axis=0, keepdims=True)
		#model_mean = np.mean(np.array(model_pcd.points), axis=0, keepdims=True)

		# Convert to open3d point clouds.
		template_ = o3d.geometry.PointCloud()
		source_ = o3d.geometry.PointCloud()
		#model_ = o3d.geometry.PointCloud()

		# Subtract respective mean from each point cloud.
		template_.points = o3d.utility.Vector3dVector(np.array(template_pcd.points) - self.template_mean)
		source_.points = o3d.utility.Vector3dVector(np.array(source_pcd.points) - self.source_mean)

		### Voxel downsamplig
		# if target=="t":
		# 	voxel_size = 0.004 #0.02(Haris)
		# 	template_ = template_.voxel_down_sample(voxel_size)
		# 	source_ = source_.voxel_down_sample(voxel_size)
		#voxel_size = 0.004 #0.02(Haris)
		#template_ = template_.voxel_down_sample(voxel_size)
		#source_ = source_.voxel_down_sample(voxel_size)
		return template_, source_

	# Postprocess on transformation matrix.
	def postprocess(self, res):
		# Way to deal with mean substraction
		# Pt = R*Ps + t 								original data (1)
		# Pt - Ptm = R'*[Ps - Psm] + t' 				mean subtracted from template and source.
		# Pt = R'*Ps + t' - R'*Psm + Ptm 				rearrange the equation (2)
		# From eq. 1 and eq. 2,
		# R = R' 	&	t = t' - R'*Psm + Ptm			(3)
		est_R = res.transformation[0:3, 0:3]		# ICP's rotation matrix (source -> template)
		t_ = np.array(res.transformation[0:3, 3]).reshape(1, -1)			# ICP's translation vector (source -> template)
		est_T = np.array(res.transformation)								# ICP's transformation matrix (source -> template)
		est_t = np.matmul(est_R, -self.source_mean.T).T + t_ + self.template_mean[0] 	# update predicted translation according to eq. 3
		est_T[0:3, 3] = est_t
		#print(self.source_mean)
		return est_R, est_t, est_T

	# Convert result to pytorch tensors.
	@staticmethod
	def convert2tensor(result):
		if torch.cuda.is_available(): device = 'cuda'
		else: device = 'cpu'
		result['est_R']=torch.tensor(result['est_R']).to(device).float().view(-1, 3, 3) 		# Rotation matrix [B, 3, 3] (source -> template)
		result['est_t']=torch.tensor(result['est_t']).to(device).float().view(-1, 1, 3)			# Translation vector [B, 1, 3] (source -> template)
		result['est_T']=torch.tensor(result['est_T']).to(device).float().view(-1, 4, 4)			# Transformation matrix [B, 4, 4] (source -> template)
		return result

	# icp registration.
	def __call__(self, template, source, target):
		self.is_tensor = torch.is_tensor(template)
		######## tensorからo3d #########
		template, source= self.preprocess(template, source, target)
		
		########################################################
		### Method using SVD（概略マッチング） ###
		########################################################
		######## マスクテンプレの主成分分析 #########
		T_trans = np.array(template.points)
		T_trans -= T_trans.mean(axis=0)
		# PCA実行
		T_pca = PCA(n_components=3)  # 3成分を取得するために n_components=3
		T_pca.fit(T_trans)
		# 固有値、固有ベクトルの取得
		#T_W = T_pca.explained_variance_  # 固有値
		T_V_pca = T_pca.components_.T  # 固有ベクトル (sklearnでは転置されている)
		######### ソースの主成分分析 ###########
		S_trans = np.array(source.points)
		S_trans -= S_trans.mean(axis=0)
		# PCA実行
		S_pca = PCA(n_components=3)  # 3成分を取得するために n_components=3
		S_pca.fit(S_trans)
		# 固有値、固有ベクトルの取得
		#S_W = S_pca.explained_variance_  # 固有値
		S_V_pca = S_pca.components_.T  # 固有ベクトル (sklearnでは転置されている)#
		########## 鏡像マッチング対策 ############
		#print("\n", np.linalg.det(S_V_pca @ T_V_pca.T))
		#print(np.linalg.det(S_V_pca @ T_V_pca.T))
		if np.linalg.det(S_V_pca @ T_V_pca.T) < 0:
			if target=="t":
				K = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]])     # T joint pipe
			elif target=="l":
				K = np.array([[1, 0, 0],[0, -1, 0],[0, 0, 1]])     # L joint pipe
			R = S_V_pca @ K @ T_V_pca.T
		else:
			R = S_V_pca @ T_V_pca.T
		###転置処理###
		R = R.T
		transformation = np.eye(4)
		for i in range(3):
			for j in range(3):
				transformation[i][j] = R[i][j]
		#print(transformation)
		########################################################
		########################################################
		
		### ICPアルゴリズム（精密マッチング） ###
		res = o3d.pipelines.registration.registration_icp(source, template, self.threshold, transformation, criteria=self.criteria)	# icp registration in open3d.
		#print("transformation:\n", res.transformation)
		est_R, est_t, est_T = self.postprocess(res)
		result = {'est_R': est_R,
		          'est_t': est_t,
		          'est_T': est_T}
		if self.is_tensor: result = self.convert2tensor(result)
		return result

# Define Registration Algorithm.
def registration_algorithm():
  reg_algorithm = SVD()
  return reg_algorithm

# Register template and source pairs.
class Registration:
	def __init__(self):
		self.reg_algorithm = registration_algorithm()

	@staticmethod
	def pc2points(data):
		if len(data.shape) == 3:
			return data[:, :, :3]
		elif len(data.shape) == 2:
			return data[:, :3]

	def register(self, template, source, target):
		# template, source: 		Point Cloud [B, N, 3] (torch tensor)
		result = self.reg_algorithm(template, source, target)
		return result