##############################
# MaskNetによるマスク推定の精度を評価するスクリプト
#T joint pipe::  従来手法：69.4%　　提案手法：78.4%
#L joint pipe::  従来手法：70.4%　　提案手法：74.6%
##############################
import numpy as np
import torch
import open3d as o3d
import Registration_SVD as Registration
import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import PointCloud2
import copy
from geometry_msgs.msg import TransformStamped
import tf_conversions
import os
import tf2_ros
from tf2_msgs.msg import TFMessage
from filterpy.kalman import KalmanFilter
import time
import pandas as pd
import matplotlib.pyplot as plt
import network

def create_pcd(xyz):
    n = xyz.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def inverse_transform_matrix(transform_matrix):
    # 変換行列の逆行列を計算
    return np.linalg.inv(transform_matrix)

# KDTreeでマスクを生成する関数
def generate_mask_from_template(template_pcd, model_pcd):
	template_numpy = template_pcd.detach().cpu().numpy()[0]  # テンソル → NumPy 配列
	masked_template_pcd = o3d.geometry.PointCloud()  # Open3D の PointCloud 作成
	masked_template_pcd.points = o3d.utility.Vector3dVector(template_numpy)  # NumPy → PointCloud

    # KDTree を作成
	model_points = np.asarray(model_pcd.points)
	kdtree = o3d.geometry.KDTreeFlann(model_pcd)
    
    # マスクの初期化
	predicted_mask_from_template = np.zeros(len(model_points), dtype=np.int32)
    
    # template_pcd の各点について最近傍探索を実行
	template_points = np.asarray(masked_template_pcd.points)
	for point in template_points:
		[success, idx, _] = kdtree.search_knn_vector_3d(point, 1)  # 最も近い1点を探索
		if success > 0:
			predicted_mask_from_template[idx[0]] = 1  # 最近傍点を1とする
    
	return predicted_mask_from_template


class PointCloudProcessor:
	def __init__(self):
		rospy.init_node('posture_estimation', anonymous=True)
		
		###########################
		#Target change
		###########################
		self.target = "t"
		###########################

        
		#TFブロードキャスト
		self.br = tf2_ros.StaticTransformBroadcaster()
		self.frame_id = None
		self.child_frame_id = None
		self.sub_tf = rospy.Subscriber('/tf_static', TFMessage, self.tf_callback, queue_size=100000)

		self.measured_pcd = None
		if self.target=="t":
			self.model_pcd = o3d.io.read_point_cloud("TNUTEJN016_10000_down.ply")
		elif self.target=="l":
			self.model_pcd = o3d.io.read_point_cloud("WMU2LR2020_10000_down.ply")
		
		self.filter =  o3d.io.read_point_cloud("/home/nishidalab0/vision_ws_blender/output/Tpipe/FilterMasks/filter.ply")

		self.checkpoint = torch.load("checkpoint/model_epoch500_45deg.pth")
		self.masknet_load = network.MaskNet()
		self.masknet_load.load_state_dict(self.checkpoint)

		#self.masknet_load = torch.load("checkpoint/model_weight_epoch300_717_batchsize32.pth")
		self.masknet_load = torch.load("checkpoint/model_weight_epoch300_batchsize32_plane.pth")
		#self.masknet_load = torch.load("checkpoint/unnoise_transformed_epoch100.pth")
		#self.masknet_load = torch.load("checkpoint/pretrained.pth")
		#self.masknet_load = torch.load("checkpoint/first_train.pth")
		#self.masknet_load = torch.load("checkpoint/model_epoch300_45deg_shins2.pth")
		#self.masknet_load = torch.load("checkpoint/unnoise_transformed_epoch100_plane.pth")
		
		self.sub = rospy.Subscriber('/processed_point_cloud', PointCloud2, self.point_cloud_callback, queue_size=100000)

		self.accuracy_list = []  # 一致率を保存するリスト
	

	def point_cloud_callback(self, msg):
		# Convert ROS PointCloud2 message to Open3D PointCloud
		pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
		points = np.array(list(pc_data))
		if points.size == 0:
			return
		xyz = points[:, :3]
		try:
			self.measured_pcd = create_pcd(xyz)

			self.process_point_cloud()
		except ValueError as e:
			rospy.logerr("ValueError: %s", str(e))

		
	def tf_callback(self, msg):
		if self.frame_id is not None:
			return
		for transform in msg.transforms:
			if transform.child_frame_id == "camera_depth_optical_frame":
				self.frame_id = transform.header.frame_id
				self.child_frame_id = transform.child_frame_id
				self.translation_x = transform.transform.translation.x
				self.translation_y = transform.transform.translation.y
				self.translation_z = transform.transform.translation.z
				self.rotation_x = transform.transform.rotation.x
				self.rotation_y = transform.transform.rotation.y
				self.rotation_z = transform.transform.rotation.z
				self.rotation_w = transform.transform.rotation.w

			
	def process_point_cloud(self):
		torch.cuda.empty_cache()
		if self.measured_pcd is None:
			return
    
    	# 計測点群とモデル点群の取得
		numpy_model_pcd = np.array(self.model_pcd.points)
		numpy_measured_pcd = np.array(self.measured_pcd.points)
		len_measured_pcd = len(numpy_measured_pcd)

		#model_pcd = self.model_pcd.voxel_down_sample(voxel_size)
    		
		# ダウンサンプリングとトレース（元のインデックスを取得）
		model_pcd = self.model_pcd
    		
    
    	# 点群のデバイスへの移行
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.masknet_load.to(device)
		template_tensor = torch.tensor(np.array(model_pcd.points), dtype=torch.float32).unsqueeze(0).to(device)
		source_tensor = torch.tensor(numpy_measured_pcd, dtype=torch.float32).unsqueeze(0).to(device)

		print(template_tensor.shape) 
		print(source_tensor.shape) 
    	# MaskNetによるマスク推定
		with torch.no_grad():
			masked_template_cheese, predicted_mask_cheese = self.masknet_load(template_tensor, source_tensor)

		# 推定されたマスクをCPUに移行
		#predicted_mask = predicted_mask_cheese.detach().cpu().numpy()[0]
		predicted_mask = generate_mask_from_template(masked_template_cheese, model_pcd)

		#threshold = 0.5  # しきい値を設定
		#predicted_mask[predicted_mask > threshold] = 1  # しきい値を超えると1にする
		#predicted_mask[predicted_mask <= threshold] = 0
			
		masked_pcd = o3d.geometry.PointCloud()
		#masked_pcd.points = o3d.utility.Vector3dVector(masked_template_cheese.detach().cpu().numpy()[0])
		masked_pcd.points = o3d.utility.Vector3dVector(np.asarray(model_pcd.points)[predicted_mask == 1])
		masked_pcd.paint_uniform_color([0, 0, 1])
		o3d.io.write_point_cloud("mask.pcd", masked_pcd)

    
    	# 保存済みマスクのロード
		if self.target=="t":
			saved_mask_path = "/home/nishidalab0/vision_ws_blender/output/Tpipe/gtmask/mask.npy"  # 保存済みマスクのパス
		elif self.target=="l":
			saved_mask_path = "/home/nishidalab0/vision_ws_blender/output/Lpipe/gtmask/mask.npy"
		if not os.path.exists(saved_mask_path):
			rospy.logerr(f"Saved mask file not found at {saved_mask_path}")
			return
		saved_mask = np.load(saved_mask_path)
    		
		#print(saved_mask)
		print(np.shape(saved_mask))
		#print(predicted_mask)
		print(np.shape(predicted_mask))

		extracted_pcd = o3d.geometry.PointCloud()
		extracted_pcd.points = o3d.utility.Vector3dVector(np.asarray(model_pcd.points)[saved_mask > 0])
		extracted_pcd.paint_uniform_color([1, 0, 0])
		o3d.io.write_point_cloud("extracted.pcd", extracted_pcd)

		self.filter.paint_uniform_color([0, 1, 0])

    	# 一致率 (Accuracy) 計算
		if saved_mask.shape != predicted_mask.shape:
			rospy.logerr("Saved mask and predicted mask have different shapes")
			return
		accuracy = np.sum(predicted_mask == saved_mask) / len(saved_mask)
		print(f"Accuracy: {accuracy:.2%}")
		o3d.visualization.draw_geometries([masked_pcd])

		self.accuracy_list.append(accuracy)

        # 一致率が100回に達した場合、平均を計算して表示
		if len(self.accuracy_list) == 100:
			mean_accuracy = np.mean(self.accuracy_list)
			print(f"Mean Accuracy over 100 runs: {mean_accuracy:.2%}") 
			self.accuracy_list = []  # リセット

def main():
    processor = PointCloudProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()
