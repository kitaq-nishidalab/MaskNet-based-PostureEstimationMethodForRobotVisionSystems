#! /usr/bin/env python3
# -*- coding: utf-8 -*-

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
from tf.transformations import quaternion_multiply, quaternion_conjugate

def create_pcd(xyz):
    n = xyz.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def inverse_transform_matrix(transform_matrix):
    # 変換行列の逆行列を計算
    return np.linalg.inv(transform_matrix)

class PointCloudProcessor:
	def __init__(self):
		rospy.init_node('posture_estimation', anonymous=True)
		
		###########################
		#Target change
		###########################
		self.target = "t"
		###########################
		
		# カルマンフィルタの初期化
		self.kf = KalmanFilter(dim_x=7, dim_z=7)  # 6次元状態: x, y, z, roll, pitch, yaw
		self.kf.F = np.eye(7)  # 状態遷移行列 A
		self.kf.H = np.eye(7)  # 観測行列 C
		self.kf.P *= 1  # 初期共分散行列 P
		### 0.003, 0.003, 0.003, 0.034, 0.034, 0.034, 0.005
		self.kf.Q = np.diag([0.003, 0.003, 0.003, 0.034, 0.034, 0.034, 0.005])   # プロセスノイズ　v_k の共分散 
		self.kf.R = np.diag([0.003, 0.003, 0.003, 0.034, 0.034, 0.034, 0.005])   # 観測ノイズ　w_k の共分散　　
		self.is_initialized = False  # 初期化フラグを追加
        
		#TFブロードキャスト
		# self.br = tf2_ros.StaticTransformBroadcaster()
		self.br = tf2_ros.TransformBroadcaster()
		self.frame_id = None
		self.child_frame_id = None
		self.sub_tf = rospy.Subscriber('/tf_static', TFMessage, self.tf_callback, queue_size=100000)

		self.measured_pcd = None
		if self.target=="t":
			self.model_pcd = o3d.io.read_point_cloud("../TNUTEJN016_10000.pcd")
		elif self.target=="l":
			self.model_pcd = o3d.io.read_point_cloud("../WMU2LR2020_10000.pcd")
		
		# Initialize the posture estimation flag
		rospy.set_param('/posture_estimation_done', True)
		
		self.checkpoint = torch.load("../checkpoint/model_epoch500_45deg.pth")
		self.masknet_load = network.MaskNet()
		self.masknet_load.load_state_dict(self.checkpoint)
		# Timer to monitor parameter
		self.timer = rospy.Timer(rospy.Duration(0.005), self.check_posture_estimation_done)

		self.sub = rospy.Subscriber('/processed_point_cloud', PointCloud2, self.point_cloud_callback, queue_size=100000)
		#self.pub = rospy.Publisher('/estimated_transform', TransformStamped, queue_size=10)
		
		self.processing_times = []
		self.inference_times = []
		self.control_times = []
		self.start_masknet_time = 0
		self.end_masknet_time = 0
		self.start_kalman_time = 0
		self.end_kalman_time = 0

		# 新しい属性を初期化
		self.prev_translation = None
		self.prev_euler_angles = None
		self.last_update_time = rospy.Time.now()  # 最初の更新時刻を設定

		self.process_flag = True

		
	def check_posture_estimation_done(self, event):
        # Check the value of '/posture_estimation_done' parameter
		posture_done = rospy.get_param('/posture_estimation_done')
		#print(posture_done)
		if posture_done:
			#rospy.loginfo("Posture estimation is done. Skipping point cloud processing.")
			if self.sub:  # サブスクライバが存在していれば解除
				self.sub.unregister()
				self.sub = None  # サブスクライバを明示的にNoneに設定
				rospy.loginfo("Subscriber has been unregistered.")
		else:
			self.process_flag = False
			if not self.sub:  # サブスクライバがない場合にのみ新しく作成
				rospy.loginfo("Resumed point cloud processing.")
				self.sub = rospy.Subscriber('/processed_point_cloud', PointCloud2, self.point_cloud_callback, queue_size=100000)

	def point_cloud_callback(self, msg):
		if self.process_flag:
			return
		#rospy.loginfo("Processing point cloud.")
		
		start_time = time.time()  # 開始時刻
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
			
		# 処理時間の記録
		end_time = time.time()  # 終了時刻
		processing_time = end_time - start_time
		# 推論時間を記録
		inference_time = self.end_masknet_time - self.start_masknet_time
		# カルマンフィルタの処理時間を記録
		control_time = self.end_kalman_time - self.start_kalman_time
		
		self.processing_times.append(processing_time)
		self.inference_times.append(inference_time)
		self.control_times.append(control_time)
			
		# 処理回数が1000回に到達したら保存してグラフを表示
		if len(self.processing_times) < 1000:
			self.processing_times.append(processing_time)
			#print(len(self.processing_times))
	
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
		#ここで, Proposed methodのプログラムを書く (self.)
		if self.measured_pcd is None:
			return
		numpy_model_pcd = np.array(self.model_pcd .points)
		numpy_measured_pcd = np.array(self.measured_pcd.points)
		len_model_pcd = len(numpy_model_pcd)
		len_measured_pcd = len(numpy_measured_pcd)
		
		### 計測点群の密度に合わせる ###
		min_bound = np.min(numpy_measured_pcd, axis=0)
		max_bound = np.max(numpy_measured_pcd, axis=0)

		box_width = max_bound[0] - min_bound[0]
		box_height = max_bound[1] - min_bound[1]
		box_depth = max_bound[2] - min_bound[2]
		###print("ボクセルサイズの大きさ", box_width, box_height, box_depth)
		density = len_measured_pcd / ( box_width * box_height * box_depth )
		###print("密度：", density) 
		if self.target=="t":
			weight = 0.81
		elif self.target=="l":
			weight = 0.8

		voxel_size = weight * (1.0 / density) ** (1/3)

		model_pcd = self.model_pcd.voxel_down_sample(voxel_size) #初期値0.006

		measured_pcd = copy.deepcopy(self.measured_pcd)
		model_pcd.paint_uniform_color([1.0, 0, 0])
		measured_pcd.paint_uniform_color([1.0, 0, 0])
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.masknet_load.to(device)
		template_tensor = torch.tensor(np.array(model_pcd.points), dtype=torch.float32).unsqueeze(0)
		source_tensor = torch.tensor(numpy_measured_pcd, dtype=torch.float32).unsqueeze(0)
		template_tensor = template_tensor.to(device)
		source_tensor = source_tensor.to(device)
		
		# 位置合わせ準備を行う（関数を呼び出す）
		registration_model = Registration.Registration()
		
		self.start_masknet_time = time.time()
		with torch.no_grad():
			masked_template_cheese, predicted_mask_cheese = self.masknet_load(template_tensor, source_tensor)
		self.end_masknet_time = time.time()
		
		masked_pcd = o3d.geometry.PointCloud()
		masked_pcd.points = o3d.utility.Vector3dVector(masked_template_cheese.detach().cpu().numpy()[0])
		# 提案手法（MaskNet、SVD、ICP）の実行（実際のデータを代入）
		result_cheese = registration_model.register(masked_template_cheese, source_tensor, self.target)
		est_T_cheese = result_cheese['est_T']     # est_T：ICPの変換行列
		
		# 姿勢の平行移動と回転を取得
		transform_matrix = est_T_cheese.cpu().numpy()[0]
		translation = transform_matrix[:3, 3]
		rotation_matrix = transform_matrix[:3, :3]
		
		self.start_kalman_time = time.time()
		# 観測データをまとめる (x, y, z, qx, qy, qz, qw)
		quaternion = tf_conversions.transformations.quaternion_from_matrix(transform_matrix)
		z = np.hstack((translation, quaternion))
		
		# 初期化されていない場合、初期状態を設定
		if not self.is_initialized:
			self.kf.x = z  # 初期状態を観測値で設定
			self.is_initialized = True  # 初期化フラグを更新
		else:
			self.kf.predict()  # 予測ステップ
			self.kf.update(z)   # 観測値で更新
			
		# フィルタリング後の位置と姿勢
		filtered_translation = self.kf.x[:3]
		filtered_quaternion = self.kf.x[3:]
		self.end_kalman_time = time.time()
		# フィルタリングされたデータを変換行列に戻す
		filtered_rotation_matrix = tf_conversions.transformations.quaternion_matrix(filtered_quaternion)[:3, :3]
		filtered_transform_matrix = np.eye(4)
		filtered_transform_matrix[:3, :3] = filtered_rotation_matrix
		filtered_transform_matrix[:3, 3] = filtered_translation.T

    		# フィルタリングされた姿勢をROSのトランスフォームとして送信
		self.publish_transform(torch.tensor(filtered_transform_matrix))
		
	
	def publish_transform(self, est_T_cheese):
		if est_T_cheese is None:
			print("est_T_cheese is None")
			return
		transform_msg = TransformStamped()
		
		# ROS Header
		transform_msg.header.stamp = rospy.Time.now()

		transform_msg.header.frame_id = self.child_frame_id
		transform_msg.child_frame_id = "Posture_of_object"  # ソースフレームの名前（適宜変更）
		
		# 変換行列をTransformに変換
		transform_matrix = est_T_cheese.cpu().numpy()
		inverse_transform_matrix = np.linalg.inv(transform_matrix)

		translation = inverse_transform_matrix[:3, 3].reshape(-1)
		#rotation_matrix = inverse_transform_matrix[:3, :3]
		
		# Quaternionに変換
		quaternion = tf_conversions.transformations.quaternion_from_matrix(inverse_transform_matrix)
		
		# TransformStampedメッセージに設定
		transform_msg.transform.translation.x = translation[0]
		transform_msg.transform.translation.y = translation[1]
		transform_msg.transform.translation.z = translation[2]
		transform_msg.transform.rotation.x = quaternion[0]
		transform_msg.transform.rotation.y = quaternion[1]
		transform_msg.transform.rotation.z = quaternion[2]
		transform_msg.transform.rotation.w = quaternion[3]
		# トランスフォームをブロードキャスト
		self.br.sendTransform(transform_msg)


def main():
    processor = PointCloudProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()
