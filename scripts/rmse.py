###################################################
#MaskNet(raycasting learned)+SVD+kalman filter
#姿勢推定の精度を評価するスクリプト
# ISAROB 2025
###################################################
import rospy
import numpy as np
from tf2_msgs.msg import TFMessage
import tf_conversions
import tf2_ros

class PoseEstimator:
    def __init__(self):
        self.pose_list = []
        self.ground_truth_pose = None  # 正解の姿勢を格納
        self.max_samples = 1000

        self.position_rmse = 0.0  # 位置RMSEの初期化
        self.orientation_rmse = 0.0  # 回転RMSEの初期化

        # tf_staticとtfのサブスクライバーの設定
        self.sub_static_pose = rospy.Subscriber('/tf_static', TFMessage, self.static_pose_callback)
        self.sub_tf = rospy.Subscriber('/tf', TFMessage, self.tf_callback)

        # TF2ブロードキャスターの初期化
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.static_pose = None  # 静的姿勢を格納
        self.prev_static_angles = None  # 静的姿勢のオイラー角補正用
        self.prev_ground_truth_angles = None  # 正解姿勢のオイラー角補正用

    def static_pose_callback(self, msg):
        for transform in msg.transforms:
            # "Posture_of_object"フレームの姿勢を取得
            if transform.child_frame_id == "Posture_of_object":
                self.store_static_pose(transform)

    def tf_callback(self, msg):
        for transform in msg.transforms:
            # "PostureGT+Initial"フレームの正解姿勢を取得
            if transform.child_frame_id == "GT_posture":
                self.store_ground_truth_pose(transform)

        # 必要サンプル数に達したら統計情報とRMSEを計算
        if len(self.pose_list) >= self.max_samples:
            self.calculate_statistics()

    def store_static_pose(self, transform):
        # クオータニオンをそのまま保存
        quat = (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        )

        # 位置をそのまま保存
        position = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])

        self.static_pose = np.concatenate([position, quat])  # 位置とクオータニオンを一緒に保存
        self.pose_list.append(self.static_pose)

    def store_ground_truth_pose(self, transform):
        # クオータニオンをそのまま保存
        quat = (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        )

        # 位置をそのまま保存
        position = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])

        self.ground_truth_pose = np.concatenate([position, quat])  # 位置とクオータニオンを一緒に保存
    def calculate_statistics(self):
        # RMSEリストを初期化
        position_errors = []
        orientation_errors = []
        
        # 各推定姿勢に対してRMSEを計算
        for pose in self.pose_list:
        	# 位置誤差の計算 (x, y, z)
            position_error = pose[:3] - self.ground_truth_pose[:3]
            position_errors.append(np.linalg.norm(position_error))

        	# クオータニオン誤差の計算
            q_static = pose[3:]  # クオータニオン
            q_gt = self.ground_truth_pose[3:]  # 正解のクオータニオン
        
        	# クオータニオン間の誤差を計算（最短回転を考慮）
            dot_product = np.dot(q_static, q_gt)
            if dot_product < 0:
                q_gt = -q_gt  # 逆転を考慮
                dot_product = -dot_product
        
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_error = 2 * np.arccos(dot_product)
        
        	# 180度（πラジアン）を超える場合に反転
            if angle_error > np.pi:
            		angle_error = 2 * np.pi - angle_error
                    
            orientation_errors.append(angle_error)
            
        # 平均RMSEを計算
        self.position_rmse = np.mean(position_errors) if position_errors else 0.0
        self.orientation_rmse = np.mean(orientation_errors) if orientation_errors else 0.0

        # 結果をログに表示
        rospy.loginfo(f"Position RMSE: {self.position_rmse}")
        rospy.loginfo(f"Orientation RMSE (rad): {self.orientation_rmse}")

        # ポーズリストをクリアして次のサンプルを収集
        self.pose_list.clear()

    	

def main():
    rospy.init_node('pose_estimator', anonymous=True)
    estimator = PoseEstimator()
    rospy.spin()

if __name__ == '__main__':
    main()
