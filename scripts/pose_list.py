############################################################
# Time evolution of the pose errors のデータをCSVファイルに保存する
# ISAROB 2025 
############################################################
import rospy
import numpy as np
import tf_conversions
import csv
from tf2_msgs.msg import TFMessage

class PoseEstimator:
    def __init__(self):
        self.pose_list = []
        self.max_samples = 200
        self.sub = rospy.Subscriber('/tf_static', TFMessage, self.pose_callback)
        self.previous_pose = None  # 前回のサンプルを保持する変数

    def pose_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == "Posture_of_object":
                # 1. クォータニオンを取得
                quat = (
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                )

                # 2. 現在の平行移動を取得
                current_pose = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    quat[0],  # 回転クォータニオンのx
                    quat[1],  # 回転クォータニオンのy
                    quat[2],  # 回転クォータニオンのz
                    quat[3]   # 回転クォータニオンのw
                ])

                # 前回のサンプルが存在する場合のみ差分を計算
                if self.previous_pose is not None:
                    # 位置誤差（絶対差分の合計）
                    delta_translation = np.abs(current_pose[:3] - self.previous_pose[:3])  # 平行移動の絶対差分
                    translation_error = np.sum(delta_translation)  # 合計誤差

                    # 回転誤差（クォータニオン間の角度誤差）
                    angle_error = self.quaternion_angle_difference(self.previous_pose[3:], current_pose[3:])
                    
                    # 誤差をリストに追加
                    self.pose_list.append([translation_error, angle_error])

                    if len(self.pose_list) >= self.max_samples:
                        self.save_to_csv()

                # 現在のサンプルを前回のサンプルとして更新
                self.previous_pose = current_pose
                
    def quaternion_angle_difference(self, q1, q2):
        """クォータニオン間の回転誤差を計算（最短経路）"""
        # 2つのクォータニオンから角度差を計算
        dot_product = np.dot(q1, q2)
        
        # 逆転した回転が含まれる場合は反転
        if dot_product < 0:
            q2 = -q2
            dot_product = -dot_product
        
        # dot productが1の場合（回転差がない場合）
        if dot_product > 1.0:
            dot_product = 1.0
        elif dot_product < -1.0:
            dot_product = -1.0
        
        # 回転角度（ラジアン）を計算
        angle = 2 * np.arccos(dot_product)
        
        # 180度（πラジアン）を超える回転があった場合、反転させる
        if angle > np.pi:
            angle = 2 * np.pi - angle
        
        return angle

    def save_to_csv(self):
        # CSVファイルにデータを保存
        if len(self.pose_list) < self.max_samples:
            rospy.logwarn("Insufficient samples to save to CSV.")
            return
        
        #####################
        #T joint pipe の場合は「csv_T/ ~」
        #L joint pipe の場合は「csv_L/ ~」
        # CSVファイルに保存
        #####################
        with open('csv_T/pose_errors.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Translation Error (m)', 'Rotation Error (rad)'])  # ヘッダー
            writer.writerows(self.pose_list)

        rospy.loginfo("Pose error data saved to 'pose_errors.csv'.")

        # サンプルリストをクリア
        self.pose_list.clear()

def main():
    rospy.init_node('pose_estimator', anonymous=True)
    pose_estimator = PoseEstimator()
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Pose estimator node terminated.")

if __name__ == '__main__':
    main()
