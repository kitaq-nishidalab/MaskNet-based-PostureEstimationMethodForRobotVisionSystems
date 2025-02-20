#######################################################
#MaskNet(raycasting learned)+SVD+kalman filter
#SD(Standerd Deviation)を評価するプログラム
# ISAROB 2025
######################################################
import rospy
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

class PoseEstimator:
    def __init__(self):
        self.pose_list = []
        self.max_samples = 1000
        self.sub = rospy.Subscriber('/tf_static', TFMessage, self.pose_callback)

    def pose_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == "Posture_of_object":
                # 平行移動を取得
                translation = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])

                # クォータニオンを取得し正規化
                quat = np.array([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ])
                quat = quat / np.linalg.norm(quat)  # 正規化

                # 平行移動とクォータニオンをリストに追加
                self.pose_list.append((translation, quat))

                if len(self.pose_list) >= self.max_samples:
                    self.calculate_statistics()

    def calculate_statistics(self):
        # 平行移動とクォータニオンを分離
        translations = np.array([pose[0] for pose in self.pose_list])
        quaternions = np.array([pose[1] for pose in self.pose_list])

        # 平行移動の平均と標準偏差を計算
        mean_translation = np.mean(translations, axis=0)
        std_dev_translation = np.std(translations, axis=0)

        # クォータニオンの平均を計算
        mean_quaternion = self.average_quaternions(quaternions)

        # クォータニオン間の角度誤差を計算
        quaternion_differences = [
            self.quaternion_angle_error(mean_quaternion, q) for q in quaternions
        ]
        quaternion_angle_std_dev = np.std(quaternion_differences)

        # 結果をログに表示
        # rospy.loginfo(f"Mean Translation (x, y, z): {mean_translation}")
        rospy.loginfo(f"Standard Deviation (Translation): {sum(std_dev_translation)}")
        # rospy.loginfo(f"Mean Quaternion: {mean_quaternion}")
        rospy.loginfo(f"Quaternion Angular Error Std Dev: {quaternion_angle_std_dev}")

        # ポーズリストをクリアして次のサンプルを収集
        self.pose_list.clear()

    @staticmethod
    def average_quaternions(quaternions):
        """
        クォータニオンの平均を計算
        """
        q_matrix = np.array(quaternions).T  # 各クォータニオンを列ベクトルにする
        A = q_matrix @ q_matrix.T  # 内積行列（Gram行列）

        # 主成分（最大固有値に対応する固有ベクトル）を計算
        eigenvalues, eigenvectors = np.linalg.eig(A)
        max_index = np.argmax(eigenvalues)
        return eigenvectors[:, max_index]

    @staticmethod
    def quaternion_angle_error(q1, q2):
        """
        2つのクォータニオン間の角度誤差を計算。
        """
        dot_product = np.dot(q1, q2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # 数値誤差対策
        return 2 * np.arccos(dot_product)

def main():
    rospy.init_node('Var', anonymous=True)
    estimator = PoseEstimator()
    rospy.spin()

if __name__ == '__main__':
    main()
