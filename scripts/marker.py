import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker

class PointCloudZMarker:
    def __init__(self):
        rospy.init_node("point_cloud_z_marker", anonymous=True)
        
        # ポイントクラウドを購読
        self.point_cloud_sub = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.point_cloud_callback)
        # マーカーを発行
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)

    def point_cloud_callback(self, msg):
        # PointCloud2からポイントを取得
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points_np = np.array(points)

        # x=0, y=0 に最も近いポイントを探す
        distances = np.linalg.norm(points_np[:, :2], axis=1)  # x, y 座標から距離を計算
        closest_idx = np.argmin(distances)  # 距離が最小のインデックス
        closest_point = points_np[closest_idx]

        x_value, y_value, z_value = closest_point  # 最近傍点のx, y, z座標を取得

        # マーカーを更新
        self.publish_marker(x_value, y_value, z_value, marker_id=0)  # 原点のマーカー
        self.publish_marker(x_value + 0.13, y_value - 0.09, z_value - 0.0057, marker_id=1)  # x + 9cm, y - 13cm
        self.publish_marker(x_value - 0.13, y_value + 0.09, z_value + 0.0057, marker_id=2)  # x - 9cm, y + 13cm

    def publish_marker(self, x, y, z, marker_id):
        marker = Marker()
        marker.header.frame_id = "camera_depth_optical_frame"  # 座標フレームに合わせる
        marker.header.stamp = rospy.Time.now()

        # マーカーの基本設定
        marker.ns = "z_marker"
        marker.id = marker_id
        marker.type = Marker.SPHERE  # 球体のマーカー
        marker.action = Marker.ADD

        # マーカーの位置
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # サイズ (0.5cmの球体)
        marker.scale.x = 0.0075
        marker.scale.y = 0.0075
        marker.scale.z = 0.0075

        # 色の設定 (異なる色を設定する例)
        if marker_id == 0:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0 
        elif marker_id == 1:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0  
        elif marker_id == 2:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0  

        marker.color.a = 1.0  # 不透明

        # マーカーをパブリッシュ
        self.marker_pub.publish(marker)
        rospy.loginfo(f"Published marker {marker_id} at ({x}, {y}, {z}).")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        z_marker = PointCloudZMarker()
        z_marker.run()
    except rospy.ROSInterruptException:
        pass


