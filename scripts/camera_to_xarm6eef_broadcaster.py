#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import TransformStamped

class CameraToWorldBroadcaster:
    # def broadcast_transform():
    #     rospy.init_node('camera_to_world_broadcaster')
    #     br = tf.TransformBroadcaster()
    #     rate = rospy.Rate(15.0)

    #     while not rospy.is_shutdown():
    #         # カメラの位置と向きを設定（実際の設置に合わせて調整する）
    #         br.sendTransform((0.0673, 0.0, 0.0),  # 平行移動 (x, y, z)
    #                         tf.transformations.quaternion_from_euler(0, -1.57, 3.14),  # 回転 (roll, pitch, yaw)
    #                         rospy.Time.now(),
    #                         "camera_link",  # RealSenseのカメラフレーム名
    #                         "link_eef"  # ロボットのエンドエフェクタフレーム名
    #                         )
    #         rate.sleep()
    def __init__(self):
        rospy.init_node('camera_to_world_broadcaster')
        self.br = tf.TransformBroadcaster()
        self.rate = rospy.Rate(15.0)

    def broadcast_transform(self):
        while not rospy.is_shutdown():
            # カメラの位置と向きを設定（実際の設置に合わせて調整する）
            self.br.sendTransform((0.0673, 0.0, 0.0),  # 平行移動 (x, y, z)
                            tf.transformations.quaternion_from_euler(0, -1.57, 3.14),  # 回転 (roll, pitch, yaw)
                            rospy.Time.now(),
                            "camera_link",  # RealSenseのカメラフレーム名
                            "link_eef"  # ロボットのエンドエフェクタフレーム名
                            )
            self.rate.sleep()


if __name__ == '__main__':
    rospy.loginfo("Start camera_to_world_broadcaster")
    camera_to_world_broadcaster = CameraToWorldBroadcaster()
    camera_to_world_broadcaster.broadcast_transform()
