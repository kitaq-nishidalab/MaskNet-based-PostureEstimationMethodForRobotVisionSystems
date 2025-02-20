import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d
import numpy as np
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped

class CADModelPublisher:
    def __init__(self):
        rospy.init_node('cad_model_publisher', anonymous=True)
        self.pub = rospy.Publisher('/transformed_point_cloud', PointCloud2, queue_size=10000)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()  # Use TransformBroadcaster for dynamic transforms

    def load_cad_model(self, filepath):
        """Load the CAD model."""
        cad_model = o3d.io.read_point_cloud(filepath)
        return cad_model

    def compute_translation(self, cad_model, target_min_xy, target_max_z):
        """Calculate the translation needed to align x, y min and z max to target values."""
        # Define rotation angles
        angle_rad_z = np.deg2rad(180)
        angle_rad_y = np.deg2rad(2)

        # Rotation matrices
        rotation_matrix_z = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad_z])
        rotation_matrix_y = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle_rad_y, 0])

        # Apply rotations to the model
        cad_model.rotate(rotation_matrix_z, center=(0, 0, 0))
        cad_model.rotate(rotation_matrix_y, center=(0, 0, 0))

        points = np.asarray(cad_model.points)
        
        # Calculate the current min and max values
        current_min_xy = points.min(axis=0)[:2]  # x, y の最小値
        current_max_z = points.max(axis=0)[2]   # z の最大値

        translation = np.zeros(3)
        translation[:2] = target_min_xy - current_min_xy  # x, y の最小値を合わせる
        translation[2] = target_max_z - current_max_z     # z の最大値を合わせる

        return translation, rotation_matrix_z, rotation_matrix_y

    def apply_translation(self, cad_model, translation, rotation_matrix_z, rotation_matrix_y):
        """Apply translation and rotation to the point cloud."""
        # Apply translation
        cad_model.translate(translation)
        
        # Get the final transformation matrix (translation + rotation)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix_y @ rotation_matrix_z
        transformation_matrix[:3, 3] = translation
        
        return cad_model, transformation_matrix

    def publish_point_cloud(self, cad_model):
        """Convert Open3D point cloud to ROS PointCloud2 and publish it."""
        points = np.asarray(cad_model.points)
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_depth_optical_frame"

        # Convert to PointCloud2
        cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
    
        # Publish
        self.pub.publish(cloud_msg)

    def publish_tf(self, transformation_matrix):
        """Publish the transformation as a TF."""
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "camera_depth_optical_frame"  # Parent frame
        transform.child_frame_id = "GT_posture"  # Child frame

        # Extract translation from the transformation matrix
        translation = transformation_matrix[:3, 3]
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        
        # Get the rotation matrix (top-left 3x3 block) and convert to quaternion
        rotation_matrix = transformation_matrix[:3, :3]
        
        # Create 4x4 matrix by adding a row [0, 0, 0, 1]
        full_rotation_matrix = np.eye(4)
        full_rotation_matrix[:3, :3] = rotation_matrix
        
        # Convert to quaternion
        quaternion = tf_conversions.transformations.quaternion_from_matrix(full_rotation_matrix)

        # Set the quaternion to the transform
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        
        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

    def run(self, input_filepath, target_min_xy, target_max_z):
        # Load the CAD model
        cad_model = self.load_cad_model(input_filepath)

        # Compute translation and rotation
        translation, rotation_matrix_z, rotation_matrix_y = self.compute_translation(cad_model, target_min_xy, target_max_z)
        
        # Apply translation and rotation to the model
        transformed_model, transformation_matrix = self.apply_translation(cad_model, translation, rotation_matrix_z, rotation_matrix_y)

        # Publish the transformed point cloud and TF periodically
        rate = rospy.Rate(10)  # 10Hz for periodic publishing
        while not rospy.is_shutdown():
            #self.publish_point_cloud(transformed_model)
            self.publish_tf(transformation_matrix)
            rate.sleep()  # Sleep to maintain the publishing rate


if __name__ == '__main__':
    try:
        # File path to the CAD model
        #input_cad_filepath ="TNUTEJN016_10000.pcd"
        input_cad_filepath = "WMU2LR2020_10000.pcd"

        # Target bounds
        target_min_xy = np.array([-0.04, -0.03])  # x, y の最小値
        target_max_z = 0.393                      # z の最大値

        # Run the CAD model publisher
        publisher = CADModelPublisher()
        publisher.run(input_cad_filepath, target_min_xy, target_max_z)
        
    except rospy.ROSInterruptException:
        pass

