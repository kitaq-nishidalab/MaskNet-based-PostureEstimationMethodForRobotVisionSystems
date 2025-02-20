import open3d as o3d

# Load the PCD file
#pcd = o3d.io.read_point_cloud("TNUTEJN016_10000.pcd")
pcd = o3d.io.read_point_cloud("WMU2LR2020_10000.pcd")

downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.0025)

# Save as PLY file
#o3d.io.write_point_cloud("TNUTEJN016_10000_down.ply", downsampled_pcd)
o3d.io.write_point_cloud("WMU2LR2020_10000_down.ply", downsampled_pcd)

print("Conversion complete!")
