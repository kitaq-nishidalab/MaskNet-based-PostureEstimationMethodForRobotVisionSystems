import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
import re
# カスタムソート関数
def sort_key(filename):
    camera_match = re.search(r'Camera(\d+)_Run(\d+)', filename)
    if camera_match:
        camera_num = int(camera_match.group(1))
        run_num = int(camera_match.group(2))
        return camera_num, run_num
    return float('inf'), float('inf')  # マッチしない場合、リストの最後に移動

class RegistrationData(Dataset):
    def __init__(self, template_path, source_dir, gt_mask_path, num_points=1024):
        self.template_path = template_path
        self.source_dir = source_dir
        self.gt_mask_path = gt_mask_path
        self.num_points = num_points

        # グラウンドトゥルースマスクを読み込む
        self.gt_masks = np.load(gt_mask_path)

        # ソースディレクトリ内の全てのPLYファイルをリストする
        self.source_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.ply')],key=sort_key)
        #print(self.source_files)  # ディレクトリ内のファイルを確認

        # 一貫性のチェック
        if len(self.source_files) != len(self.gt_masks):
            raise ValueError("ソースファイルとグラウンドトゥルースマスクの数が一致しません。")

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, index):
        # テンプレート点群を読み込む
        template = o3d.io.read_point_cloud(self.template_path)
        template_points = np.asarray(template.points)
        """
        if len(template_points) > self.num_points:
            np.random.shuffle(template_points)
            template_points = template_points[:self.num_points]
        else:
            template_points = np.pad(template_points, ((0, self.num_points - len(template_points)), (0, 0)), mode='constant')
        """
        # ソース点群を読み込む
        source_file = os.path.join(self.source_dir, self.source_files[index])
        source = o3d.io.read_point_cloud(source_file)
        source_points = np.asarray(source.points)
        """
        if len(source_points) > self.num_points:
            np.random.shuffle(source_points)
            source_points = source_points[:self.num_points]
        else:
            source_points = np.pad(source_points, ((0, self.num_points - len(source_points)), (0, 0)), mode='constant')
        """
        # グラウンドトゥルースマスクを読み込む
        gt_mask = self.gt_masks[index]

        # PyTorchテンソルに変換
        template_points = torch.from_numpy(template_points).float()
        source_points = torch.from_numpy(source_points).float()
        gt_mask = torch.from_numpy(gt_mask).float()

        return template_points, source_points, gt_mask
    
if __name__=="__main__":
    index = 160

    template_path = "C:/Users/shuny/catkin_ws/src/MaskSVD/preprocess/TNUTEJN016_1024.ply"
    source_dir = "C:/Blender/M_study/output/Tpipe/FilterMasks"
    gt_mask_path = "C:/Blender/M_study/output/Tpipe/gtmask/masks.npy"

    #print(os.path.exists(template_path))  # ファイルが存在するか確認
    
    trainset = RegistrationData(template_path=template_path, source_dir=source_dir, gt_mask_path=gt_mask_path)
    
    template_sample, source_sample, gt_mask_sample = trainset.__getitem__(index)
    print(np.shape(template_sample.to('cpu').detach().numpy().copy()))
    print(np.shape(source_sample.to('cpu').detach().numpy().copy()))
    #print(np.shape(gt_mask_sample))

    pcd_template_sample = o3d.geometry.PointCloud()
    pcd_template_sample.points = o3d.utility.Vector3dVector(template_sample.to('cpu').detach().numpy())
    pcd_source_sample = o3d.geometry.PointCloud()
    pcd_source_sample.points = o3d.utility.Vector3dVector(source_sample.to('cpu').detach().numpy())
    pcd_template_sample.paint_uniform_color([1, 0, 0])
    pcd_source_sample.paint_uniform_color([0, 1, 0])
     # グラウンドトゥルースマスクを適用したテンプレート点群の切り出し
    masked_template_points = template_sample[gt_mask_sample.bool()]
    pcd_masked_template_sample = o3d.geometry.PointCloud()
    pcd_masked_template_sample.points = o3d.utility.Vector3dVector(masked_template_points.to('cpu').detach().numpy())
    pcd_masked_template_sample.paint_uniform_color([0, 0, 1])
    print(np.shape(np.array(pcd_masked_template_sample.points)))

    o3d.visualization.draw_geometries([pcd_masked_template_sample, pcd_source_sample])