import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import open3d as o3d
from glob import glob
import argparse
from tqdm import tqdm
import network  # MaskNet のネットワーク定義を含むスクリプト
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser("MaskNet training", add_help=True)
parser.add_argument("--save_path", "-s", type=str, default="checkpoint/model_epoch500_upsampling.pth", help="Path to save the trained model")
#parser.add_argument("--base_dir", "-b", type=str, default="/home/nishidalab0/vision_ws_blender/output_ModelNet40", help="Base directory for the dataset")
parser.add_argument("--epoch", type=int, default=300, help="Number of epochs")
parser.add_argument("--batch", type=int, default=8, help="Batch size")
parser.add_argument("--num_points", type=int, default=1024, help="Number of points per point cloud")
args = parser.parse_args()

def custom_collate_fn(batch):
    templates, sources, gt_masks = zip(*batch)

    # 最大点数を取得
    max_points = max([max(template.size(0), source.size(0)) for template, source in zip(templates, sources)])

    # テンソルのパディング
    padded_templates = []
    padded_sources = []
    padded_gt_masks = []

    for template, source, gt_mask in zip(templates, sources, gt_masks):
        # テンプレートのパディング
        if template.size(0) < max_points:
            pad_size = max_points - template.size(0)
            padded_template = torch.cat([template, torch.zeros(pad_size, template.size(1))], dim=0)
        else:
            padded_template = template  # そのまま使用
        padded_templates.append(padded_template)

        # ソース点群のパディング
        if source.size(0) < max_points:
            pad_size = max_points - source.size(0)
            padded_source = torch.cat([source, torch.zeros(pad_size, source.size(1))], dim=0)
        else:
            padded_source = source  # そのまま使用
        padded_sources.append(padded_source)

        # マスクのパディング
        if gt_mask.size(0) < max_points:
            pad_size = max_points - gt_mask.size(0)
            padded_gt_mask = torch.cat([gt_mask, torch.zeros(pad_size)], dim=0)
        else:
            padded_gt_mask = gt_mask  # そのまま使用
        padded_gt_masks.append(padded_gt_mask)

    # バッチとしてまとめる
    return torch.stack(padded_templates), torch.stack(padded_sources), torch.stack(padded_gt_masks)



class MultiObjectRegistrationData(Dataset):
    def __init__(self, base_dir, num_points=1024):
        self.base_dir = base_dir
        self.num_points = num_points

        # データペア (元のPLY, フィルタ後のPLY, マスク) を収集
        self.data_pairs = []  # (original_ply, filtered_ply, gt_mask)
        for obj_dir in os.listdir(base_dir):
            ply_dir = os.path.join(base_dir, obj_dir, "ply")
            filterply_dir = os.path.join(base_dir, obj_dir, "filterply")
            gtmask_dir = os.path.join(base_dir, obj_dir, "gtmask")
            
            if not (os.path.isdir(ply_dir) and os.path.isdir(filterply_dir) and os.path.isdir(gtmask_dir)):
                continue  # 必須フォルダがない場合はスキップ

            # 元のPLY, フィルタ後のPLY, マスクを対応付け
            for ply_file in sorted(glob(os.path.join(ply_dir, "*.ply"))):
                base_name = os.path.splitext(os.path.basename(ply_file))[0]
                filtered_ply_file = os.path.join(filterply_dir, f"{base_name}_filtered.ply")
                mask_file = os.path.join(gtmask_dir, f"{base_name}_gtmask.npy")
                
                if os.path.exists(filtered_ply_file) and os.path.exists(mask_file):
                    # フィルタ後の点群を読み込み
                    original_ply = o3d.io.read_point_cloud(ply_file)
                    filtered_ply = o3d.io.read_point_cloud(filtered_ply_file)
                    original_points = np.asarray(original_ply.points)
                    filtered_points = np.asarray(filtered_ply.points)
                    
                    # フィルタ後の点数が10未満の場合はスキップ
                    if len(filtered_points) < 5 or len(original_points) == len(filtered_points):
                        #print(f"Skipped {filtered_ply_file}: only {len(filtered_points)} points.")
                        continue
                    
                    # データペアに追加
                    self.data_pairs.append((ply_file, filtered_ply_file, mask_file))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        original_ply_path, filtered_ply_path, mask_path = self.data_pairs[index]
        
        # 元の点群データを読み込む
        original_ply = o3d.io.read_point_cloud(original_ply_path)
        original_points = np.asarray(original_ply.points)

        # フィルタ後の点群データを読み込む
        filtered_ply = o3d.io.read_point_cloud(filtered_ply_path)
        filtered_points = np.asarray(filtered_ply.points)

        # マスクデータを読み込む
        gt_mask = np.load(mask_path)
        
        # 必要に応じて点数を調整
        if len(original_points) > self.num_points:
            idx = np.random.choice(len(original_points), self.num_points, replace=False)
            original_points = original_points[idx]
            # マスクとフィルタ後の点群も同じインデックスでサンプリング
            gt_mask = gt_mask[idx]
            filtered_points = filtered_points[idx]
        else:
            pad_size = self.num_points - len(original_points)
            original_points = np.pad(original_points, ((0, pad_size), (0, 0)), mode='constant')
            gt_mask = np.pad(gt_mask, (0, pad_size), mode='constant')
            filtered_points = np.pad(filtered_points, ((0, pad_size), (0, 0)), mode='constant')
        
        # Tensor に変換
        original_points = torch.tensor(original_points, dtype=torch.float32)
        filtered_points = torch.tensor(filtered_points, dtype=torch.float32)
        gt_mask = torch.tensor(gt_mask, dtype=torch.float32)
        
        return original_points, filtered_points, gt_mask
        
###################################
#-- ネットワークの定義 --
###################################
model = network.MaskNet()  # MaskNetのインスタンスを生成
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
###################################
#-- 誤差関数とオプティマイザ --
###################################
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


###################################
#-- データセットの準備 --
###################################
base_dir = "/home/nishidalab0/vision_ws_blender/output_ModelNet40_upsampling"
dataset = MultiObjectRegistrationData(base_dir=base_dir)
print(f"データ数: {len(dataset)}")

original, filtered, mask = dataset[0]
print(f"元の点群形状: {original.shape}, フィルタ後点群形状: {filtered.shape}, マスク形状: {mask.shape}")

train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=custom_collate_fn)


"""
pcd_original = o3d.geometry.PointCloud()
pcd_original.points = o3d.utility.Vector3dVector(original.to('cpu').detach().numpy())
pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(filtered.to('cpu').detach().numpy())
pcd_original.paint_uniform_color([1, 0, 0])
pcd_filtered.paint_uniform_color([0, 1, 0])
# グラウンドトゥルースマスクを適用したテンプレート点群の切り出し
masked_points = original[mask.bool()]
pcd_masked_points = o3d.geometry.PointCloud()
pcd_masked_points.points = o3d.utility.Vector3dVector(masked_points.to('cpu').detach().numpy())
pcd_masked_points.paint_uniform_color([0, 0, 1])
print(np.shape(np.array(pcd_masked_points.points)))

o3d.visualization.draw_geometries([pcd_original, pcd_masked_points, pcd_filtered])
"""

###################################
#-- ネットワークの学習 --
###################################
epochs = args.epoch

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    train_loss = 0.0
    count = 0

    for i, (template, source, gt_mask) in enumerate(tqdm(train_loader)):
        template = template.to(device)
        source = source.to(device)
        gt_mask = gt_mask.to(device)

        optimizer.zero_grad()

        # モデルのフォワードパス
        masked_template, predicted_mask = model(template, source)

        # 損失計算
        loss_mask = criterion(predicted_mask, gt_mask)
        loss_mask.backward()
        optimizer.step()

        train_loss += loss_mask.item()
        count += 1

        avg_loss = train_loss / count
    print(f"Training Loss: {avg_loss:.6f}")

###################################
#-- ネットワークの保存 --
###################################
save_path = args.save_path
torch.save(model, save_path)
print(f"Model saved to {save_path}")



