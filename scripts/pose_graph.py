######################################################
# Time evolution of the pose errors のグラフを描画する
# ISAROB 2025
######################################################

import matplotlib.pyplot as plt
import csv
import numpy as np

# グローバルなフォントサイズ設定
plt.rcParams.update({
    'font.size': 18,       # 全体の基本フォントサイズ
    #'axes.titlesize': 20,  # タイトルのフォントサイズ
    'axes.labelsize': 20,  # 軸ラベルのフォントサイズ
    'legend.fontsize': 20, # 凡例のフォントサイズ
    'xtick.labelsize': 18, # x軸の目盛りラベルのフォントサイズ
    'ytick.labelsize': 18, # y軸の目盛りラベルのフォントサイズ
})

def load_csv(file_path):
    """CSVファイルを読み込み、位置誤差と回転誤差を返す"""
    translation_errors = []
    rotation_errors = []
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # ヘッダー行をスキップ
        
        for row in reader:
            # 位置誤差（m）と回転誤差（rad）を取得
            translation_errors.append(float(row[0]))
            rotation_errors.append(float(row[1]))
    
    return translation_errors, rotation_errors

def limit_samples(data, max_samples=100):
    """データを最大サンプル数まで制限"""
    return data[:max_samples]

def plot_translation_error(file1_data, file2_data, file3_data, max_samples=200):
    """位置誤差をプロットして保存"""
    time_steps = list(range(max_samples))
    file1_trans = limit_samples(file1_data[0], max_samples)
    file2_trans = limit_samples(file2_data[0], max_samples)
    file3_trans = limit_samples(file3_data[0], max_samples)

    translation_ylim = 0.01  # 位置誤差の縦軸の上限

    plt.figure(figsize=(10, 6))
    clipped_trans = np.clip(file1_trans, 0, translation_ylim)
    plt.plot(time_steps, clipped_trans, label="Fast global Registration", color='r', linestyle='--')
    plt.plot(time_steps, file2_trans, label="Conventional method", color='g')
    plt.plot(time_steps, file3_trans, label="Proposed method", color='b')
    # plt.title("Translation Error")
    plt.xlabel("Sample")
    plt.ylabel("Translation Error (m)")
    plt.ylim(0, translation_ylim)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("translation_error.png")
    plt.show()
    print("Translation error plot saved as 'translation_error.png'")

def plot_rotation_error(file1_data, file2_data, file3_data, max_samples=200):
    """回転誤差をプロットして保存"""
    time_steps = list(range(max_samples))
    file1_rot = limit_samples(file1_data[1], max_samples)
    file2_rot = limit_samples(file2_data[1], max_samples)
    file3_rot = limit_samples(file3_data[1], max_samples)

    rotation_ylim = 0.3  # 回転誤差の縦軸の上限

    plt.figure(figsize=(10, 6))
    clipped_rot = np.clip(file1_rot, 0, rotation_ylim)
    plt.plot(time_steps, clipped_rot, label="Fast global Registration", color='r', linestyle='--')
    plt.plot(time_steps, file2_rot, label="Conventional method", color='g')
    plt.plot(time_steps, file3_rot, label="Proposed method", color='b')
    # plt.title("Rotation Error")
    plt.xlabel("Sample")
    plt.ylabel("Rotation Error (rad)")
    plt.ylim(0, rotation_ylim)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("rotation_error.png")
    plt.show()
    print("Rotation error plot saved as 'rotation_error.png'")

def main():
    ##########################################
    # T joint pipe の場合は「csv_T/ ~」
    # L joint pipe の場合は「csv_L/ ~」
    ##########################################
    # 3つのCSVファイルのパスを指定
    file1 = 'csv_T/fgr_pose_errors.csv'
    file2 = 'csv_T/con_pose_errors.csv'
    file3 = 'csv_T/pro_pose_errors.csv'

    # 各ファイルからデータを読み込む
    file1_data = load_csv(file1)
    file2_data = load_csv(file2)
    file3_data = load_csv(file3)

    # それぞれの誤差グラフを描画して保存
    plot_translation_error(file1_data, file2_data, file3_data)
    plot_rotation_error(file1_data, file2_data, file3_data)

if __name__ == '__main__':
    main()
