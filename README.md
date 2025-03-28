#  MaskNet-based Posture Estimation Method for Robot Vision Systems

以下の研究において，ロボットビジョンシステムのための姿勢推定手法を開発しました．

姿勢推定とは，LiDARやステレオカメラのような三次元計測センサにより計測された三次元的な位置座標を基に，センサに映ったオブジェクトの位置と向きを推定する手法です．ロボットは推定したオブジェクトの位置と向きを認識し，オブジェクトの把持姿勢を決定します．

In the following research, we developed a pose estimation method for a robotic vision system.

Pose estimation is a technique that estimates the position and orientation of an object based on three-dimensional coordinates measured by 3D sensing devices such as LiDAR or stereo cameras. The robot recognizes the estimated position and orientation of the object and determines the appropriate grasping pose.

Paper: https://www.researchgate.net/publication/388860828_Integration_of_MaskNet-Based_Posture_Estimation_into_Robot_Vision_Systems

<p align="center">
      <img src="https://github.com/user-attachments/assets/66977d8f-ea21-4f9d-96a6-0030557f2d76" width: 100% height: auto >
</p>

# 実行環境の準備（Setup of the Execution Environment）

### 本パッケージで使用する実機（The actual hardware used in this package）

実機（ロボット）はUFACTORYのxArm6を使用しています。（https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf）
実機は下図のように計算機と接続しています。

The actual robot used in this setup is the xArm6 from UFACTORY.
The robot is connected to the computer as shown in the diagram below.

<p align="center">
      <img src="https://github.com/user-attachments/assets/6df9336d-0786-4401-8cc2-c790c5523522" width: 50% height: auto >
</p>

### Ubuntu環境の用意（Packages within the workspace）

本パッケージはUbuntu20.04のOS上で動作します．

This package runs on Ubuntu 20.04 OS.

### Robot Operating System（ROS）のインストール（Installing ROS）

ROS Noeticをインストールします．
（参考：https://wiki.ros.org/noetic/Installation/Ubuntu）

ROS Noetic is installed.
(Reference:https://wiki.ros.org/noetic/Installation/Ubuntu)


### ワークスペース内のパッケージ（Packages within the workspace）

計算機上の環境として、必要となるパッケージは以下の4つです。

1. MaskSVD (このサイトのパッケージ)
2. xarm_ros (https://github.com/xArm-Developer/xarm_ros)
3. realsense-ros (https://github.com/IntelRealSense/realsense-ros)
4. xarm6_pick_and_place_pkg (https://github.com/Iwaiy/xarm6_pick_and_place_pkg/tree/iwai/devel/new_vision)

本環境では、ワークスペースの中のパッケージが以下のような構造になっていることを想定しています(上記のパッケージ以外はなくても良い)。

The required packages for this environment are as follows:

1. MaskSVD (This site's package)
2. xarm_ros (https://github.com/xArm-Developer/xarm_ros)
3. realsense-ros (https://github.com/IntelRealSense/realsense-ros)
4. xarm6_pick_and_place_pkg (https://github.com/Iwaiy/xarm6_pick_and_place_pkg/tree/iwai/devel/new_vision)

This environment assumes that the workspace follows the structure below, with only the packages listed above being necessary.

<div>
      <img src="https://github.com/user-attachments/assets/9aa88dec-7590-442f-8d00-f607eeaf0f00" width="300" height="400">
</div>

### ロボットの姿勢キャリブレーション（Calibration）

以下のサイトを参考にしてロボットのキャリブレーションを行ってください。

Please perform the robot calibration by referring to the following site.

https://www.notion.so/1ae4a31c2634801285b5d7b515f05091

# 実行手順（Execution steps）


1. 実機の xArm6 を 192.168.1.195 で接続し、RealSenseカメラ (D435i) も立ち上げ
2. MoveIt!を使ったロボットアームのプランニング機能を設定
3. エンドエフェクタの座標系に対してカメラの座標系を正しく登録するプログラムを起動
4. RealSenseカメラ (D435i) から得た点群の前処理（ノイズ除去、ダウンサンプリング、背景除去 など）を行うプログラムを起動
5. 前処理を行った点群に対して３Ｄオブジェクトの姿勢を推定するプログラムを起動

上記のことを実行するために以下のコマンドが必要になります（ターミナル1）。

To execute the following tasks, 

1. Connect the xArm6 to 192.168.1.195 and launch the RealSense D435i camera.
2. Configure the planning functionality for the robotic arm using MoveIt!.
3. Launch a program to register the camera's coordinate system relative to the end effector's coordinate system.
4. Start a program to preprocess point cloud data obtained from the RealSense D435i (e.g., noise removal, downsampling, background removal).
5. Launch a program to estimate the 3D object's pose based on the preprocessed point cloud.

the following command is required(Terminal 1):

```bash
cd ~/<work_space>
source devel/setup.bash
roslaunch xarm6_pick_and_place_pkg task.launch pipeline:=ompl
```

上記のコマンドを実行すると以下のようになります。

When the above command is executed, it results in the following.

<div style="display: flex; gap: 10px;">
    <img src="https://github.com/user-attachments/assets/23aa2a8f-dee3-4607-8b37-e40d105f628f" alt="Image 1" width="300">
    <img src="https://github.com/user-attachments/assets/25eec3d1-b905-4baa-8757-d23228e5f8f8" alt="Image 2" width="300">
</div>

もう一つターミナルを開いて以下のコマンドを実行します(ターミナル2)。

Open another terminal and execute the following command (Terminal 2). 

```bash
cd ~/<work_space>
roslaunch realsense2_camera rs_camera.launch enable_pointcloud:=true
```

上記のコマンドを実行すると以下のようになります。

When the above command is executed, it results in the following.

<div style="display: flex; gap: 10px;">
    <img src="https://github.com/user-attachments/assets/780fd3d5-f920-4616-8485-043151d0bcd1" alt="Image 1" width="300">
    <img src="https://github.com/user-attachments/assets/261ff09a-5886-44c3-82bf-0d36f127981d" alt="Image 2" width="300">
</div>

# 使い方（Usage）

本プログラムを用いた実験では、以下の動画のようにリアルタイムなオブジェクトの姿勢推定からロボットによるオブジェクトの把持、操作まで行うことができました。

In the experiment using this program, as shown in the following video, real-time object pose estimation, object grasping, and manipulation by the robot could be performed.

<div style="display: flex; gap: 10px;">
    <video src="https://github.com/user-attachments/assets/2fdad2bc-df26-4aa1-b1c3-d2336ef581a0" autoplay muted loop></video>
</div>

実行手順を終えると、ターミナル1では「Start>>>」という文字が出ていると思います。そこで、Enterキーを押すと以下のような画面になります。

After completing the execution steps, you should see the text "Start>>>" in Terminal 1. When you press the Enter key, the following screen should appear.

<div style="display: flex; gap: 10px;">
    <img src="https://github.com/user-attachments/assets/8e2174d8-53b0-412e-8118-79d6d47e4a05" alt="Image 1" width="300">
    <img src="https://github.com/user-attachments/assets/42bb829f-25bc-4dae-90a4-ed9c1c61977d" alt="Image 2" width="300">
</div>

このとき、ロボットは計測した点群データを基にオブジェクトの姿勢を繰り返し推定している状態になります。この状態の間であれば、オブジェクトの位置を自由に動かしてもらって構いません。ただし、動かしている人の手がセンサに写っていると姿勢推定のプログラムが止まるので、動かした後は手がセンサに映らないようにしましょう。オブジェクトを動かした後、**オブジェクトの姿勢を表すTFが異常でないかを確認してから**再度ENTERキーを押します。すると、推定した姿勢を基にオブジェクトの適切な箇所(TFの中心から赤い軸を基に＋約４cmの位置)を掴むようにロボットが動作します。

At this point, the robot is continuously estimating the object's pose based on the captured point cloud data. During this state, you are free to move the object around. However, if the hand of the person moving the object is visible to the sensor, the pose estimation program will stop, so make sure the hand is not in view of the sensor after moving the object. After moving the object, ensure that the TF representing the object's pose is correct before pressing the Enter key again. When you do, the robot will move to grasp the appropriate part of the object (about 4 cm from the center of the TF, based on the red axis).

# Acknowledgement

https://github.com/vinits5/masknet/tree/main

# Author
 
* Yu Iwai
* The University of Kitakyushu
 
# License

This project is release under the MIT License.
