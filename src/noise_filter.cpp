#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

class PointCloudProcessor
{
public:
    PointCloudProcessor() //: last_publish_time(ros::Time::now())
    {
        // /posture_estimation_doneパラメータを初期化（ここではFalseに設定）
        nh_.setParam("/posture_estimation_done", true);  // ここでパラメータを設定

        // サブスクライバの設定（PointCloud2メッセージを受け取る）
        sub_ = nh_.subscribe("/camera/depth/color/points", 100000, &PointCloudProcessor::pointCloudCallback, this);
        
        // パブリッシャの設定（平面除去後の点群をパブリッシュ）
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/processed_point_cloud", 100000);

        // パラメータ監視のためにタイマーを設定（例えば、1Hzでチェック）
        param_check_timer_ = nh_.createTimer(ros::Duration(0.005), &PointCloudProcessor::checkPostureEstimationDone, this);
    }

    void checkPostureEstimationDone(const ros::TimerEvent&)
    {
        // posture_estimation_doneパラメータの値を取得
        bool posture_done;
        if (nh_.getParam("/posture_estimation_done", posture_done))
        {
            // パラメータがfalseの場合、点群処理を行う
            if (!posture_done)
            {
                // pointCloudCallbackを実行するためのフラグを設定
                process_point_cloud_ = true;
            }
            else
            {
                
                // posture_estimation_doneがtrueの場合、点群処理を停止
                process_point_cloud_ = false;

                pcl::PointCloud<pcl::PointXYZ>::Ptr empty_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                // posture_estimation_done が true の場合は空のメッセージをパブリッシュ
                sensor_msgs::PointCloud2 empty_msg;
                // ヘッダーを設定
                empty_msg.header.frame_id = latest_msg_.header.frame_id;
                empty_msg.header.stamp = ros::Time::now();
    
                pub_.publish(empty_msg);
            }
        }
        else
        {
            ROS_WARN("Failed to get parameter '/posture_estimation_done'");
        }
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        // 受け取った点群メッセージをメンバー変数に保存
        latest_msg_ = *msg;
        if (!process_point_cloud_)
        {
            return;  // posture_estimation_doneがtrueの場合は処理しない
        }

        // 受け取ったPointCloud2メッセージをPCL形式に変換
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // 平面検出
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.009);

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty())
        {
            ROS_WARN("Could not estimate a planar model for the given dataset.");
            return;
        }

        // 平面を除去
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*cloud_filtered);

        // ダウンサンプリング
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(cloud_filtered);
        voxel_filter.setLeafSize(0.0027f, 0.0027f, 0.0027f);  //従来手法：0.0025   提案手法：0.003
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*cloud_downsampled);

        // 外れ値除去
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_downsampled);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_denoised(new pcl::PointCloud<pcl::PointXYZ>);
        sor.filter(*cloud_denoised);

        // ノイズ除去後の点群をPointCloud2形式に変換
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_denoised, output);
        output.header.frame_id = msg->header.frame_id;
        output.header.stamp = ros::Time::now();

        // パブリッシュの頻度を制限
        //if ((ros::Time::now() - last_publish_time).toSec() > publish_interval) // publish_intervalは秒数で定義
        //{
        //    pub_.publish(output);
        //    last_publish_time = ros::Time::now();
        //}
        
        pub_.publish(output);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
    ros::Timer param_check_timer_;  // パラメータを定期的に確認するタイマー
    bool process_point_cloud_ = false;  // posture_estimation_doneがfalseの場合のみ点群処理を行うフラグ
    // 最新の点群メッセージを保持する変数
    sensor_msgs::PointCloud2 latest_msg_;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud_; // 前回の点群を保存する変数
    //ros::Time last_publish_time; // 最後にパブリッシュした時間
    //double publish_interval = 0.05; // 約30Hzでパブリッシュ
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_processor");
    PointCloudProcessor processor;
    ros::spin();
    return 0;
}
