/*
 File name: HW6.h
 Author: Ender Yağcılar
 EE 576 - Project 6
 E-mail: ender.yagcilar@boun.edu.tr
 Date created: 23.05.2023
 Date last modified: 24.05.2023
 */

#include <iostream>
#include <vector>
#include <set>
#include <dirent.h>
#include <cstring>
#include <algorithm>
#include <thread>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>
#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h> // for pcl::removeNaNFromPointCloud
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/filters/voxel_grid.h>

using namespace std::chrono_literals;

struct ObjectFeatures
{
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr desc;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster;
    std::vector<bool>
        exist;
    Eigen::Vector3f translation;
};

class Term_Project
{
public:
    std::string dataset_path = "./dataset/";
    std::vector<std::string> filenames;

    // Methods
    void get_unique_filenames();                                                                                                                             // this method gets filenames like frame-000001 frame-000002 etc.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr create_pcd(std::string filename);                                                                                 // This method creates .pcd file from .png and .pgm file
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> region_growing_rgb_segmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string filename); // https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_rgb_segmentation.html#region-growing-rgb-segmentation
    void showPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud);
    void print2DPicture(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud, std::string filename);
    std::vector<Eigen::Vector3f> trackSegments(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusterClouds1, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusterClouds2);
    void print2DPicture(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string filename, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters, std::vector<Eigen::Vector3f> translations);
    void print2DPicture_w_bounding_box(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusterClouds, std::string filename);
    void processImage(std::string filename);
    std::vector<ObjectFeatures> _objects;

private:
};