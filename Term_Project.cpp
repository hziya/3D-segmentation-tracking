#include "Term_Project.h"

#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/common/centroid.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>

void Term_Project::get_unique_filenames()
{
    std::set<std::string> uniqueNames;

    DIR *dir = opendir(dataset_path.c_str());
    if (dir)
    {
        dirent *entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            if (entry->d_type == DT_REG)
            {
                std::string filename(entry->d_name);
                std::string first12Chars = filename.substr(0, 12);

                if (uniqueNames.count(first12Chars) == 0)
                {
                    uniqueNames.insert(first12Chars);
                    filenames.push_back(first12Chars);
                }
            }
        }
        closedir(dir);
    }
    else
    {
        std::cout << "Failed to open directory." << std::endl;
    }

    // Sort the filenames in ascending order
    std::sort(filenames.begin(), filenames.end());
    std::cout << "filenames are taken." << std::endl;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Term_Project::create_pcd(std::string filename)
{
    cv::Mat rgb_image = cv::imread(dataset_path + filename + ".color.jpg");
    cv::Mat depth_map = cv::imread(dataset_path + filename + ".depth.pgm", cv::IMREAD_UNCHANGED);

    // Camera intrinsics
    float fx = 577.591;
    float fy = 578.73;
    float cx = 318.905;
    float cy = 242.684;
    float m_colorWidth = 1296;
    float m_colorHeight = 968;
    float m_depthWidth = 640;
    float m_depthHeight = 480;
    float m_depthShift = 1000;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->width = m_depthWidth;
    cloud->height = m_depthHeight;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);
    cv::resize(rgb_image, rgb_image, depth_map.size());

    for (int y = 0; y < m_depthHeight; y++)
    {
        for (int x = 0; x < m_depthWidth; x++)
        {
            pcl::PointXYZRGB &point = cloud->at(x, y);

            // Set RGB values
            cv::Vec3b rgb = rgb_image.at<cv::Vec3b>(y, x);
            point.r = rgb[2];
            point.g = rgb[1];
            point.b = rgb[0];

            // Set depth value
            ushort depth_value = depth_map.at<ushort>(y, x);
            point.z = static_cast<float>(depth_value) / m_depthShift;

            // Set 3D coordinates
            point.x = (x - cx) * point.z / fx;
            point.y = (y - cy) * point.z / fy;
        }
    }

    // pcl::io::savePCDFileBinary("output_cloud.pcd", *cloud);
    std::cout << "PCD File is created for " << filename << std::endl;

    // print2DPicture(cloud);
    return cloud;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> Term_Project::region_growing_rgb_segmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string filename)
{
    // Downsample the point cloud using a voxel grid filter
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(0.01, 0.01, 0.01); // Adjust the leaf size as needed
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    voxel_grid.filter(*downsampled_cloud);

    // Remove NaN points from the downsampled cloud
    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*downsampled_cloud, *indices);

    // Perform region growing RGB segmentation
    pcl::search::Search<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
    reg.setInputCloud(downsampled_cloud);
    reg.setIndices(indices);
    reg.setSearchMethod(tree);
    reg.setDistanceThreshold(10);
    reg.setPointColorThreshold(6);
    reg.setRegionColorThreshold(5);
    reg.setMinClusterSize(600);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusterClouds;

    // Define the camera orientation vector
    Eigen::Vector3f cameraOrientation(-0.269932, -0.872939, -0.406343); // I read this from pose.txt file and hardcoded here

    for (const auto &cluster : clusters)
    {
        // std::cout << "Cluster size: " << cluster.indices.size() << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (const auto &index : cluster.indices)
        {
            clusterCloud->points.push_back(downsampled_cloud->points[index]);
        }

        // FLAT OLMAYANLARI BURADA ELİYORUZ
        Eigen::Vector4f plane_parameters;
        float curvature;
        pcl::computePointNormal(*clusterCloud, plane_parameters, curvature);
        // std::cout << "Curvature: " << curvature << std::endl;
        // if not flat surface
        if (curvature > 0.04)
        {
            // std::cout << "Not a flat surface" << std::endl;
            continue;
        }

        // BURADA HORİZONTAL OLMAYANLARI ELİYORUZ.
        //  Compute the normal vector of the cluster
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud(clusterCloud);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
        normalEstimation.setSearchMethod(tree);
        normalEstimation.setKSearch(10); // Adjust the value as needed
        normalEstimation.compute(*normals);

        float cos_angle_sum = 0;

        // Check if the cluster is horizontal (e.g., normal vector aligns with camera orientation)
        bool isHorizontal = true;
        for (const auto &normal : normals->points)
        {
            Eigen::Vector3f normalVector(normal.normal_x, normal.normal_y, normal.normal_z);
            float dotProduct = normalVector.dot(cameraOrientation);
            float cosAngle = dotProduct / (normalVector.norm() * cameraOrientation.norm());

            cos_angle_sum += cosAngle;
        }
        float cos_angle_avg = cos_angle_sum / cluster.indices.size();

        if (cos_angle_avg < 0.3)
        {
            normals.reset();
            continue; // Skip non-horizontal clusters
        }

        clusterClouds.push_back(clusterCloud);
        normals.reset();
    }

    print2DPicture_w_bounding_box(clusterClouds, filename);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    print2DPicture(colored_cloud, filename + "_colored_cloud.png");

    // Get the colored point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr flatClusters(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto &cluster : clusterClouds)
    {
        *flatClusters += *cluster;
    }
    print2DPicture(flatClusters, filename + "_flat_clusters.png");
    indices.reset();
    downsampled_cloud.reset();
    colored_cloud.reset();
    tree.reset();
    flatClusters.reset();

    // delete indices;
    // delete downsampled_cloud;
    // delete flatClusters;
    // delete colored_cloud;
    // delete normals;
    // delete tree;

    return clusterClouds;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeSiftDescriptors(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    // Parameters for sift computation
    const float min_scale = 0.01f;
    const int n_octaves = 3;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.001f;

    // Compute sift keypoints
    pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> sift_keypoints;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());

    sift.setInputCloud(cloud);
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.compute(sift_keypoints);
    // std::cout << "SIFT keypoints found: " << sift_keypoints.points.size() << std::endl;
    sift_keypoints.points.resize(20);

    // Convert the sift keypoints into a PointXYZ format
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sift_keypoints_xyz(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(sift_keypoints, *sift_keypoints_xyz);

    // Compute the normals
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    ne.setInputCloud(sift_keypoints_xyz);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.05); // Radius for normal estimation
    ne.compute(*normals);

    // Compute the FPFH features
    pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features(new pcl::PointCloud<pcl::FPFHSignature33>);

    fpfh.setInputCloud(sift_keypoints_xyz);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(0.05); // Radius for FPFH computation
    fpfh.compute(*fpfh_features);

    tree.reset();
    normals.reset();
    sift_keypoints_xyz.reset();

    return fpfh_features;
}

double distanceBetweenDescriptors(pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud1, pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud2)
{
    // Ensure both point clouds have the same size
    assert(cloud1->size() == cloud2->size());
    double total_distance = 0;
    // Calculate and print all distances
    for (size_t i = 0; i < cloud1->size(); ++i)
    {
        pcl::FPFHSignature33 &desc1 = cloud1->points[i];
        pcl::FPFHSignature33 &desc2 = cloud2->points[i];

        double distance = 0.0;
        for (int j = 0; j < pcl::FPFHSignature33::descriptorSize(); j++)
        {
            double diff = desc1.histogram[j] - desc2.histogram[j];
            distance += diff * diff;
        }
        distance = std::sqrt(distance);
        total_distance += distance;

        // std::cout << "Distance between point " << i << " descriptors: " << distance << std::endl;
    }
    return total_distance;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeDescriptors(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    normalEstimation.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normalEstimation.setRadiusSearch(0.03); // Set the radius for the computation of the normals
    normalEstimation.compute(*normals);

    // Then, compute the FPFH descriptors
    pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh.setRadiusSearch(0.05); // Set the radius for the computation of the descriptors
    fpfh.compute(*descriptors);

    return descriptors;
}

std::vector<Eigen::Vector3f> Term_Project::trackSegments(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusterClouds1, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusterClouds2)
{
    std::vector<Eigen::Vector3f> translations;
    // Track segments
    // std::cout << "number of clusters " << clusterClouds1.size() << std::endl;
    for (const auto &cluster1 : clusterClouds1)
    {
        bool isFound = false;
        auto desc1 = computeSiftDescriptors(cluster1);
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transformations;
        double min_distance = 1e6;
        int min_index = 0;
        for (int j = 0; j < clusterClouds2.size(); j++)
        {
            auto cluster2 = clusterClouds2[j];
            auto desc2 = computeSiftDescriptors(cluster2);

            auto distance = distanceBetweenDescriptors(desc1, desc2);
            if (distance < min_distance)
            {
                min_distance = distance;
                min_index = j;
            }
        }
        // std::cout << "min Distance: " << min_distance << std::endl;

        if (min_distance < 2000)
        {
            auto desc = computeDescriptors(cluster1);
            auto cluster2 = clusterClouds2[min_index];
            auto desc2 = computeDescriptors(cluster2);
            pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::FPFHSignature33> sac_ia;
            sac_ia.setInputSource(cluster1);
            sac_ia.setSourceFeatures(desc);
            sac_ia.setInputTarget(cluster2);
            sac_ia.setTargetFeatures(desc2);
            sac_ia.setMinSampleDistance(0.05f);
            sac_ia.setMaxCorrespondenceDistance(0.1f);
            sac_ia.setMaximumIterations(500);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr alignedCluster(new pcl::PointCloud<pcl::PointXYZRGB>);
            sac_ia.align(*alignedCluster);

            if (sac_ia.hasConverged())
            {
                // The estimated transformation is given by sac_ia.getFinalTransformation()
                Eigen::Matrix4f transformation = sac_ia.getFinalTransformation();

                // The translation vector is in the last column
                Eigen::Vector3f translation = transformation.block<3, 1>(0, 3);
                // std::cout << "Cluster aligned!" << std::endl;
                transformations.push_back(sac_ia.getFinalTransformation());
                // std::cout << translation << std::endl;
                // normalize
                translation = translation / translation.norm();
                translations.push_back(translation);
                isFound = true;
            }
            // else
            // {
            //     std::cout << "didn't converge\n";
            // }
            alignedCluster.reset();
        }

        if (!isFound)
        {
            // std::cout << "Cluster not found!" << std::endl;
            translations.push_back(Eigen::Vector3f(0, 0, 0));
        }
    }

    // Transform the point cloud
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // std::cout << "translations size " << translations.size() << std::endl;
    return translations;
}

void Term_Project::showPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud)
{

    pcl::visualization::PCLVisualizer viewerr("Point Cloud Viewer");
    viewerr.addPointCloud(pointCloud, "cloud");
    while (!viewerr.wasStopped())
    {
        viewerr.spinOnce();
    }
}

void Term_Project::print2DPicture_w_bounding_box(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusterClouds, std::string filename)
{
    // Define the size of the image
    int width = 640;
    int height = 480;

    // Define the range of x and y coordinates of the points
    double x_min = -1.0, x_max = 1.0;
    double y_min = -1.0, y_max = 1.0;

    // Create an empty OpenCV image
    cv::Mat image = cv::imread(dataset_path + filename + ".color.jpg");
    cv::resize(image, image, cv::Size(width, height));

    for (const auto &cluster : clusterClouds)
    {
        // Find the minimum and maximum coordinates of the cluster
        float cluster_x_min = std::numeric_limits<float>::max();
        float cluster_x_max = std::numeric_limits<float>::lowest();
        float cluster_y_min = std::numeric_limits<float>::max();
        float cluster_y_max = std::numeric_limits<float>::lowest();

        for (const auto &point : cluster->points)
        {
            // Normalize the x and y coordinates and convert them to image coordinates
            int u = (point.x - x_min) / (x_max - x_min) * width;
            int v = (point.y - y_min) / (y_max - y_min) * height;

            // Make sure the coordinates are within the image
            u = std::min(std::max(u, 0), width - 1);
            v = std::min(std::max(v, 0), height - 1);

            // Update the cluster's bounding box coordinates
            cluster_x_min = std::min(cluster_x_min, point.x);
            cluster_x_max = std::max(cluster_x_max, point.x);
            cluster_y_min = std::min(cluster_y_min, point.y);
            cluster_y_max = std::max(cluster_y_max, point.y);
        }

        // Calculate the center of the bounding box
        float center_x = (cluster_x_min + cluster_x_max) / 2.0;
        float center_y = (cluster_y_min + cluster_y_max) / 2.0;

        // Normalize the center coordinates and convert them to image coordinates
        int center_u = (center_x - x_min) / (x_max - x_min) * width;
        int center_v = (center_y - y_min) / (y_max - y_min) * height;

        // Draw the bounding box on the image for the current cluster
        int bbox_u_min = (cluster_x_min - x_min) / (x_max - x_min) * width;
        int bbox_u_max = (cluster_x_max - x_min) / (x_max - x_min) * width;
        int bbox_v_min = (cluster_y_min - y_min) / (y_max - y_min) * height;
        int bbox_v_max = (cluster_y_max - y_min) / (y_max - y_min) * height;

        cv::rectangle(image, cv::Point(bbox_u_min, bbox_v_min), cv::Point(bbox_u_max, bbox_v_max), cv::Scalar(0, 255, 0), 2);

        // Draw a cross at the center of the bounding box
        cv::drawMarker(image, cv::Point(center_u, center_v), cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 2);
    }

    // Save the image
    cv::imwrite(filename + "_flat_clusters_on_color_image.png", image);
}

void Term_Project::print2DPicture(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string filename)
{
    // Define the size of the image
    int width = 640;
    int height = 480;

    // Define the range of x and y coordinates of the points
    double x_min = -1.0, x_max = 1.0;
    double y_min = -1.0, y_max = 1.0;

    // Create an empty OpenCV image
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    for (const auto &point : cloud->points)
    {
        // Normalize the x and y coordinates and convert them to image coordinates
        int u = (point.x - x_min) / (x_max - x_min) * width;
        int v = (point.y - y_min) / (y_max - y_min) * height;

        // Make sure the coordinates are within the image
        u = std::min(std::max(u, 0), width - 1);
        v = std::min(std::max(v, 0), height - 1);

        // Set the pixel color
        // In this example, we color the pixel based on the z-coordinate of the point
        // The points with higher z will be brighter
        image.at<cv::Vec3b>(v, u) = cv::Vec3b(point.b, point.g, point.r);
    }

    // Save the image
    cv::imwrite(filename, image);
}

void Term_Project::print2DPicture(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string filename, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters, std::vector<Eigen::Vector3f> translations)
{
    // Define the size of the image
    int width = 640;
    int height = 480;

    // Define the range of x and y coordinates of the points
    double x_min = -1.0, x_max = 1.0;
    double y_min = -1.0, y_max = 1.0;

    // Create an empty OpenCV image
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    for (const auto &point : cloud->points)
    {
        // Normalize the x and y coordinates and convert them to image coordinates
        int u = (point.x - x_min) / (x_max - x_min) * width;
        int v = (point.y - y_min) / (y_max - y_min) * height;

        // Make sure the coordinates are within the image
        u = std::min(std::max(u, 0), width - 1);
        v = std::min(std::max(v, 0), height - 1);

        // Set the pixel color
        // In this example, we color the pixel based on the z-coordinate of the point
        // The points with higher z will be brighter
        image.at<cv::Vec3b>(v, u) = cv::Vec3b(point.b, point.g, point.r);
    }
    // std::cout << "image created\n";
    int i = 0;

    // std::cout << "clusters size " << clusters.size() << std::endl;
    // std::cout << "translation size " << translations.size() << std::endl;
    for (int i = 0; i < clusters.size(); i++)
    {
        auto translation = translations[i];
        if (translation[0] == 0 && translation[1] == 0 && translation[2] == 0)
        {
            continue;
        }
        // std::cout << "translation " << translation[0] << ", " << translation[1] << ", " << translation[2] << std::endl;
        auto cluster = clusters[i];
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cluster, centroid);
        // std::cout << "centroid " << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << std::endl;
        int u1 = (centroid[0] - x_min) / (x_max - x_min) * width;
        int v1 = (centroid[1] - y_min) / (y_max - y_min) * height;
        int u2 = (centroid[0] + 0.1 * translation[0] - x_min) / (x_max - x_min) * width;
        int v2 = (centroid[1] + 0.1 * translation[1] - y_min) / (y_max - y_min) * height;

        cv::line(image, cv::Point(u1, v1), cv::Point(u2, v2), cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    // Save the image
    cv::imwrite(filename, image);
}

void Term_Project::processImage(std::string filename)
{
    auto cloud = create_pcd(filename);
    auto clusters = region_growing_rgb_segmentation(cloud, filename);
    std::vector<bool> exist(_objects.size(), false);
    std::vector<Eigen::Vector3f> translations;
    // Track segments
    // std::cout << "number of clusters " << clusters.size() << std::endl;
    for (const auto &cluster1 : clusters)
    {
        bool isFound = false;
        auto desc1 = computeSiftDescriptors(cluster1);
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transformations;
        double min_distance = 1e6;
        int min_index = 0;
        for (int j = 0; j < _objects.size(); j++)
        {
            auto cluster2 = _objects[j].cluster;
            auto desc2 = _objects[j].desc;

            auto distance = distanceBetweenDescriptors(desc1, desc2);
            if (distance < min_distance)
            {
                min_distance = distance;
                min_index = j;
            }
        }
        // std::cout << "min Distance: " << min_distance << std::endl;

        if (min_distance < 2000)
        {
            exist[min_index] = true;
            auto desc = computeDescriptors(cluster1);
            auto cluster2 = _objects[min_index].cluster;
            auto desc2 = computeDescriptors(cluster2);
            pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::FPFHSignature33> sac_ia;
            sac_ia.setInputSource(cluster1);
            sac_ia.setSourceFeatures(desc);
            sac_ia.setInputTarget(cluster2);
            sac_ia.setTargetFeatures(desc2);
            sac_ia.setMinSampleDistance(0.05f);
            sac_ia.setMaxCorrespondenceDistance(0.1f);
            sac_ia.setMaximumIterations(500);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr alignedCluster(new pcl::PointCloud<pcl::PointXYZRGB>);
            sac_ia.align(*alignedCluster);

            if (sac_ia.hasConverged())
            {
                // The estimated transformation is given by sac_ia.getFinalTransformation()
                Eigen::Matrix4f transformation = sac_ia.getFinalTransformation();

                // The translation vector is in the last column
                Eigen::Vector3f translation = transformation.block<3, 1>(0, 3);
                // std::cout << "Cluster aligned!" << std::endl;
                transformations.push_back(sac_ia.getFinalTransformation());
                // std::cout << translation << std::endl;
                // normalize
                translation = translation / translation.norm();
                translations.push_back(translation);
                isFound = true;
                _objects[min_index].translation = translation;
            }
            else
            {
                // std::cout << "didn't converge\n";
            }
            alignedCluster.reset();
        }

        else
        {
            ObjectFeatures object;
            object.translation = Eigen::Vector3f(0, 0, 0);
            object.desc = desc1;
            object.exist.push_back(true);
            object.cluster = cluster1;
            _objects.push_back(object);
        }
    }

    for (int i = 0; i < exist.size(); i++)
    {
        _objects[i].exist.push_back(exist[i]);
    }

    // std::cout << "objects: ";

    // for (int i = 0; i < _objects.size(); i++)
    // {
    //     if (_objects[i].exist.back())
    //     {
    //         std::cout << "exist ";
    //     }
    //     else
    //     {
    //         std::cout << "none ";
    //     }
    // }
    // std::cout << std::endl;
}