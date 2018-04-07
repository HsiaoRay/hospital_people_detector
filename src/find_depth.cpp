#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include "boost/filesystem.hpp"
#include <boost/algorithm/string/replace.hpp>

//#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

#include "sensor_msgs/CameraInfo.h"
#include <tf/transform_listener.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <yaml-cpp/yaml.h>

using namespace cv;
using namespace std;
using namespace boost::filesystem;

ros::Publisher pcl_pub;

struct Proposal
{
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double depth;
};

int last_xpos, last_ypos;

std::ofstream label_feature_file;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if ( flags == (EVENT_FLAG_LBUTTON) )
     {
          cout << "Left mouse button is clicked - position (" << x << ", " << y << ")" << endl;
          last_xpos = x;
          last_ypos = y;
          //cvDestroyWindow("Display window");
     }
}

double get_depth(cv::Mat &depthImage, double fx, double fy, double cx, double cy)
{
    double depth = -1;

    //Convert image to point cloud
    //Sample every 4th pixel in the image to build the point cloud
    int sampler = 1;

    int width_im = depthImage.cols;
    int height_im = depthImage.rows;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_in->width = int(width_im/sampler);
    cloud_in->height = int(height_im/sampler);
    cloud_in->is_dense = true; //all the data in cloud_in is finite (No Inf/NaN values)
    cloud_in->points.resize (cloud_in->width * cloud_in->height);

    // Factor to convert camera measuremets to meters
    double meter_factor = 1000.0;

    int i=0;
    double max_depth = 0;
    for (int x=0; x<width_im; x+=sampler) {
        for (int y=0; y<height_im; y+=sampler) {

            float depth = (float)depthImage.at<uint16_t>(y, x) / meter_factor;
            if(depth > max_depth)
                max_depth = depth;

            if (std::isnan(depth))
                depth = INFINITY;

            if (depth==INFINITY) continue;

            float pclx = (x - cx ) / fx * depth;
            float pcly = (y - cy ) / fy * depth;

            if ( i <= cloud_in->points.size() )
            {
                cloud_in->points[i].x = pclx;
                cloud_in->points[i].y = pcly;
                cloud_in->points[i].z = depth;
            }
            else
                cout << "not in size" << endl;

            i++;

        }
    }

    //Voxel filter: Downsample the point cloud using a leaf size of 0.1mts
    double leaf_size = 0.025f;
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setLeafSize (leaf_size, leaf_size, leaf_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(cloud_in);
    vg.filter(*cloud_down);
//    cloud_down = cloud_in;

    //ground plane coefficients
    double coeff_a = 0.0848112;
    double coeff_b = -0.993395;
    double coeff_c = -0.0772927;
    double coeff_d = 1.10195;

//    // Get index of points close to ground plane
//    pcl::PointIndices index_cloud;
//    for (size_t i = 0; i < cloud_down->points.size (); ++i)
//    {
//        double plane_y = -(coeff_a*cloud_down->points[i].x + coeff_c*cloud_down->points[i].z +coeff_d)/coeff_b;
//        // Keep points above the plane and under the ceiling (between 30cm - 2mts)
//        if(cloud_down->points[i].y < plane_y-0.3 && cloud_down->points[i].y > plane_y-2){
//            index_cloud.indices.push_back(i);}
//    }

//    // Remove ground plane: Remove points close to ground plane
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_plane (new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::ExtractIndices<pcl::PointXYZ> eifilter (true);
//    eifilter.setInputCloud (cloud_down);
//    eifilter.setIndices (boost::make_shared<const pcl::PointIndices> (index_cloud));
//    eifilter.filter (*cloud_without_plane);

//    cout << "num points cloud wo plane: " << cloud_without_plane->points.size() << endl;

    // Creating the KdTree object for segmentation
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_down);

    // Apply segmentation
    std::vector<pcl::PointIndices> point_clusters;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.2); //0.12 points separated less than 12cm are part of same cluster
    ec.setMinClusterSize (15);
    ec.setMaxClusterSize (20000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_down);
    ec.extract (point_clusters);

    cout << "found " << point_clusters.size() << " clusters" << endl;

    double center_x = width_im/2;
    double center_y = height_im/2;

    const float scaleFactor = 0.05f;
    depthImage.convertTo(depthImage, CV_8UC1, scaleFactor);

    //cv::normalize(depthImage, depthImage, 0, 20000, NORM_MINMAX, CV_16UC1);
    cv::cvtColor(depthImage, depthImage, cv::COLOR_GRAY2BGR);

    std::vector<double> cluster_sizes;
    std::vector<double> cluster_center_distances;
    std::vector<double> cluster_z_distances;
    std::vector<pcl::PointXYZ> cluster_centers;

    //evaluate each cluster
    RNG rng;
    for(int i=0; i<point_clusters.size(); i++)
    {
      pcl::PointIndices cluster = point_clusters.at(i);
      cluster_sizes.push_back(cluster.indices.size());

      pcl::PointXYZ cluster_center;
      Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));

      int size_cluster = 0;
      for(int j=0; j<cluster.indices.size(); j++)
      {
          if(cloud_down->points[cluster.indices[j]].z > 0.1)
          {
              int im_x = int(cloud_down->points[cluster.indices[j]].x * (fx / cloud_down->points[cluster.indices[j]].z) + cx);
              int im_y = int(cloud_down->points[cluster.indices[j]].y * (fy / cloud_down->points[cluster.indices[j]].z) + cy);

              cluster_center.x += im_x;
              cluster_center.y += im_y;
              cluster_center.z += cloud_down->points[cluster.indices[j]].z;
              size_cluster += 1;

              circle(depthImage, Point(im_x, im_y),1, color,CV_FILLED, 1,0);
          }
          else
              cout << "z component very small" << endl;
      }
      cluster_center.x /= size_cluster;
      cluster_center.y /= size_cluster;
      cluster_center.z /= size_cluster;

      circle(depthImage, Point(cluster_center.x, cluster_center.y),7, Scalar(255,255,255),CV_FILLED, 8,0);
      circle(depthImage, Point(cluster_center.x, cluster_center.y),5, color,CV_FILLED, 8,0);

      cluster_centers.push_back(cluster_center);

      cluster_center_distances.push_back(hypot(cluster_center.x-center_x, cluster_center.y-center_y));
      cluster_z_distances.push_back(cluster_center.z);
    }

    double size_ref = cloud_down->points.size();
    double distance_ref = hypot(center_x,center_y); //max possible distance
    double z_distance_ref = 10; //max distance from camera

    for(int i=0; i<cluster_sizes.size(); i++)
    {
      cluster_sizes.at(i) = cluster_sizes.at(i)/size_ref;
      cluster_center_distances.at(i) = 1 - (cluster_center_distances.at(i)/distance_ref);
      if(cluster_center_distances.at(i) < 0)
      {
          cout << "small number" << endl;
          cout << cluster_centers.at(i) << endl;
          cout << center_x << endl;
          cout << center_y << endl;
      }

      cluster_z_distances.at(i) = 1 - (cluster_z_distances.at(i)/z_distance_ref);

//      cout << "i " << i << " cluster size " << cluster_sizes.at(i) << endl;
//      cout << "i " << i << " cluster center dist " << cluster_center_distances.at(i) << endl;
//      cout << "i " << i << " cluster z dist " << cluster_z_distances.at(i) << endl;
//      cout << "i " << i << " score " << cluster_center_distances.at(i) * cluster_sizes.at(i) * cluster_z_distances.at(i) << endl;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE);// Create a window for display.
    setMouseCallback("Display window", CallBackFunc, NULL);
    imshow( "Display window", depthImage);                   // Show our image inside it.
    waitKey(0);

    int best_cluster = -1;
    double min_dist = 1000;
    pcl::PointXYZ best_cluster_center;
    for (int i=0; i<cluster_center_distances.size(); i++)
    {
        double dist = hypot((cluster_centers.at(i).x - last_xpos), (cluster_centers.at(i).y - last_ypos));
        if(dist < min_dist)
        {
            min_dist = dist;
            best_cluster = i;
            best_cluster_center = cluster_centers.at(i);
        }
    }

//    //find best cluster according to heuristic
//    int best_cluster = -1;
//    double best_cluster_score = -1;
//    pcl::PointXYZ best_cluster_center;

//    for(int i=0; i<cluster_center_distances.size(); i++)
//    {
//      double score = cluster_center_distances.at(i) * cluster_sizes.at(i) * cluster_z_distances.at(i);
//      if(score > best_cluster_score)
//      {
//        best_cluster_score = score;
//        best_cluster = i;
//        best_cluster_center = cluster_centers.at(i);
//      }
//    }

//    cout << "best cluster: " << best_cluster << " with score " << best_cluster_score << endl;

    if(best_cluster != -1)
    {
      //visualize biggest cluster
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_max_cluster (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud (cloud_down);
      extract.setIndices (boost::make_shared<const pcl::PointIndices> (point_clusters.at(best_cluster)));
      extract.setNegative (false);
      extract.filter (*cloud_max_cluster);

      pcl::PointCloud<pcl::PointXYZ>::Ptr msg_pcl (new pcl::PointCloud<pcl::PointXYZ>);
      copyPointCloud(*cloud_max_cluster, *msg_pcl);
      msg_pcl->header.frame_id = "camera_frame";
      msg_pcl->header.stamp = ros::Time().toNSec();

      for(int i=0; i<cloud_max_cluster->points.size(); i++)
      {
          int im_x = int(cloud_max_cluster->points[i].x * (fx / cloud_max_cluster->points[i].z) + cx);
          int im_y = int(cloud_max_cluster->points[i].y * (fy / cloud_max_cluster->points[i].z) + cy);
          circle(depthImage, Point(im_x, im_y),1, CvScalar(255,255,255),CV_FILLED, 8,0);
      }

      for(int i=0; i<10; i++)
      {
        pcl_pub.publish(msg_pcl);
        ros::Duration(0.05).sleep();
      }

      int im_x = best_cluster_center.x;
      int im_y = best_cluster_center.y;

      depth = best_cluster_center.z;
      stringstream ss;
      ss << depth;

      circle(depthImage, Point(im_x, im_y),5, CvScalar(255,255,255),CV_FILLED, 8,0);
      putText(depthImage, ss.str().c_str() , cvPoint(im_x+5, im_y),
              FONT_HERSHEY_SIMPLEX, 0.5, CvScalar(255,255,255), 1, CV_AA);

    }

    namedWindow( "Display window", WINDOW_AUTOSIZE);// Create a window for display.
    setMouseCallback("Display window", CallBackFunc, NULL);
    imshow( "Display window", depthImage);                   // Show our image inside it.
    waitKey(0);

    for(int i=0; i<cluster_centers.size(); i++)
    {
        int is_class = 0;
        if(i == best_cluster)
            is_class = 1;

        label_feature_file << cluster_sizes.at(i) << " " << cluster_center_distances.at(i) << " " << cluster_z_distances.at(i) << " " << is_class << endl;
    }

    return depth;
}

int main (int argc, char** argv)
{
    ros::init (argc, argv, "proposals_generation");
    ros::NodeHandle nh;

    string feature_file("/home/kollmitz/datasets/hospital_people/data/depth_label_features2.txt");
    label_feature_file.open(feature_file);
    label_feature_file << "size_feature, xy-center-dist-feature z-dist-feature cls" << endl;

    pcl_pub = nh.advertise< pcl::PointCloud<pcl::PointXYZ> >("mobility_pcl", 1);

    std::string annotation_folder = "/home/kollmitz/datasets/hospital_people/data/Annotations_DepthJet/";
    std::string depth_annotation_folder = "/home/kollmitz/datasets/dataset/Annotations_RGB_Depth/";
    std::string image_path = "/home/kollmitz/datasets/dataset/Depth/";

    // camera calibration files
    double fx= 540.686;
    double fy = 540.686;

    double cx = 479.75;
    double cy = 269.75;

//    //find all annotation files in folder

//    path p (annotation_folder);

//    directory_iterator end_itr;
//    std::vector<string> annotation_files;

//    // cycle through the directory
//    for (directory_iterator itr(p); itr != end_itr; ++itr)
//    {
//        // If it's not a directory, list it. If you want to list directories too, just remove this check.
//        if (is_regular_file(itr->path())) {
//            // assign current file name to current_file and echo it out to the console.
//            string current_file = itr->path().string();
////            cout << current_file << endl;
//            if(current_file.find(".yml")!=string::npos)
//                annotation_files.push_back(current_file);
//        }
//    }

//    cout << "found " << annotation_files.size() << " annotation files." << endl;

    std::string image_set_file = "/home/kollmitz/datasets/hospital_people/data/ImageSets/train_RGB.txt";

    //open imageset file
    std::string line;
    std::ifstream myfile (image_set_file);

    if (myfile.is_open())
    {
        int num_entries = 0;
        while ( std::getline (myfile,line) && ros::ok() && num_entries < 100)
        {
            std::stringstream ss;
            ss << annotation_folder << line << ".yml";

            double depth;
            bool flipped = false;

            cout << ss.str().c_str() << endl;
            YAML::Node config = YAML::LoadFile(ss.str().c_str());

            std::string image_file = config["annotation"]["filename"].as<std::string>();
            if(image_file.find("inv")!=string::npos)
            {
                flipped = true;
                image_file.erase(image_file.find("_inv"), 4);
            }

            stringstream ss2;
            ss2 << image_path << image_file;
            cv::Mat depth_image;

            cout << "image file " << ss2.str().c_str() << endl;
            depth_image = imread(ss2.str().c_str(), CV_16UC1);
            //resize image (not best option for depth image to label dimensions, but works ok)
            resize(depth_image, depth_image, cvSize(960, 540));
            if(flipped)
                cv::flip(depth_image,depth_image,1);

            YAML::Node bboxes = config["annotation"]["object"];

            for(int j=0; j<bboxes.size(); j++)
            {
                int xmax = bboxes[j]["bndbox"]["xmax"].as<int>();
                int xmin = bboxes[j]["bndbox"]["xmin"].as<int>();
                int ymax = bboxes[j]["bndbox"]["ymax"].as<int>();
                int ymin = bboxes[j]["bndbox"]["ymin"].as<int>();

                cv::Rect myROI(xmin, ymin, xmax-xmin, ymax-ymin);
                cv::Mat croppedImage = depth_image(myROI);

                depth = get_depth(croppedImage, fx, fy, cx, cy);
                bboxes[j]["bndbox"]["depth"] = depth;

                cout << depth << endl;
                num_entries += 1;

//                namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//                imshow( "Display window", croppedImage );                   // Show our image inside it.
//                waitKey(25);
            }

//            //write new yaml file
//            stringstream ss3;
//            ss3 << depth_annotation_folder << line << ".yml";

//            //check if file exists
////            ifstream ifile(ss3.str().c_str());
////            if (ifile) {
////              cout << "file " << ss3.str().c_str() << " already exists!!" << endl;
////              return -1;
////            }

//            std::ofstream yaml_with_depth;
//            yaml_with_depth.open(ss3.str().c_str());
//            if(yaml_with_depth.is_open())
//            {
//              yaml_with_depth << "%YAML:1.0" << endl;

//              string yaml_string = YAML::Dump(config);
//              boost::replace_all(yaml_string, "!<!>", "");

//              yaml_with_depth << yaml_string;
//              yaml_with_depth.close();
//            }
        }
    }

//    //write one file per image with proposal coordinates <min_x,max_x,min_y,max_y,depth>
//    std::string image_set_file = "/home/kollmitz/datasets/hospital_people/data/ImageSets/test_RGB.txt";
//    std::string image_path = "/home/kollmitz/datasets/hospital_people/data/Depth/";

//    //open imageset file
//    std::string line;
//    std::ifstream myfile (image_set_file);
//    double depth;

//    // camera calibration files
//    double fx= 540.686;
//    double fy = 540.686;

//    double cx = 479.75;
//    double cy = 269.75;

//    //  if (myfile.is_open())
//    //  {
//    //    while ( std::getline (myfile,line) )
//    //    {

//    std::stringstream ss;
//    line = "seq_1468844955.6972733080";
//    ss << image_path << line << ".png";
//    cv::Mat depth_image;
//    depth_image = imread(ss.str().c_str(), CV_16UC1);
//    //{xmax: '647', xmin: '534', ymax: '420', ymin: '67'}
//    cv::Rect myROI(534, 67, 647-534, 420-67);
//    cv::Mat croppedImage = depth_image(myROI);

//    depth = get_depth(croppedImage, fx, fy, cx, cy);
//    cout << depth << endl;

//    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//    imshow( "Display window", croppedImage );                   // Show our image inside it.
//    waitKey(0);

    //    }
    //    myfile.close();
    //  }

    //else std::cout << "Unable to open file";

    label_feature_file.close();
    return 0;
}
