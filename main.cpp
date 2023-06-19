#include "Term_Project.h"

#include <pcl/common/centroid.h>

int main()
{

  Term_Project project;

  // Files should be inside dataset folder with same name for depth and color images and .pgm and .png extensions respectively.
  project.get_unique_filenames();

  for (int i = 0; i < project.filenames.size(); i++)
  {
    project.processImage(project.filenames[i]);
  }

  for (int i = 0; i < project._objects.size(); i++)
  {
    std::cout << "\n### Object " << i << " has " << project._objects[i].cluster->size() << " points." << std::endl;
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*project._objects[i].cluster, centroid);
    std::cout << "First position " << centroid << std::endl;
    std::cout << "moved by vector " << project._objects[i].translation << std::endl;
    if (project._objects[i].exist.back())
    {
      std::cout << "visible now" << std::endl;
    }
    else
    {
      std::cout << "not visible now" << std::endl;
    }
    int count = 0;
    for (int j = 0; j < project._objects[i].exist.size(); j++)
    {
      if (project._objects[i].exist[j])
      {
        count++;
      }
    }
    std::cout << "visible count " << count << std::endl;
  }

  auto cloud1 = project.create_pcd(project.filenames[0]);
  // project.showPointCloud(cloud);
  auto clusters1 = project.region_growing_rgb_segmentation(cloud1, project.filenames[0]);

  auto cloud2 = project.create_pcd(project.filenames[10]);
  // project.showPointCloud(cloud);
  auto clusters2 = project.region_growing_rgb_segmentation(cloud2, project.filenames[10]);

  auto trackedSegments = project.trackSegments(clusters1, clusters2);

  project.print2DPicture(cloud1, "tracking.png", clusters1, trackedSegments);
  cloud1->clear();
  cloud2->clear();

  return (0);
}
