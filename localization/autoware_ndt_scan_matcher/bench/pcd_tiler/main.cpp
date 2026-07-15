#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace fs = std::filesystem;

int main(int argc, char ** argv)
{
  if (argc != 4) {
    std::cerr << "usage: ndt_pcd_tiler INPUT.pcd OUTPUT_DIR TILE_SIZE_M\n";
    return 2;
  }

  const fs::path input = argv[1];
  const fs::path output = argv[2];
  const double tile_size = std::stod(argv[3]);
  if (!(tile_size > 0.0)) {
    std::cerr << "tile size must be positive\n";
    return 2;
  }

  pcl::PointCloud<pcl::PointXYZ> source;
  if (pcl::io::loadPCDFile(input.string(), source) != 0) {
    std::cerr << "failed to load " << input << "\n";
    return 1;
  }

  using Key = std::pair<long long, long long>;
  std::map<Key, pcl::PointCloud<pcl::PointXYZ>> tiles;
  for (const auto & point : source.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    const auto ix = static_cast<long long>(std::floor(point.x / tile_size));
    const auto iy = static_cast<long long>(std::floor(point.y / tile_size));
    tiles[{ix, iy}].push_back(point);
  }
  const auto source_points = source.size();
  source.clear();
  source.points.shrink_to_fit();

  fs::create_directories(output);
  std::ofstream metadata(output / "metadata.yaml");
  metadata << "x_resolution: " << tile_size << "\n";
  metadata << "y_resolution: " << tile_size << "\n";
  for (auto & [key, cloud] : tiles) {
    const auto [ix, iy] = key;
    const std::string name = "tile_" + std::to_string(ix) + "_" + std::to_string(iy) + ".pcd";
    cloud.width = static_cast<std::uint32_t>(cloud.size());
    cloud.height = 1;
    if (pcl::io::savePCDFileBinary((output / name).string(), cloud) != 0) {
      std::cerr << "failed to save " << name << "\n";
      return 1;
    }
    metadata << name << ": [" << ix * tile_size << ", " << iy * tile_size << "]\n";
  }
  std::cout << "points=" << source_points << " tiles=" << tiles.size() << "\n";
  return metadata.good() ? 0 : 1;
}
