#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <octomap/octomap.h>
#include <pybind11/numpy.h>

#include  <superray_octomap/SuperRayOcTree.h>

namespace py = pybind11;

PYBIND11_MODULE(stereo_slam_pybind, m) {
    py::class_<octomap::OcTree>(m, "OcTree")
        .def(py::init<double>())  // resolution
        .def("setProbHit", [](octomap::OcTree &tree, double prob_hit) {
            tree.setProbHit(prob_hit);
        })
        .def("setProbMiss", [](octomap::OcTree &tree, double prob_miss) {
            tree.setProbMiss(prob_miss);
        })
        .def("insert_point", [](octomap::OcTree &tree, std::vector<double> point) {
            tree.updateNode(octomap::point3d(point[0], point[1], point[2]), true);
        })
        .def("insert_ray", [](octomap::OcTree &tree, std::vector<double> origin, std::vector<double> end,  bool lazy_eval) {
            tree.insertRay(octomap::point3d(origin[0], origin[1], origin[2]),
                           octomap::point3d(end[0], end[1], end[2]),
                           -1.0, lazy_eval);
        })
        .def("read_binary", [](octomap::OcTree &tree, const std::string &filename) {
            tree.readBinary(filename);
        })
        .def("write_binary", [](octomap::OcTree &tree, const std::string &filename) {
            tree.writeBinary(filename);
        })
        .def("search_bounding_box", [](const octomap::OcTree &tree, std::vector<double> min_point, std::vector<double> max_point) {
            octomap::point3d min = octomap::point3d(min_point[0], min_point[1], min_point[2]);
            octomap::point3d max = octomap::point3d(max_point[0], max_point[1], max_point[2]);
            std::vector<std::array<double, 5>> volume_points;
            for (octomap::OcTree::leaf_bbx_iterator it = tree.begin_leafs_bbx(min, max); it != tree.end_leafs_bbx(); ++it) {
                octomap::point3d point = it.getCoordinate();
                volume_points.push_back({point.x(), point.y(), point.z(), it->getOccupancy(), it.getSize()});
            }

            size_t element_size = sizeof(double);
            size_t shape[2] = {volume_points.size(), 5};
            size_t strides[2] = {5 * element_size, element_size};
            py::array_t<double> volume_points_array = py::array_t<double>(shape, strides);
            auto view = volume_points_array.mutable_unchecked<2>();
            for (int i = 0; i < volume_points.size(); i++) {
                view(i, 0) = volume_points[i][0];
                view(i, 1) = volume_points[i][1];
                view(i, 2) = volume_points[i][2];
                view(i, 3) = volume_points[i][3];
                view(i, 4) = volume_points[i][4];
            }
            return volume_points_array;
        })
        .def("updateInnerOccupancy", [](octomap::OcTree &tree) {
            tree.updateInnerOccupancy();
        })
        .def("insertPointCloudRays", [](octomap::OcTree &tree, std::vector<double> sensor_origin, py::array_t<double>  points, bool lazy_eval) {

            octomap::Pointcloud cloud;
            auto points_array = points.unchecked<2>();
            cloud.reserve(points_array.shape(0));
            for (int i = 0; i < points_array.shape(0); i++) {
                cloud.push_back(octomap::point3d(points_array(i, 0), points_array(i, 1), points_array(i, 2)));
            }
            tree.insertPointCloudRays(cloud, octomap::point3d(sensor_origin[0], sensor_origin[1], sensor_origin[2]), lazy_eval);
        })
        .def("insertPointCloud", [](octomap::OcTree &tree, std::vector<double> sensor_origin, py::array_t<double>  points, bool lazy_eval) {
            octomap::Pointcloud cloud;
            auto points_array = points.unchecked<2>();
            cloud.reserve(points_array.shape(0));
            for (int i = 0; i < points_array.shape(0); i++) {
                cloud.push_back(octomap::point3d(points_array(i, 0), points_array(i, 1), points_array(i, 2)));
                std::cout << "insertPointCloud" << points_array(i, 0) << " " << points_array(i, 1) << " " << points_array(i, 2) << std::endl;
            }
            std::cout << "insertPointCloud" << points_array.shape(0) << " and " << cloud.size() << std::endl;
            tree.insertPointCloud(cloud, octomap::point3d(sensor_origin[0], sensor_origin[1], sensor_origin[2]), lazy_eval);
        })
        .def("size", &octomap::OcTree::size);

    py::class_<octomap::SuperRayOcTree>(m, "SuperRayOcTree")
        .def(py::init<double>())
        .def("insertSuperRayCloudRays", [](octomap::SuperRayOcTree &tree, std::vector<double> sensor_origin, py::array_t<double>  points, bool lazy_eval) {
            octomap::Pointcloud cloud;
            auto points_array = points.unchecked<2>();
            cloud.reserve(points_array.shape(0));
            for (int i = 0; i < points_array.shape(0); i++) {
                cloud.push_back(octomap::point3d(points_array(i, 0), points_array(i, 1), points_array(i, 2)));
            }
            tree.insertSuperRayCloudRays(cloud, octomap::point3d(sensor_origin[0], sensor_origin[1], sensor_origin[2]), lazy_eval);
        })
        .def("write_binary", [](octomap::SuperRayOcTree &tree, const std::string &filename) {
            tree.writeBinary(filename);
        });
}