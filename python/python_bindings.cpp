#include "cuda_bundle_adjustment.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "object_creator.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(cuba, m) {
    m.doc() = R"pbdoc(
        CUDA Bundle Adjustment
        -----------------------
        .. currentmodule:: cuba
        .. autosummary::
           :toctree: _generate
           add
    )pbdoc";

    /* 
    Run bundle adjustment.

    Args:
        - camera_params: camera parameters as tuple (fx, fy, cx, cy)
        - poses_t: camera translations as (N, 3)
        - poses_q: camera rotations as (N, 4)
        - landmarks: 3D points as (N, 3)
        - edges: edges as (N, 2) containing the indices of the poses and landmarks
        - measurements: measurements as (N, 2) containing the 2D measurements
        - num_iterations: number of iterations for the optimization
    
    Returns:
        - optimized_poses_t: optimized camera translations as (N, 3)
        - optimized_poses_q: optimized camera rotations as (N, 4)
        - optimized_landmarks: optimized 3D points as (N, 3)
        - chi_squares: chi squares for each edge as (N,)
    */
    m.def("run_ba", [](
        std::tuple<double, double, double, double> camera_params,
        py::array_t<double> poses_t,
        py::array_t<double> poses_q,
        py::array_t<double> landmarks,
        py::array_t<int> edges,
        py::array_t<double> measurements,
        int num_iterations=10
    ) {
        // use memory manager for vertices and edges, since CudaBundleAdjustment doesn't delete those pointers
        cuba::ObjectCreator obj;

        // read camera parameters
        cuba::CameraParams camera;
        camera.fx = std::get<0>(camera_params);
        camera.fy = std::get<1>(camera_params);
        camera.cx = std::get<2>(camera_params);
        camera.cy = std::get<3>(camera_params);

        // create optimizer
        auto optimizer = cuba::CudaBundleAdjustment::create();

        // assert that poses_t and poses_q have the same shape
        if (poses_t.shape(0) != poses_q.shape(0)) {
            throw std::runtime_error("poses_t and poses_q must have the same shape");
        }

        // create pose vertices
        for (int poseIndex = 0; poseIndex < poses_t.shape(0); poseIndex++) {
            auto t = Eigen::Vector3d(poses_t.data(poseIndex));
            auto q = Eigen::Quaterniond(poses_q.data(poseIndex));
            auto v = obj.create<cuba::PoseVertex>(poseIndex, q, t, camera, poseIndex == 0);
            optimizer->addPoseVertex(v);
        }

        // create landmark vertices
        for (int landmarkIndex = 0; landmarkIndex < landmarks.shape(0); landmarkIndex++) {
            auto pos = Eigen::Vector3d(landmarks.data(landmarkIndex));
            auto v = obj.create<cuba::LandmarkVertex>(landmarkIndex, pos, false);
            optimizer->addLandmarkVertex(v);
        }

        // create monocular edges
        int edgeId;
        auto cubaEdges = std::vector<cuba::MonoEdge*>();

        try {
            for (edgeId = 0; edgeId < edges.shape(0); edgeId++) {
                auto edge = edges.data(edgeId);
                auto poseId = edge[0];
                auto landmarkId = edge[1];
                auto measurement = Eigen::Vector2d(measurements.data(edgeId));
                // std::cout << "Adding edge " << edgeId << " with pose " << poseId << " and landmark " << landmarkId << std::endl;
                auto vP = optimizer->poseVertex(poseId);
                auto vL = optimizer->landmarkVertex(landmarkId);
                auto e = obj.create<cuba::MonoEdge>(measurement, 1.0, vP, vL);
                optimizer->addMonocularEdge(e);
                cubaEdges.push_back(e);
            }
        } catch (const std::exception& e) {
            // Respond with invalid edge Id
            throw std::runtime_error("Invalid edge: " + std::to_string(edgeId) + ": " + e.what());
        }

        // optimize
        optimizer->initialize();
        optimizer->optimize(num_iterations);

        /* for (const auto& [name, value] : optimizer->timeProfile())
            std::printf("%-30s : %8.1f[msec]\n", name.c_str(), 1e3 * value);
        std::cout << std::endl;

        std::cout << "=== Objective function value : " << std::endl;
        for (const auto& stat : optimizer->batchStatistics())
            std::printf("iter: %2d, chi2: %.1f\n", stat.iteration + 1, stat.chi2); */

        // construct optimized poses_t and poses_q arrays
        py::size_t poses_t_shape[2] = {poses_t.shape(0), 3};
        py::size_t poses_t_stride[2] = {3 * sizeof(double), sizeof(double)};
        py::array_t<double> optimized_poses_t(poses_t_shape, poses_t_stride);
        
        py::size_t poses_q_shape[2] = {poses_q.shape(0), 4};
        py::size_t poses_q_stride[2] = {4 * sizeof(double), sizeof(double)};
        py::array_t<double> optimized_poses_q(poses_q_shape, poses_q_stride);

        for (int poseIndex = 0; poseIndex < poses_t.shape(0); poseIndex++) {
            auto v = optimizer->poseVertex(poseIndex);
            auto t = v->t;
            auto q = v->q;
            for (int i = 0; i < 3; i++) {
                optimized_poses_t.mutable_at(poseIndex, i) = t[i];
            }
            for (int i = 0; i < 4; i++) {
                optimized_poses_q.mutable_at(poseIndex, i) = q.coeffs()[i];
            }
        }

        // construct optimized landmarks array
        py::size_t landmarks_shape[2] = {landmarks.shape(0), 3};
        py::size_t landmarks_stride[2] = {3 * sizeof(double), sizeof(double)};
        py::array_t<double> optimized_landmarks(landmarks_shape, landmarks_stride);

        for (int landmarkIndex = 0; landmarkIndex < landmarks.shape(0); landmarkIndex++) {
            auto v = optimizer->landmarkVertex(landmarkIndex);
            auto pos = v->Xw;
            for (int i = 0; i < 3; i++) {
                optimized_landmarks.mutable_at(landmarkIndex, i) = pos[i];
            }
        }

        // construct chi_squares array
        py::size_t chi_squares_shape[1] = {edges.shape(0)};
        py::array_t<double> chi_squares(chi_squares_shape);

        for (int edgeId = 0; edgeId < edges.shape(0); edgeId++) {
            chi_squares.mutable_at(edgeId) = optimizer->chiSquared(cubaEdges[edgeId]);
        }

        return std::make_tuple(optimized_poses_t, optimized_poses_q, optimized_landmarks, chi_squares);
    });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
