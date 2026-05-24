#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <set>
#include <vector>
#include "glm/glm.hpp"

#include "pxr/usd/usdGeom/xform.h"

namespace USTC_CG::mass_spring {

inline auto flatten = [&](const Eigen::MatrixXd& A) {
    Eigen::MatrixXd A_flatten = A.transpose();
    A_flatten.resize(A.rows() * A.cols(), 1);
    return A_flatten;
};
inline auto unflatten = [&](const Eigen::MatrixXd& A_flatten) {
    Eigen::MatrixXd A = A_flatten;
    A.resize(3, A_flatten.rows() / 3);
    A.transposeInPlace();
    return A;
};

inline Eigen::MatrixXi usd_faces_to_eigen(
    const std::vector<int>& faceVertexCount,
    const std::vector<int>& faceVertexIndices)
{
    unsigned nTriangles = 0;
    for (int n : faceVertexCount) {
        if (n >= 3) {
            nTriangles += n - 2;
        }
    }

    Eigen::MatrixXi F(nTriangles, 3);
    unsigned index_offset = 0;
    unsigned triangle_id = 0;
    for (int n : faceVertexCount) {
        for (int j = 1; j + 1 < n; j++) {
            F(triangle_id, 0) = faceVertexIndices[index_offset];
            F(triangle_id, 1) = faceVertexIndices[index_offset + j];
            F(triangle_id, 2) = faceVertexIndices[index_offset + j + 1];
            triangle_id++;
        }
        index_offset += n;
    }
    return F;
}

inline Eigen::MatrixXd usd_vertices_to_eigen(
    const std::vector<glm::vec3>& v)
{
    unsigned nVertices = v.size();
    Eigen::MatrixXd V(nVertices, 3);
    for (int i = 0; i < nVertices; i++) {
        for (int j = 0; j < 3; j++) {
            V(i, j) = v[i][j];
        }
    }
    return V;
}

inline std::vector<glm::vec3> eigen_to_usd_vertices(
    const Eigen::MatrixXd& V)
{
    std::vector<glm::vec3> vertices;
    for (int i = 0; i < V.rows(); i++) {
        vertices.push_back(glm::vec3(V(i, 0), V(i, 1), V(i, 2)));
    }
    return vertices;
}

using Edge = std::pair<int, int>;
using EdgeSet = std::set<Edge>;
// Here F is of shape [nFaces, 3] for triangular mesh
inline EdgeSet get_edges(const Eigen::MatrixXi& F)
{
    EdgeSet edges;
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < F.cols(); j++) {
            int v0 = F(i, j);
            int v1 = F(i, (j + 1) % F.cols());
            if (v0 > v1) {
                std::swap(v0, v1);
            }
            edges.insert(std::make_pair(v0, v1));
        }
    }
    return edges;
}

inline std::vector<bool> VtIntArray_to_vector_bool(const pxr::VtArray<float>& v)
{
    std::vector<bool> mask;
    for (int i = 0; i < v.size(); i++) {
        mask.push_back(v[i] > 0);
    }
    return mask;
}

}  // namespace USTC_CG::mass_spring
