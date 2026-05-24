#include "FastMassSpring.h"

#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace USTC_CG::mass_spring {
FastMassSpring::FastMassSpring(
    const Eigen::MatrixXd& X,
    const EdgeSet& E,
    const float stiffness,
    const float h)
    : MassSpring(X, E)
{
    // construct L and J at initialization
    std::cout << "init fast mass spring" << std::endl;

    unsigned n_vertices = X.rows();
    this->stiffness = stiffness;
    this->h = h;

    double mass_per_vertex = mass / n_vertices;
    A.resize(n_vertices * 3, n_vertices * 3);
    A.setZero();

    // (HW Optional) precompute A and prefactorize
    // Note: one thing to take care of: A is related with stiffness, if
    // stiffness changes, A need to be recomputed
    std::vector<Trip_d> triplets;
    triplets.reserve(n_vertices * 3 + E.size() * 36);

    for (int i = 0; i < n_vertices * 3; i++) {
        triplets.emplace_back(i, i, mass_per_vertex);
    }

    const double spring_coeff = h * h * stiffness;
    for (const auto& e : E) {
        int v0 = e.first;
        int v1 = e.second;
        for (int axis = 0; axis < 3; axis++) {
            int dof0 = 3 * v0 + axis;
            int dof1 = 3 * v1 + axis;
            triplets.emplace_back(dof0, dof0, spring_coeff);
            triplets.emplace_back(dof1, dof1, spring_coeff);
            triplets.emplace_back(dof0, dof1, -spring_coeff);
            triplets.emplace_back(dof1, dof0, -spring_coeff);
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    std::vector<bool> fixed_dofs(n_vertices * 3, false);
    for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
        if (dirichlet_bc_mask[i]) {
            for (int axis = 0; axis < 3; axis++) {
                fixed_dofs[3 * i + axis] = true;
            }
        }
    }

    Eigen::MatrixXd fixed_rhs_offset_flat =
        Eigen::MatrixXd::Zero(n_vertices * 3, 1);
    Eigen::MatrixXd X_flat = flatten(this->X);
    for (int k = 0; k < A.outerSize(); k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            if (!fixed_dofs[it.row()] && fixed_dofs[it.col()]) {
                fixed_rhs_offset_flat(it.row(), 0) +=
                    it.value() * X_flat(it.col(), 0);
            }
        }
    }
    fixed_rhs_offset = unflatten(fixed_rhs_offset_flat);

    std::vector<Trip_d> constrained_triplets;
    constrained_triplets.reserve(A.nonZeros() + n_vertices * 3);
    for (int k = 0; k < A.outerSize(); k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            if (!fixed_dofs[it.row()] && !fixed_dofs[it.col()]) {
                constrained_triplets.emplace_back(
                    it.row(), it.col(), it.value());
            }
        }
    }
    for (int dof = 0; dof < fixed_dofs.size(); dof++) {
        if (fixed_dofs[dof]) {
            constrained_triplets.emplace_back(dof, dof, 1.0);
        }
    }

    A.setZero();
    A.setFromTriplets(
        constrained_triplets.begin(), constrained_triplets.end());
    A.makeCompressed();
    solver.compute(A);
}

void FastMassSpring::step()
{
    TIC(step)

    Eigen::Vector3d acceleration_ext = gravity + wind_ext_acc;
    unsigned n_vertices = X.rows();
    double mass_per_vertex = mass / n_vertices;

    Eigen::MatrixXd old_X = X;
    Eigen::MatrixXd Y = X + h * vel;
    Y.rowwise() += (h * h * acceleration_ext).transpose();

    if (enable_sphere_collision) {
        Y += h * h *
             getSphereCollisionForce(
                 sphere_center.cast<double>(),
                 sphere_radius) /
                 mass_per_vertex;
    }

    Eigen::MatrixXd rhs_base = mass_per_vertex * Y;
    const std::vector<Edge> edges(E.begin(), E.end());
    for (unsigned iter = 0; iter < max_iter; iter++) {
        // (HW Optional)
        // local_step and global_step alternating solving
        Eigen::MatrixXd rhs = rhs_base;

#ifdef _OPENMP
        const int thread_count = omp_get_max_threads();
        std::vector<Eigen::MatrixXd> local_rhs(
            thread_count, Eigen::MatrixXd::Zero(n_vertices, X.cols()));

#pragma omp parallel if (edges.size() > 64)
        {
            int thread_id = omp_get_thread_num();
            auto& local = local_rhs[thread_id];

#pragma omp for
            for (int edge_id = 0; edge_id < static_cast<int>(edges.size());
                 edge_id++) {
                const auto& e = edges[edge_id];
                Eigen::Vector3d diff = X.row(e.first) - X.row(e.second);
                double len = diff.norm();
                if (len > 1e-8) {
                    Eigen::Vector3d d = E_rest_length[edge_id] * diff / len;
                    Eigen::Vector3d contribution = h * h * stiffness * d;
                    local.row(e.first) += contribution.transpose();
                    local.row(e.second) -= contribution.transpose();
                }
            }
        }

        for (const auto& local : local_rhs) {
            rhs += local;
        }
#else
        for (int edge_id = 0; edge_id < static_cast<int>(edges.size());
             edge_id++) {
            const auto& e = edges[edge_id];
            Eigen::Vector3d diff = X.row(e.first) - X.row(e.second);
            double len = diff.norm();
            if (len > 1e-8) {
                Eigen::Vector3d d = E_rest_length[edge_id] * diff / len;
                Eigen::Vector3d contribution = h * h * stiffness * d;
                rhs.row(e.first) += contribution.transpose();
                rhs.row(e.second) -= contribution.transpose();
            }
        }
#endif

        rhs -= fixed_rhs_offset;
        for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
            if (dirichlet_bc_mask[i]) {
                rhs.row(i) = old_X.row(i);
            }
        }

        X = unflatten(solver.solve(flatten(rhs)));
    }

    vel = (X - old_X) / h;
    if (enable_damping) {
        vel *= damping;
    }

    for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
        if (dirichlet_bc_mask[i]) {
            X.row(i) = old_X.row(i);
            vel.row(i).setZero();
        }
    }

    TOC(step)
    printDebugStats("liu13");
}

}  // namespace USTC_CG::mass_spring
