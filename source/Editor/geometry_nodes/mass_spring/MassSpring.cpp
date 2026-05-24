#include "MassSpring.h"

#include <algorithm>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace USTC_CG::mass_spring {
MassSpring::MassSpring(const Eigen::MatrixXd& X, const EdgeSet& E)
{
    this->X = this->init_X = X;
    this->vel = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    this->E = E;

    std::cout << "number of edges: " << E.size() << std::endl;
    std::cout << "init mass spring" << std::endl;

    // Compute the rest pose edge length
    for (const auto& e : E) {
        Eigen::Vector3d x0 = X.row(e.first);
        Eigen::Vector3d x1 = X.row(e.second);
        this->E_rest_length.push_back((x0 - x1).norm());
    }

    // Initialize the mask for Dirichlet boundary condition
    dirichlet_bc_mask.resize(X.rows(), false);

    // (HW_TODO) Fix two vertices, feel free to modify this
    unsigned n_fix = sqrt(X.rows());  // Here we assume the cloth is square
    dirichlet_bc_mask[0] = true;
    dirichlet_bc_mask[n_fix - 1] = true;
}

void MassSpring::step()
{
    Eigen::Vector3d acceleration_ext = gravity + wind_ext_acc;

    unsigned n_vertices = X.rows();

    // The reason to not use 1.0 as mass per vertex: the cloth gets heavier as
    // we increase the resolution
    double mass_per_vertex = mass / n_vertices;

    //----------------------------------------------------
    // (HW Optional) Bonus part: Sphere collision
    Eigen::MatrixXd acceleration_collision =
        getSphereCollisionForce(sphere_center.cast<double>(), sphere_radius) /
        mass_per_vertex;
    //----------------------------------------------------

    if (time_integrator == IMPLICIT_EULER) {
        // Implicit Euler
        TIC(step)

        Eigen::MatrixXd old_X = X;
        Eigen::MatrixXd Y = X + h * vel;
        Y.rowwise() += (h * h * acceleration_ext).transpose();
        if (enable_sphere_collision) {
            Y += h * h * acceleration_collision;
        }

        Eigen::SparseMatrix<double> A = computeHessianSparse(stiffness);
        double inertial_coeff = mass_per_vertex / (h * h);
        for (int i = 0; i < n_vertices * 3; i++) {
            A.coeffRef(i, i) += inertial_coeff;
        }
        A.makeCompressed();

        Eigen::MatrixXd grad_g =
            inertial_coeff * (X - Y) + computeGrad(stiffness);
        Eigen::MatrixXd rhs = -flatten(grad_g);

        std::vector<bool> fixed_dofs(n_vertices * 3, false);
        for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
            if (dirichlet_bc_mask[i]) {
                for (int axis = 0; axis < 3; axis++) {
                    int dof = 3 * i + axis;
                    fixed_dofs[dof] = true;
                    rhs(dof, 0) = 0.0;
                }
            }
        }

        std::vector<Trip_d> constrained_triplets;
        constrained_triplets.reserve(A.nonZeros() + n_vertices * 3);
        for (int k = 0; k < A.outerSize(); k++) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it;
                 ++it) {
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

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        Eigen::MatrixXd delta_X = unflatten(solver.solve(rhs));
        X = old_X + delta_X;
        vel = (X - old_X) / h;

        for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
            if (dirichlet_bc_mask[i]) {
                X.row(i) = old_X.row(i);
                vel.row(i).setZero();
            }
        }

        TOC(step)
        printDebugStats("implicit_euler");
    }
    else if (time_integrator == SEMI_IMPLICIT_EULER) {
        // Semi-implicit Euler
        TIC(step)

        Eigen::MatrixXd acceleration =
            -computeGrad(stiffness) / mass_per_vertex;
        acceleration.rowwise() += acceleration_ext.transpose();

        // -----------------------------------------------
        // (HW Optional)
        if (enable_sphere_collision) {
            acceleration += acceleration_collision;
        }
        // -----------------------------------------------

        // (HW TODO): Implement semi-implicit Euler time integration
        Eigen::MatrixXd old_X = X;
        for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
            if (dirichlet_bc_mask[i]) {
                acceleration.row(i).setZero();
            }
        }

        vel += h * acceleration;
        if (enable_damping) {
            vel *= damping;
        }
        X += h * vel;

        for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
            if (dirichlet_bc_mask[i]) {
                X.row(i) = old_X.row(i);
                vel.row(i).setZero();
            }
        }

        TOC(step)
        printDebugStats("semi_implicit_euler");
    }
    else {
        std::cerr << "Unknown time integrator!" << std::endl;
        return;
    }
}

// There are different types of mass spring energy:
// For this homework we will adopt Prof. Huamin Wang's energy definition
// introduced in GAMES103 course Lecture 2 E = 0.5 * stiffness * sum_{i=1}^{n}
// (||x_i - x_j|| - l)^2 There exist other types of energy definition, e.g.,
// Prof. Minchen Li's energy definition
// https://www.cs.cmu.edu/~15769-f23/lec/3_Mass_Spring_Systems.pdf
double MassSpring::computeEnergy(double stiffness)
{
    double sum = 0.;
    const std::vector<Edge> edges(E.begin(), E.end());

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum) if (edges.size() > 64)
#endif
    for (int i = 0; i < static_cast<int>(edges.size()); i++) {
        const auto& e = edges[i];
        auto diff = X.row(e.first) - X.row(e.second);
        auto l = E_rest_length[i];
        sum += 0.5 * stiffness * std::pow((diff.norm() - l), 2);
    }
    return sum;
}

Eigen::MatrixXd MassSpring::computeGrad(double stiffness)
{
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    const std::vector<Edge> edges(E.begin(), E.end());

#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> local_grads(
        thread_count, Eigen::MatrixXd::Zero(X.rows(), X.cols()));

#pragma omp parallel if (edges.size() > 64)
    {
        int thread_id = omp_get_thread_num();
        auto& local_g = local_grads[thread_id];

#pragma omp for
        for (int i = 0; i < static_cast<int>(edges.size()); i++) {
            const auto& e = edges[i];
            Eigen::Vector3d diff = X.row(e.first) - X.row(e.second);
            double len = diff.norm();
            double rest_len = E_rest_length[i];

            if (len > 1e-8) {
                Eigen::Vector3d grad =
                    stiffness * (len - rest_len) * diff / len;
                local_g.row(e.first) += grad.transpose();
                local_g.row(e.second) -= grad.transpose();
            }
        }
    }

    for (const auto& local_g : local_grads) {
        g += local_g;
    }
#else
    for (int i = 0; i < static_cast<int>(edges.size()); i++) {
        const auto& e = edges[i];
        // --------------------------------------------------
        // (HW TODO): Implement the gradient computation
        Eigen::Vector3d diff = X.row(e.first) - X.row(e.second);
        double len = diff.norm();
        double rest_len = E_rest_length[i];

        if (len > 1e-8) {
            Eigen::Vector3d grad =
                stiffness * (len - rest_len) * diff / len;
            g.row(e.first) += grad.transpose();
            g.row(e.second) -= grad.transpose();
        }

        // --------------------------------------------------
    }
#endif
    return g;
}

Eigen::SparseMatrix<double> MassSpring::computeHessianSparse(double stiffness)
{
    unsigned n_vertices = X.rows();
    Eigen::SparseMatrix<double> H(n_vertices * 3, n_vertices * 3);

    auto k = stiffness;
    const auto I = Eigen::Matrix3d::Identity();
    const std::vector<Edge> edges(E.begin(), E.end());
    std::vector<Trip_d> triplets;
    triplets.reserve(E.size() * 36);

    auto add_edge_hessian = [&](std::vector<Trip_d>& target,
                                const Edge& e,
                                int edge_id) {
        Eigen::Vector3d diff = X.row(e.first) - X.row(e.second);
        double len = diff.norm();
        double rest_len = E_rest_length[edge_id];

        if (len > 1e-8) {
            Eigen::Vector3d dir = diff / len;
            Eigen::Matrix3d outer = dir * dir.transpose();
            Eigen::Matrix3d H_e;

            if (rest_len > len) {
                H_e = k * outer;
            }
            else {
                H_e = k * (outer + (1.0 - rest_len / len) * (I - outer));
            }

            int v0 = e.first;
            int v1 = e.second;
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    double value = H_e(r, c);
                    int row0 = 3 * v0 + r;
                    int col0 = 3 * v0 + c;
                    int row1 = 3 * v1 + r;
                    int col1 = 3 * v1 + c;

                    target.emplace_back(row0, col0, value);
                    target.emplace_back(row1, col1, value);
                    target.emplace_back(row0, col1, -value);
                    target.emplace_back(row1, col0, -value);
                }
            }
        }
    };

#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
    std::vector<std::vector<Trip_d>> local_triplets(thread_count);

#pragma omp parallel if (edges.size() > 64)
    {
        int thread_id = omp_get_thread_num();
        auto& local = local_triplets[thread_id];
        local.reserve(edges.size() * 36 / thread_count + 36);

#pragma omp for
        for (int i = 0; i < static_cast<int>(edges.size()); i++) {
            add_edge_hessian(local, edges[i], i);
        }
    }

    for (auto& local : local_triplets) {
        triplets.insert(triplets.end(), local.begin(), local.end());
    }
#else
    for (int i = 0; i < static_cast<int>(edges.size()); i++) {
        // --------------------------------------------------
        // (HW TODO): Implement the sparse version Hessian computation
        // Remember to consider fixed points
        // You can also consider positive definiteness here
        add_edge_hessian(triplets, edges[i], i);
        // --------------------------------------------------
    }
#endif

    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();
    return H;
}

bool MassSpring::checkSPD(const Eigen::SparseMatrix<double>& A)
{
    // Eigen::SimplicialLDLT<SparseMatrix_d> ldlt(A);
    // return ldlt.info() == Eigen::Success;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    auto eigen_values = es.eigenvalues();
    return eigen_values.minCoeff() >= 1e-10;
}

void MassSpring::reset()
{
    std::cout << "reset" << std::endl;
    this->X = this->init_X;
    this->vel.setZero();
}

void MassSpring::printDebugStats(const char* solver_name)
{
    if (!enable_debug_output) {
        return;
    }

    double max_velocity = 0.0;
    double avg_velocity = 0.0;
    double max_displacement = 0.0;
    if (X.rows() > 0) {
        Eigen::VectorXd speeds = vel.rowwise().norm();
        max_velocity = speeds.maxCoeff();
        avg_velocity = speeds.mean();
        max_displacement = (X - init_X).rowwise().norm().maxCoeff();
    }

    double max_edge_strain = 0.0;
    double sum_edge_strain = 0.0;
    int valid_edge_count = 0;
    const std::vector<Edge> edges(E.begin(), E.end());

    for (int i = 0; i < static_cast<int>(edges.size()); i++) {
        const double rest_len = E_rest_length[i];
        if (rest_len <= 1e-8) {
            continue;
        }
        const auto& e = edges[i];
        const double len = (X.row(e.first) - X.row(e.second)).norm();
        const double strain = std::abs(len - rest_len) / rest_len;
        max_edge_strain = std::max(max_edge_strain, strain);
        sum_edge_strain += strain;
        valid_edge_count++;
    }

    int fixed_vertices = 0;
    for (bool is_fixed : dirichlet_bc_mask) {
        if (is_fixed) {
            fixed_vertices++;
        }
    }

    int openmp_threads = 1;
#ifdef _OPENMP
    openmp_threads = omp_get_max_threads();
#endif

    std::cout << "[MassSpring][" << solver_name << "] "
              << "vertices=" << X.rows() << " edges=" << E.size()
              << " energy=" << computeEnergy(stiffness)
              << " max_velocity=" << max_velocity
              << " avg_velocity=" << avg_velocity
              << " max_displacement=" << max_displacement
              << " max_edge_strain=" << max_edge_strain
              << " avg_edge_strain="
              << (valid_edge_count > 0 ? sum_edge_strain / valid_edge_count
                                       : 0.0)
              << " fixed_vertices=" << fixed_vertices
              << " openmp_threads=" << openmp_threads << std::endl;
}

// ----------------------------------------------------------------------------------
// (HW Optional) Bonus part
Eigen::MatrixXd MassSpring::getSphereCollisionForce(
    Eigen::Vector3d center,
    double radius)
{
    Eigen::MatrixXd force = Eigen::MatrixXd::Zero(X.rows(), X.cols());
#ifdef _OPENMP
#pragma omp parallel for if (X.rows() > 64)
#endif
    for (int i = 0; i < X.rows(); i++) {
        // (HW Optional) Implement penalty-based force here
        Eigen::Vector3d diff = X.row(i).transpose() - center;
        double dist = diff.norm();
        double penetration = collision_scale_factor * radius - dist;

        if (penetration > 0.0 && dist > 1e-8) {
            Eigen::Vector3d direction = diff / dist;
            force.row(i) =
                (collision_penalty_k * penetration * direction).transpose();
        }
    }
    return force;
}
// ----------------------------------------------------------------------------------

bool MassSpring::set_dirichlet_bc_mask(const std::vector<bool>& mask)
{
    if (mask.size() == X.rows()) {
        dirichlet_bc_mask = mask;
        return true;
    }
    else
        return false;
}

bool MassSpring::update_dirichlet_bc_vertices(const MatrixXd& control_vertices)
{
    for (int i = 0; i < dirichlet_bc_control_pair.size(); i++) {
        int idx = dirichlet_bc_control_pair[i].first;
        int control_idx = dirichlet_bc_control_pair[i].second;
        X.row(idx) = control_vertices.row(control_idx);
    }

    return true;
}

bool MassSpring::init_dirichlet_bc_vertices_control_pair(
    const MatrixXd& control_vertices,
    const std::vector<bool>& control_mask)
{
    if (control_mask.size() != control_vertices.rows())
        return false;

    // TODO: optimize this part from O(n) to O(1)
    // First, get selected_control_vertices
    std::vector<VectorXd> selected_control_vertices;
    std::vector<int> selected_control_idx;
    for (int i = 0; i < control_mask.size(); i++) {
        if (control_mask[i]) {
            selected_control_vertices.push_back(control_vertices.row(i));
            selected_control_idx.push_back(i);
        }
    }

    // Then update mass spring fixed vertices
    for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
        if (dirichlet_bc_mask[i]) {
            // O(n^2) nearest point search, can be optimized
            // -----------------------------------------
            int nearest_idx = 0;
            double nearst_dist = 1e6;
            VectorXd X_i = X.row(i);
            for (int j = 0; j < selected_control_vertices.size(); j++) {
                double dist = (X_i - selected_control_vertices[j]).norm();
                if (dist < nearst_dist) {
                    nearst_dist = dist;
                    nearest_idx = j;
                }
            }
            //-----------------------------------------

            X.row(i) = selected_control_vertices[nearest_idx];
            dirichlet_bc_control_pair.push_back(
                std::make_pair(i, selected_control_idx[nearest_idx]));
        }
    }

    return true;
}

}  // namespace USTC_CG::mass_spring
