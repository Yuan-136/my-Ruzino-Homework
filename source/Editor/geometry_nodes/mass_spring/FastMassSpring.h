#pragma once
#include <memory>

#include "MassSpring.h"

namespace USTC_CG::mass_spring {
// Impliment the Liu13's paper:
// https://tiantianliu.cn/papers/liu13fast/liu13fast.pdf
class FastMassSpring : public MassSpring {
   public:
    FastMassSpring() = default;
    ~FastMassSpring() = default;

    FastMassSpring(
        const Eigen::MatrixXd& X,
        const EdgeSet& E,
        const float stiffness,
        const float h);
    void step() override;
    unsigned max_iter = 100;  // (HW Optional) add UI for this parameter

   protected:
    Eigen::SparseMatrix<double> A;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::MatrixXd fixed_rhs_offset;
};
}  // namespace USTC_CG::mass_spring
