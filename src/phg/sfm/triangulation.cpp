#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    int a_r = 2 * count;
    int a_c = 4;

    Eigen::MatrixXd A(a_r, a_c);

    for (int i = 0 ; i< count; i++) {
        auto A_2i   = (Ps[i].row(2) * ms[i](0) - Ps[i].row(0) * ms[i](2));
        auto A_2i_1 = (Ps[i].row(2) * ms[i](1) - Ps[i].row(1) * ms[i](2));

        for (int j = 0; j < 4; j++) {
            A.row(2*i)[j] = A_2i(j);
            A.row(2*i+1)[j] = A_2i_1(j);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::VectorXd null_space = svda.matrixV().col(a_c - 1);

    cv::Vec4d result;

    for (int j = 0; j < 4; j++) result[j] = null_space[j];

    return result;
}
