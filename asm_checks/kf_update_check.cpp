
#include "wet/estimation/kalman.hpp"
#include "wet/matrix/colvec.hpp"

using KF = wet::KalmanFilter<2, 1, 1, 2, 1, float>;

bool kf_update(KF& kf, const wet::ColVec<1, float>& y) {
    return kf.update(y);
}
