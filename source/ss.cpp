#include "ss.hpp"

#include "LTI.hpp"
#include "control.hpp"
#include "tf.hpp"
#include "types.hpp"
#include "zpk.hpp"

namespace control {

static void validateStateSpaceMatrices(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("StateSpace: A must be square");
    }
    if (B.rows() != A.rows()) {
        throw std::invalid_argument("StateSpace: B.rows() must match A.rows()");
    }
    if (C.cols() != A.cols()) {
        throw std::invalid_argument("StateSpace: C.cols() must match A.cols()");
    }
    if (D.rows() != C.rows() || D.cols() != B.cols()) {
        throw std::invalid_argument("StateSpace: D shape must be (C.rows(), B.cols())");
    }
}

static void validateNoiseInputMatrices(const Matrix& A, const Matrix& C, const Matrix& G, const Matrix& H) {
    if (G.rows() != A.rows()) {
        throw std::invalid_argument("StateSpace: G.rows() must match A.rows() (number of states)");
    }
    if (H.rows() != C.rows()) {
        throw std::invalid_argument("StateSpace: H.rows() must match C.rows() (number of outputs)");
    }
    if (G.cols() != H.cols()) {
        throw std::invalid_argument("StateSpace: G.cols() must match H.cols() (noise dimension)");
    }
}

StateSpace::StateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, std::optional<double> Ts)
    : A(A), B(B), C(C), D(D) {
    validateStateSpaceMatrices(A, B, C, D);
    this->Ts = Ts;
    // Default G and H: G = I (process noise affects all states), H = 0 (no measurement noise input)
    G = Matrix::Identity(A.rows(), A.rows());
    H = Matrix::Zero(C.rows(), A.rows());
}

StateSpace::StateSpace(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D, std::optional<double> Ts)
    : A(std::move(A)), B(std::move(B)), C(std::move(C)), D(std::move(D)) {
    validateStateSpaceMatrices(this->A, this->B, this->C, this->D);
    this->Ts = Ts;
    // Default G and H: G = I (process noise affects all states), H = 0 (no measurement noise input)
    G = Matrix::Identity(this->A.rows(), this->A.rows());
    H = Matrix::Zero(this->C.rows(), this->A.rows());
}

StateSpace::StateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, const Matrix& G, const Matrix& H, std::optional<double> Ts)
    : A(A), B(B), C(C), D(D), G(G), H(H) {
    validateStateSpaceMatrices(A, B, C, D);
    validateNoiseInputMatrices(A, C, G, H);
    this->Ts = Ts;
}

StateSpace::StateSpace(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D, Matrix&& G, Matrix&& H, std::optional<double> Ts)
    : A(std::move(A)), B(std::move(B)), C(std::move(C)), D(std::move(D)), G(std::move(G)), H(std::move(H)) {
    validateStateSpaceMatrices(this->A, this->B, this->C, this->D);
    validateNoiseInputMatrices(this->A, this->C, this->G, this->H);
    this->Ts = Ts;
}

StateSpace::StateSpace(const TransferFunction& tf)
    : StateSpace(tf.toStateSpace()) {}

StateSpace::StateSpace(TransferFunction&& tf) noexcept
    : StateSpace(std::move(tf).toStateSpace()) {}

StateSpace::StateSpace(const StateSpace& other)
    : A(other.A), B(other.B), C(other.C), D(other.D), G(other.G), H(other.H) {
    this->Ts = other.Ts;
}

StateSpace::StateSpace(StateSpace&& other) noexcept
    : A(std::move(other.A)), B(std::move(other.B)), C(std::move(other.C)), D(std::move(other.D)), G(std::move(other.G)), H(std::move(other.H)) {
    this->Ts = other.Ts;
}

// Copy assignment
StateSpace& StateSpace::operator=(const StateSpace& other) {
    if (this != &other) {
        A  = other.A;
        B  = other.B;
        C  = other.C;
        D  = other.D;
        G  = other.G;
        H  = other.H;
        Ts = other.Ts;
    }
    return *this;
}

// Move assignment
StateSpace& StateSpace::operator=(StateSpace&& other) noexcept {
    if (this != &other) {
        A  = std::move(other.A);
        B  = std::move(other.B);
        C  = std::move(other.C);
        D  = std::move(other.D);
        G  = std::move(other.G);
        H  = std::move(other.H);
        Ts = other.Ts;
    }
    return *this;
}

// Assignment from TransferFunction
StateSpace& StateSpace::operator=(const TransferFunction& tf) {
    *this = tf.toStateSpace();
    return *this;
}

// Move assignment from TransferFunction
StateSpace& StateSpace::operator=(TransferFunction&& tf) noexcept {
    *this = std::move(tf).toStateSpace();
    return *this;
}

bool StateSpace::is_stable() const {
    return control::is_stable(*this);
}

std::vector<Pole> StateSpace::poles() const {
    return control::poles(*this);
}

std::vector<Zero> StateSpace::zeros() const {
    return control::zeros(*this);
}

MarginInfo StateSpace::margin() const {
    return control::margin(*this);
}

DampingInfo StateSpace::damp() const {
    return control::damp(*this);
}

StepInfo StateSpace::stepinfo() const {
    return control::stepinfo(*this);
}

StepResponse StateSpace::step(double tStart, double tEnd, ColVec uStep) const {
    return control::step(*this, tStart, tEnd, uStep);
}

ImpulseResponse StateSpace::impulse(double tStart, double tEnd) const {
    return control::impulse(*this, tStart, tEnd);
}

BodeResponse StateSpace::bode(double fStart, double fEnd, size_t maxPoints) const {
    return control::bode(*this, fStart, fEnd, maxPoints);
}

NyquistResponse StateSpace::nyquist(double fStart, double fEnd, size_t maxPoints) const {
    return control::nyquist(*this, fStart, fEnd, maxPoints);
}

FrequencyResponse StateSpace::freqresp(const std::vector<double>& frequencies) const {
    return control::freqresp(*this, frequencies);
}

RootLocusResponse StateSpace::rlocus(double kMin, double kMax, size_t numPoints) const {
    return control::rlocus(*this, kMin, kMax, numPoints);
}

// Discretize (only if continuous)
StateSpace StateSpace::discretize(double Ts, DiscretizationMethod method, std::optional<double> prewarp) const {
    return control::c2d(*this, Ts, method, prewarp);
}

ObservabilityInfo StateSpace::observability() const {
    return control::observability(*this);
}

ControllabilityInfo StateSpace::controllability() const {
    return control::controllability(*this);
}

// Compute Gramian matrices for continuous-time systems using iterative series
Matrix StateSpace::gramian(GramianType type) const {
    return control::gramian(*this, type);
}

/**
 * @brief Extract a SISO transfer function from a MIMO StateSpace system.
 *
 * For MIMO systems, extracts the transfer function from a specific input to a specific output.
 * This creates a SISO subsystem: G_ij(s) = C_i(sI-A)^(-1)B_j + D_ij
 * where i is the output index and j is the input index (0-based).
 *
 * @param output_idx   Output index (0-based, must be < number of outputs)
 * @param input_idx    Input index (0-based, must be < number of inputs)
 * @return TransferFunction  Transfer function from input_idx to output_idx
 * @throws std::out_of_range if indices are out of bounds
 */
TransferFunction StateSpace::toTransferFunction(int output_idx, int input_idx) const {
    // Validate indices
    int num_outputs = C.rows();
    int num_inputs  = B.cols();

    if (output_idx < 0 || output_idx >= num_outputs) {
        throw std::out_of_range("Output index " + std::to_string(output_idx) +
                                " is out of range [0, " + std::to_string(num_outputs - 1) + "]");
    }

    if (input_idx < 0 || input_idx >= num_inputs) {
        throw std::out_of_range("Input index " + std::to_string(input_idx) +
                                " is out of range [0, " + std::to_string(num_inputs - 1) + "]");
    }

    int n = A.rows();  // State dimension

    // Extract SISO subsystem: C_row(i), B_col(j), D(i,j)
    RowVec C_i  = C.row(output_idx);
    ColVec B_j  = B.col(input_idx);
    double D_ij = D(output_idx, input_idx);

    // For very simple cases, handle directly
    if (n == 0) {
        // Pure gain system (no states)
        return TransferFunction({D_ij}, {1.0}, Ts);
    }

    // Compute transfer function using Faddeev-LeVerrier algorithm
    // This is numerically stable for high-order systems

    // Step 1: Compute characteristic polynomial using Faddeev-LeVerrier
    // det(sI - A) = s^n + a_{n-1}s^{n-1} + ... + a_0
    std::vector<double> p(n + 1, 0.0);  // p[0] = 0
    std::vector<Matrix> H(n + 1);
    H[0] = Matrix::Identity(n, n);

    for (int k = 1; k <= n; ++k) {
        H[k] = A * H[k - 1];
        p[k] = H[k].trace() / static_cast<double>(k);
    }

    // Build denominator polynomial: s^n + a_{n-1}s^{n-1} + ... + a_0
    std::vector<double> den(n + 1);
    den[0] = 1.0;  // Leading coefficient s^n
    for (int i = 1; i <= n; ++i) {
        den[i] = (i % 2 == 1 ? -p[i] : p[i]);
    }

    // Step 2: Compute numerator coefficients
    // For the transfer function G(s) = [C*adj(sI-A)*B + D*det(sI-A)] / det(sI-A)
    std::vector<double> num(n + 1, 0.0);

    // Add D*det(sI-A) term
    for (int i = 0; i <= n; ++i) {
        num[i] += D_ij * den[i];
    }

    // Add C*adj(sI-A)*B term using the Faddeev-LeVerrier matrices
    // The adjoint matrix adj(sI-A) = sum_{k=0}^{n-1} s^{n-1-k} * F_k
    // where F_k satisfy a similar recurrence
    if (n > 0) {
        std::vector<Matrix> F(n);
        F[0] = Matrix::Identity(n, n);

        for (int k = 1; k < n; ++k) {
            F[k] = A * F[k - 1] - p[k] * Matrix::Identity(n, n);
        }

        // Compute C * F_k * B for each k and add to appropriate power of s
        for (int k = 0; k < n; ++k) {
            double coeff = C_i.dot(F[k] * B_j);
            num[k + 1] += coeff;
        }
    }

    // Normalize denominator to have leading coefficient 1
    double den_leading = den[0];
    if (std::abs(den_leading) > 1e-10) {
        for (double& coeff : den) {
            coeff /= den_leading;
        }
        for (double& coeff : num) {
            coeff /= den_leading;
        }
    }

    // Remove leading zeros from numerator (but keep at least one coefficient)
    while (num.size() > 1 && std::abs(num[0]) < 1e-10) {
        num.erase(num.begin());
    }

    // Create and return transfer function
    TransferFunction result(num, den, Ts);
    return result;
}

StateSpace StateSpace::toStateSpace() const {
    return *this;
}

ZeroPoleGain StateSpace::toZeroPoleGain() const {
    // Convert via SS→TF→ZPK path
    TransferFunction tf_sys = this->toTransferFunction();
    return tf_sys.toZeroPoleGain();
}

StateSpace StateSpace::balred(size_t r) const {
    return control::balred(*this, r);
}

StateSpace StateSpace::minreal(double tol) const {
    return control::minreal(*this, tol);
}

StateSpace StateSpace::balreal(size_t r) const {
    return balred(r);
}

}  // namespace control