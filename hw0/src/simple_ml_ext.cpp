#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t num_batches = (m + batch - 1) / batch;
    std::vector<float> grads_theta(n * k);
    std::vector<float> logits(batch * k);

    for (size_t i = 0; i < num_batches; i++)
    {
        std::fill(logits.begin(), logits.end(), 0.0f);
        size_t start = i * batch;
        size_t end = std::min(start + batch, m);

        for (size_t j = start; j < end; j++)
        {
            for (size_t c = 0; c < k; c++)
            {
                for (size_t p = 0; p < n; p++)
                {
                    logits[(j - start) * k + c] += X[j * n + p] * theta[p * k + c];
                }
            }
        }

        for (size_t j = 0; j < batch; j++)
        {
            float sum_exp = 0;
            for (size_t c = 0; c < k; c++)
            {
                logits[j * k + c] = exp(logits[j * k + c]);
                sum_exp += logits[j * k + c];
            }

            for (size_t c = 0; c < k; c++)
            {
                logits[j * k + c] /= sum_exp;
            }
        }

        std::fill(grads_theta.begin(), grads_theta.end(), 0.0f);
        for (size_t j = 0; j < batch; j++)
        {
            for (size_t c = 0; c < k; c++)
            {
                float diff = logits[j * k + c] - (y[start + j] == c ? 1.0f : 0.0f);
                for (size_t p = 0; p < n; p++)
                {
                    grads_theta[p * k + c] += X[(start + j) * n + p] * diff;
                }
            }
        }

        for (size_t p = 0; p < n; p++)
        {
            for (size_t c = 0; c < k; c++)
            {
                theta[p * k + c] -= lr * grads_theta[p * k + c] / batch;
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
