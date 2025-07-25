
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>

using namespace sycl;

void initialize_tensor(std::vector<float>& tensor, float value) {
    std::fill(tensor.begin(), tensor.end(), value);
}

// Corrected Depthwise Convolution
void depthwise_conv(queue& q, int C, int H, int W, int KH, int KW, int pad, int stride) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;

    std::vector<float> input(C * H * W);
    std::vector<float> kernel(C * KH * KW);
    std::vector<float> output(C * OH * OW, 0);

    initialize_tensor(input, 1.0f);
    initialize_tensor(kernel, 1.0f);

    buffer<float, 1> input_buf(input.data(), range<1>(input.size()));
    buffer<float, 1> kernel_buf(kernel.data(), range<1>(kernel.size()));
    buffer<float, 1> output_buf(output.data(), range<1>(output.size()));

    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](handler& h) {
        auto in = input_buf.get_access<access::mode::read>(h);
        auto k = kernel_buf.get_access<access::mode::read>(h);
        auto out = output_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<3>(C, OH, OW), [=](id<3> idx) {
            int c = idx[0];
            int h_out = idx[1];
            int w_out = idx[2];

            float acc = 0.0f;
            for (int i = 0; i < KH; ++i) {
                for (int j = 0; j < KW; ++j) {
                    int h_in = h_out * stride + i - pad;
                    int w_in = w_out * stride + j - pad;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        float val = in[c * H * W + h_in * W + w_in];
                        float weight = k[c * KH * KW + i * KW + j];
                        acc += val * weight;
                    }
                }
            }
            out[c * OH * OW + h_out * OW + w_out] = acc;
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    double duration_us = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Depthwise convolution took " << duration_us << " us\n";

    std::ofstream result("depthwise_result1.csv");
    result << "C,H,W,KH,KW,pad,stride,time_us\n";
    result << C << "," << H << "," << W << "," << KH << "," << KW << "," << pad << "," << stride << "," << duration_us << "\n";
    result.close();
}

int main() {
    queue q;
    std::cout << "Running on " << q.get_device().get_info<info::device::name>() << "\n";

    int C = 16, H = 32, W = 32, KH = 3, KW = 3, pad = 1, stride = 1;
    depthwise_conv(q, C, H, W, KH, KW, pad, stride);
    return 0;
}
