#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
using namespace sycl;

void im2col_conv(queue& q, int IC, int OC, int H, int W, int KH, int KW, int pad, int stride) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    int col_dim = IC * KH * KW;
    int out_dim = OH * OW;

    std::vector<float> input(IC * H * W, 1.0f);
    std::vector<float> kernel(OC * IC * KH * KW, 1.0f);
    std::vector<float> output(OC * OH * OW, 0.0f);

    buffer<float> in_buf(input.data(), range<1>(input.size()));
    buffer<float> wt_buf(kernel.data(), range<1>(kernel.size()));
    buffer<float> out_buf(output.data(), range<1>(output.size()));

    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](handler& h) {
        auto in = in_buf.get_access<access::mode::read>(h);
        auto wt = wt_buf.get_access<access::mode::read>(h);
        auto out = out_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<2>(OC, OH * OW), [=](id<2> idx) {
            int oc = idx[0];
            int out_idx = idx[1];
            int h_out = out_idx / OW;
            int w_out = out_idx % OW;

            float acc = 0.0f;
            for (int ic = 0; ic < IC; ++ic)
                for (int i = 0; i < KH; ++i)
                    for (int j = 0; j < KW; ++j) {
                        int h_in = h_out * stride + i - pad;
                        int w_in = w_out * stride + j - pad;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            float val = in[ic * H * W + h_in * W + w_in];
                            float wgt = wt[oc * IC * KH * KW + ic * KH * KW + i * KW + j];
                            acc += val * wgt;
                        }
                    }
            out[oc * OH * OW + h_out * OW + w_out] = acc;
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::micro>(end - start).count();

    std::ofstream f("im2col_result.csv");
    f << "IC,OC,H,W,KH,KW,pad,stride,time_us\n";
    f << IC << "," << OC << "," << H << "," << W << "," << KH << "," << KW << "," << pad << "," << stride << "," << duration << "\n";
    f.close();
}

int main() {
    queue q;
    std::cout << "Running on " << q.get_device().get_info<info::device::name>() << "\n";
    im2col_conv(q, 16, 32, 32, 32, 3, 3, 1, 1);
    return 0;
}
