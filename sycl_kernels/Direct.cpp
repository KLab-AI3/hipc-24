#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/fpga_device_selector.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;

extern "C" {
    float* direct_conv(float* input, float* kernel, float* output,
                       int input_height, int input_width, int input_channels,
                       int kernel_height, int kernel_width, int output_channels, int stride, int padding);
}

float* direct_conv(float* input, float* kernel, float* output,
                   int input_height, int input_width, int input_channels,
                   int kernel_height, int kernel_width, int output_channels,
                   int stride, int padding) {

    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;
    int output_size = output_height * output_width;

    // Define the maximum work-group size
    const int max_work_group_size = 256;
    int work_group_size = std::min(max_work_group_size, output_size);

    // Ensure work-group size divides the output_size evenly
    int global_size = ((output_size + work_group_size - 1) / work_group_size) * work_group_size;

    sycl::queue queue(sycl::gpu_selector_v);

    // Create buffers for the input, kernel, and output
    sycl::buffer<float, 1> buffer_input(input, input_height * input_width * input_channels);
    sycl::buffer<float, 1> buffer_kernel(kernel, kernel_height * kernel_width * input_channels * output_channels);
    sycl::buffer<float, 1> buffer_output(output, output_height * output_width * output_channels);

    queue.submit([&](sycl::handler &h) {
        // Get accessors for input, kernel, and output
        auto acc_input = buffer_input.get_access<sycl::access::mode::read>(h);
        auto acc_kernel = buffer_kernel.get_access<sycl::access::mode::read>(h);
        auto acc_output = buffer_output.get_access<sycl::access::mode::write>(h);

        // Define the kernel
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(output_channels, global_size),
                                         sycl::range<2>(1, work_group_size)), [=](sycl::nd_item<2> item) {
            int m = item.get_global_id(0);
            int output_index = item.get_global_id(1);
            if (output_index >= output_size) return;  // Skip padding work-items
            int row = output_index / output_width;
            int col = output_index % output_width;
            float sum = 0.0f;

            for (int c = 0; c < input_channels; ++c) {
                for (int i = 0; i < kernel_height; ++i) {
                    for (int j = 0; j < kernel_width; ++j) {
                        int h_offset = row * stride - padding + i;
                        int w_offset = col * stride - padding + j;
                        if (h_offset >= 0 && h_offset < input_height && w_offset >= 0 && w_offset < input_width) {
                            sum += acc_input[c * input_height * input_width + h_offset * input_width + w_offset] *
                                   acc_kernel[m * input_channels * kernel_height * kernel_width + c * kernel_height * kernel_width + i * kernel_width + j];
                        }
                    }
                }
            }

            acc_output[m * output_height * output_width + row * output_width + col] = sum;
        });
    });

    queue.wait_and_throw();
    return output;
}

void initialize_tensor(float* tensor, int size) {
    for (int i = 0; i < size; ++i) {
        tensor[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

int main() {
    vector<int> input_sizes = {32, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000};
    vector<int> kernel_sizes = {1, 3, 5};
    vector<int> channels = {1, 2, 4, 8, 16, 32, 64, 128};

    srand(time(0));

    // Open a CSV file to write the results
    ofstream csv_file("direct_results.csv");
    csv_file << "InputSize,KernelSize,InputChannels,OutputChannels,ExecutionTime(s)" << endl;

    for (int input_size : input_sizes) {
        for (int kernel_size : kernel_sizes) {
            for (int input_channels : channels) {
                for (int output_channels : channels) {
                    int input_height = input_size;
                    int input_width = input_size;
                    int kernel_height = kernel_size;
                    int kernel_width = kernel_size;
                    int stride = 1;
                    int padding = 0;

                    int input_size_total = input_height * input_width * input_channels;
                    int kernel_size_total = kernel_height * kernel_width * input_channels * output_channels;
                    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
                    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;
                    int output_size_total = output_height * output_width * output_channels;

                    float* input = new float[input_size_total];
                    float* kernel = new float[kernel_size_total];
                    float* output = new float[output_size_total];

                    initialize_tensor(input, input_size_total);
                    initialize_tensor(kernel, kernel_size_total);

                    auto start = high_resolution_clock::now();
                    direct_conv(input, kernel, output, input_height, input_width, input_channels, kernel_height, kernel_width, output_channels, stride, padding);
                    auto stop = high_resolution_clock::now();
                    duration<double> duration_sec = stop - start;

                    csv_file << input_size << "," << kernel_size << "," << input_channels << "," << output_channels << "," << duration_sec.count() << endl;

                    delete[] input;
                    delete[] kernel;
                    delete[] output;
                }
            }
        }
    }

    csv_file.close();

    cout << "Results have been written to direct_results.csv" << endl;

    return 0;
}
