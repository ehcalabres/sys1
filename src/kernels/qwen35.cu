#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <mma.h>





extern "C" __global__ void qwen35_online_attention_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* output, int query_heads, int queries, int keys,
    int heads, int kv_heads, int width, float scale, int causal) {
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * 4 + threadIdx.x / 32;
    if (item >= query_heads) return;
    const int head = item % heads;
    const int row = (item / heads) % queries;
    const int batch = item / (heads * queries);
    const int kv_head = head / (heads / kv_heads);
    const int channels = width / 32;
    float query[8], acc[8];
    #pragma unroll
    for (int channel = 0; channel < 8; ++channel) {
        query[channel] = channel < channels ? float(q[item * width + lane + channel * 32]) : 0.f;
        acc[channel] = 0.f;
    }
    float maximum = -INFINITY, denominator = 0.f;
    const int count = causal ? row + 1 : keys;
    for (int key = 0; key < count; ++key) {
        const int offset = ((batch * keys + key) * kv_heads + kv_head) * width;
        float score = 0.f;
        #pragma unroll
        for (int channel = 0; channel < 8; ++channel) {
            if (channel < channels) score = __fmaf_rn(query[channel], float(k[offset + lane + channel * 32]), score);
        }
        #pragma unroll
        for (int step = 16; step > 0; step >>= 1) score = __fadd_rn(score, __shfl_down_sync(0xffffffff, score, step));
        score = __fmul_rn(__shfl_sync(0xffffffff, score, 0), scale);
        float alpha = 0.f, probability = 0.f;
        if (lane == 0) {
            const float next = fmaxf(maximum, score);
            alpha = maximum == -INFINITY ? 0.f : expf(__fsub_rn(maximum, next));
            probability = expf(__fsub_rn(score, next));
            denominator = __fmaf_rn(denominator, alpha, probability);
            maximum = next;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
        probability = __shfl_sync(0xffffffff, probability, 0);
        #pragma unroll
        for (int channel = 0; channel < 8; ++channel) {
            if (channel < channels) acc[channel] = __fmaf_rn(probability,
                float(v[offset + lane + channel * 32]), __fmul_rn(acc[channel], alpha));
        }
    }
    const float inverse = __shfl_sync(0xffffffff, lane == 0 ? __fdiv_rn(1.f, denominator) : 0.f, 0);
    #pragma unroll
    for (int channel = 0; channel < 8; ++channel) {
        if (channel < channels) output[item * width + lane + channel * 32] = __float2bfloat16_rn(__fmul_rn(acc[channel], inverse));
    }
}




extern "C" __global__ void qwen35_attention_math_probe_f32(
    const float* input, float* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    output[index * 6] = exp2f(x);
    float approximate_exp2;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(approximate_exp2) : "f"(x));
    output[index * 6 + 1] = approximate_exp2;
    output[index * 6 + 2] = float(exp2(double(x)));
    output[index * 6 + 3] = expf(x);
    output[index * 6 + 4] = __expf(x);
    output[index * 6 + 5] = float(exp(double(x)));
}




extern "C" __global__ void qwen35_attention_mma_probe_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, float* output) {
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    __shared__ __align__(32) __nv_bfloat16 tile_a[256];
    __shared__ __align__(32) __nv_bfloat16 tile_b[256];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> chain, partial, outer;
    wmma::fill_fragment(chain, 0.f);
    wmma::fill_fragment(outer, 0.f);
    for (int first_k = 0; first_k < 256; first_k += 16) {
        for (int index = threadIdx.x; index < 256; index += 32) {
            tile_a[index] = input[(index / 16) * 256 + first_k + index % 16];
            tile_b[index] = weight[(index / 16) * 256 + first_k + index % 16];
        }
        __syncthreads();
        wmma::load_matrix_sync(a, tile_a, 16);
        wmma::load_matrix_sync(b, tile_b, 16);
        wmma::mma_sync(chain, a, b, chain);
        wmma::fill_fragment(partial, 0.f);
        wmma::mma_sync(partial, a, b, partial);
#pragma unroll
        for (int item = 0; item < outer.num_elements; ++item) {
            outer.x[item] = __fadd_rn(outer.x[item], partial.x[item]);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(output, chain, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(output + 256, outer, 16, wmma::mem_row_major);
    for (int index = threadIdx.x; index < 256; index += 32) {
        const int row = index / 16, column = index % 16;
        double sum = 0.;
        for (int k = 0; k < 256; ++k) {
            const double product = __dmul_rn(double(float(input[row * 256 + k])),
                                            double(float(weight[column * 256 + k])));
            sum = __dadd_rn(sum, product);
        }
        output[512 + index] = __double2float_rn(sum);
    }
#else
    for (int index = threadIdx.x; index < 768; index += blockDim.x) output[index] = 0.f;
#endif
}













__device__ __forceinline__ bool qwen35_projection_exceptional_operand(__nv_bfloat16 value) {
    const unsigned short magnitude = __bfloat16_as_ushort(value) & 0x7fff;
    return magnitude != 0 && (magnitude < 0x2000 || magnitude > 0x5f00);
}


extern "C" __global__ void qwen35_projection_classification_probe(unsigned int* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const __nv_bfloat16 operand = __ushort_as_bfloat16((unsigned short)index);
    const float value = float(operand);
    const bool previous = value != 0.f && (!isfinite(value)
        || fabsf(value) < 0x1p-63f || fabsf(value) > 0x1p63f);
    output[index] = (unsigned int)previous | ((unsigned int)qwen35_projection_exceptional_operand(operand) << 1);
}

template<int TileRows>
__device__ __forceinline__ void qwen35_projection_tile_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int inner,
    int columns,
    const __nv_bfloat16* bias = nullptr) {
    const int first_row = blockIdx.x * TileRows;
    const int first_column = blockIdx.y * 64;
    constexpr int WarpRows = TileRows < 32 ? TileRows : 32;
    constexpr int Groups = TileRows / WarpRows;
    constexpr int Threads = WarpRows * 8;
    constexpr int Outputs = TileRows * 64;
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;





    constexpr int SharedStride = 64 + 16;
    __shared__ __align__(32) __nv_bfloat16 tile_a[TileRows * SharedStride];
    __shared__ __align__(32) __nv_bfloat16 tile_b[64 * SharedStride];



    __shared__ __align__(32) float tile_c[WarpRows * 64];
    const int warp = threadIdx.x / 32;
    const int warp_row = warp / 4;
    const int warp_column = warp % 4;
    bool exceptional = false;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[Groups];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> running[Groups];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> correction[Groups];
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        wmma::fill_fragment(running[group], 0.f);
        wmma::fill_fragment(correction[group], 0.f);
    }
    for (int first_k = 0; first_k < inner; first_k += 64) {
        for (int index = threadIdx.x; index < TileRows * 64; index += Threads) {
            const int row = first_row + index / 64;
            const int k = first_k + index % 64;
            const int shared_index = (index / 64) * SharedStride + index % 64;
            tile_a[shared_index] = row < rows && k < inner
                ? input[row * inner + k] : __float2bfloat16_rn(0.f);
            exceptional |= qwen35_projection_exceptional_operand(tile_a[shared_index]);
        }
        for (int index = threadIdx.x; index < 4096; index += Threads) {
            const int column = first_column + index / 64;
            const int k = first_k + index % 64;
            const int shared_index = (index / 64) * SharedStride + index % 64;
            tile_b[shared_index] = column < columns && k < inner
                ? weight[column * inner + k] : __float2bfloat16_rn(0.f);
            exceptional |= qwen35_projection_exceptional_operand(tile_b[shared_index]);
        }
        __syncthreads();
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            wmma::fill_fragment(c[group], 0.f);
        }
        for (int offset = 0; offset < 64 && first_k + offset < inner; offset += 16) {
            wmma::load_matrix_sync(b, tile_b + warp_column * 16 * SharedStride + offset, SharedStride);
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
                wmma::load_matrix_sync(a, tile_a + (group * WarpRows + warp_row * 16) * SharedStride + offset, SharedStride);
                wmma::mma_sync(c[group], a, b, c[group]);
            }
        }
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
#pragma unroll
            for (int i = 0; i < c[group].num_elements; ++i) {
                const float partial = c[group].x[i];
                const float sum = __fadd_rn(running[group].x[i], partial);
                const float virtual_partial = __fsub_rn(sum, running[group].x[i]);
                const float error = __fadd_rn(
                    __fsub_rn(running[group].x[i], __fsub_rn(sum, virtual_partial)),
                    __fsub_rn(partial, virtual_partial));
                correction[group].x[i] = __fadd_rn(correction[group].x[i], error);
                running[group].x[i] = sum;
            }
        }
        __syncthreads();
    }



    const int exceptional_block = __syncthreads_or(exceptional);
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
#pragma unroll
        for (int i = 0; i < c[group].num_elements; ++i) {
            c[group].x[i] = __fadd_rn(running[group].x[i], correction[group].x[i]);
        }
        wmma::store_matrix_sync(tile_c + warp * 256, c[group], 16, wmma::mem_row_major);
        __syncthreads();
        for (int index = threadIdx.x; index < WarpRows * 64; index += Threads) {
            const int row = first_row + group * WarpRows + (index / 1024) * 16 + (index % 256) / 16;
            const int column = first_column + ((index / 256) % 4) * 16 + index % 16;
            if (row < rows && column < columns) {
                float value = tile_c[index];


                if (exceptional_block || !isfinite(value)) {
                    double sum = 0.;
                    for (int k = 0; k < inner; ++k) {
                        sum = __fma_rn(double(float(input[row * inner + k])),
                                       double(float(weight[column * inner + k])), sum);
                    }
                    value = __double2float_rn(sum);
                }
                if (bias != nullptr) value = __fadd_rn(value, float(bias[column]));
                output[row * columns + column] = __float2bfloat16_rn(value);
            }
        }
        __syncthreads();
    }
#else


    for (int index = threadIdx.x; index < Outputs; index += Threads) {
        const int row = first_row + index / 64;
        const int column = first_column + index % 64;
        if (row < rows && column < columns) {
            double sum = 0.;
            for (int k = 0; k < inner; ++k) {
                sum = __fma_rn(double(float(input[row * inner + k])),
                               double(float(weight[column * inner + k])), sum);
            }
            float value = __double2float_rn(sum);
            if (bias != nullptr) value = __fadd_rn(value, float(bias[column]));
            output[row * columns + column] = __float2bfloat16_rn(value);
        }
    }
#endif
}

extern "C" __global__ void qwen35_kv_projection_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows, int inner, int columns) {
    qwen35_projection_tile_bf16<16>(input, weight, output, rows, inner, columns);
}




extern "C" __global__ void qwen35_kv_projection_rows32_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows, int inner, int columns) {
    qwen35_projection_tile_bf16<32>(input, weight, output, rows, inner, columns);
}



extern "C" __global__ void qwen35_kv_projection_rows128_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows, int inner, int columns) {
    qwen35_projection_tile_bf16<128>(input, weight, output, rows, inner, columns);
}



extern "C" __global__ void qwen35_biased_projection_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    int rows, int inner, int columns) {
    qwen35_projection_tile_bf16<16>(input, weight, output, rows, inner, columns, bias);
}
extern "C" __global__ void qwen35_biased_projection_rows32_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    int rows, int inner, int columns) {
    qwen35_projection_tile_bf16<32>(input, weight, output, rows, inner, columns, bias);
}
extern "C" __global__ void qwen35_biased_projection_rows128_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    int rows, int inner, int columns) {
    qwen35_projection_tile_bf16<128>(input, weight, output, rows, inner, columns, bias);
}






__device__ __forceinline__ void qwen35_compensated_projection_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_bfloat16* output,
    int inner,
    int columns) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (column >= columns) return;
    float running = 0.f, correction = 0.f;
    bool exceptional = false;
    for (int index = lane; index < inner; index += 32) {
        const float a = float(input[row * inner + index]);
        const float b = float(weight[column * inner + index]);
        const float product = __fmul_rn(a, b);
        const float updated = __fadd_rn(running, product);
        const float virtual_product = __fsub_rn(updated, running);
        const float error = __fadd_rn(
            __fsub_rn(running, __fsub_rn(updated, virtual_product)),
            __fsub_rn(product, virtual_product));
        correction = __fadd_rn(correction, error);
        running = updated;





        exceptional |= a != 0.f && b != 0.f && fabsf(product) < FLT_MIN;
    }
    double sum = __dadd_rn(double(running), double(correction));
    if (__any_sync(0xffffffff, exceptional || !isfinite(running) || !isfinite(correction))) {
        sum = 0.;
        for (int index = lane; index < inner; index += 32) {
            sum = __fma_rn(double(float(input[row * inner + index])),
                           double(float(weight[column * inner + index])), sum);
        }
    }
    for (int step = 16; step > 0; step >>= 1) {
        sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
    }
    if (lane == 0) {
        float value = __double2float_rn(sum);
        if (bias != nullptr) value = __fadd_rn(value, float(bias[column]));
        output[row * columns + column] = __float2bfloat16_rn(value);
    }
}

extern "C" __global__ void qwen35_gate_projection_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int inner, int columns) {
    qwen35_compensated_projection_bf16(input, weight, nullptr, output, inner, columns);
}



extern "C" __global__ void qwen35_biased_compensated_projection_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output, int inner, int columns) {
    qwen35_compensated_projection_bf16(input, weight, bias, output, inner, columns);
}




__device__ __forceinline__ void qwen35_tiled_compensated_projection_impl_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    int rows, int inner, int columns) {
    constexpr int RowTile = 4, ColumnTile = 1;
    const int lane = threadIdx.x & 31;
    const int first_row = blockIdx.x * RowTile;
    const int first_column = blockIdx.y * 4 + (threadIdx.x >> 5) * ColumnTile;
    if (first_column >= columns) return;
    float running[RowTile][ColumnTile] = {}, correction[RowTile][ColumnTile] = {};
    unsigned exceptional = 0;
    for (int index = lane; index < inner; index += 32) {
        float a[RowTile], b[ColumnTile];
        #pragma unroll
        for (int r = 0; r < RowTile; ++r)
            a[r] = first_row + r < rows ? float(input[(first_row + r) * inner + index]) : 0.f;
        #pragma unroll
        for (int c = 0; c < ColumnTile; ++c)
            b[c] = first_column + c < columns ? float(weight[(first_column + c) * inner + index]) : 0.f;
        #pragma unroll
        for (int r = 0; r < RowTile; ++r) {
            #pragma unroll
            for (int c = 0; c < ColumnTile; ++c) {
                const float product = __fmul_rn(a[r], b[c]);
                const float updated = __fadd_rn(running[r][c], product);
                const float virtual_product = __fsub_rn(updated, running[r][c]);
                const float error = __fadd_rn(
                    __fsub_rn(running[r][c], __fsub_rn(updated, virtual_product)),
                    __fsub_rn(product, virtual_product));
                correction[r][c] = __fadd_rn(correction[r][c], error);
                running[r][c] = updated;
                if (a[r] != 0.f && b[c] != 0.f && fabsf(product) < FLT_MIN)
                    exceptional |= 1u << (r * ColumnTile + c);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < RowTile; ++r) {
        #pragma unroll
        for (int c = 0; c < ColumnTile; ++c) {
            const int row = first_row + r, column = first_column + c;
            if (row >= rows || column >= columns) continue;
            double sum = __dadd_rn(double(running[r][c]), double(correction[r][c]));
            if (__any_sync(0xffffffff,(exceptional & (1u << (r * ColumnTile + c)))
                || !isfinite(running[r][c]) || !isfinite(correction[r][c]))) {
                sum = 0.;
                for (int index = lane; index < inner; index += 32)
                    sum = __fma_rn(double(float(input[row * inner + index])),
                                   double(float(weight[column * inner + index])), sum);
            }
            #pragma unroll
            for (int step = 16; step > 0; step >>= 1)
                sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
            if (lane == 0) {
                float value = __double2float_rn(sum);
                if (bias != nullptr) value = __fadd_rn(value, float(bias[column]));
                output[row * columns + column] = __float2bfloat16_rn(value);
            }
        }
    }
}

extern "C" __global__ void qwen35_tiled_compensated_projection_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows, int inner, int columns) {
    qwen35_tiled_compensated_projection_impl_bf16(input, weight, nullptr, output, rows, inner, columns);
}

extern "C" __global__ void qwen35_biased_tiled_compensated_projection_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output, int rows, int inner, int columns) {
    qwen35_tiled_compensated_projection_impl_bf16(input, weight, bias, output, rows, inner, columns);
}






extern "C" __global__ void qwen35_guarded_double_projection_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int inner,
    int columns,
    double radius_coefficient) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (column >= columns) return;
    double fast_sum = 0., absolute_sum = 0.;
    bool unsafe = false;
    for (int index = lane; index < inner; index += 32) {
        const float a = float(input[row * inner + index]);
        const float b = float(weight[column * inner + index]);
        const double product = __dmul_rn(double(a), double(b));
        fast_sum = __dadd_rn(fast_sum, product);
        absolute_sum = __dadd_ru(absolute_sum, fabs(product));
        const float fp32_product = __fmul_rn(a, b);
        unsafe |= !isfinite(fp32_product)
            || (a != 0.f && b != 0.f && fabsf(fp32_product) < FLT_MIN);
    }
    for (int step = 16; step > 0; step >>= 1) {
        fast_sum = __dadd_rn(fast_sum, __shfl_down_sync(0xffffffff, fast_sum, step));
        absolute_sum = __dadd_ru(absolute_sum, __shfl_down_sync(0xffffffff, absolute_sum, step));
    }
    const bool unsafe_warp = __any_sync(0xffffffff, unsafe);
    int accepted = 0;
    if (lane == 0 && !unsafe_warp && isfinite(absolute_sum)
            && absolute_sum > 0. && absolute_sum <= double(FLT_MAX) * 0x1p-6
            && isfinite(radius_coefficient) && radius_coefficient > 0.) {


        const double radius = __dmul_ru(radius_coefficient, absolute_sum);
        const __nv_bfloat16 low = __float2bfloat16_rn(
            __double2float_rn(__dsub_rd(fast_sum, radius)));
        const __nv_bfloat16 high = __float2bfloat16_rn(
            __double2float_rn(__dadd_ru(fast_sum, radius)));
        accepted = __bfloat16_as_ushort(low) == __bfloat16_as_ushort(high);
        if (accepted) output[row * columns + column] = low;
    }
    if (__shfl_sync(0xffffffff, accepted, 0)) return;



    float running = 0.f, correction = 0.f;
    bool exceptional = false;
    for (int index = lane; index < inner; index += 32) {
        const float a = float(input[row * inner + index]);
        const float b = float(weight[column * inner + index]);
        const float product = __fmul_rn(a, b);
        const float updated = __fadd_rn(running, product);
        const float virtual_product = __fsub_rn(updated, running);
        const float error = __fadd_rn(
            __fsub_rn(running, __fsub_rn(updated, virtual_product)),
            __fsub_rn(product, virtual_product));
        correction = __fadd_rn(correction, error);
        running = updated;
        exceptional |= !isfinite(product) || !isfinite(running) || !isfinite(correction)
            || (a != 0.f && b != 0.f && fabsf(product) < FLT_MIN);
    }
    double sum = __dadd_rn(double(running), double(correction));
    if (__any_sync(0xffffffff, exceptional)) {
        sum = 0.;
        for (int index = lane; index < inner; index += 32) {
            sum = __fma_rn(double(float(input[row * inner + index])),
                           double(float(weight[column * inner + index])), sum);
        }
    }
    for (int step = 16; step > 0; step >>= 1) {
        sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
    }
    if (lane == 0) {
        output[row * columns + column] = __float2bfloat16_rn(__double2float_rn(sum));
    }
}





template<int RowTiles>
__device__ void qwen35_guarded_tensor_double_projection_impl_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows, int inner, int columns,
    double radius_coefficient) {
    const int lane = threadIdx.x & 31;
    const int first_row = blockIdx.x * (8 * RowTiles);
    const int first_column = blockIdx.y * 32 + (threadIdx.x >> 5) * 8;
    double sums[RowTiles][2] = {}, absolutes[RowTiles][2] = {};
    bool unsafe[RowTiles] = {};
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900
    for (int first_k = 0; first_k < inner; first_k += 4) {
        const int column = first_column + (lane >> 2);
        const int k = first_k + (lane & 3);
        const float b = column < columns && k < inner ? float(weight[column * inner + k]) : 0.f;


        const double bd = double(b), ab = fabs(bd);
        #pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
        const int row = first_row + row_tile * 8 + (lane >> 2);
        const float a = row < rows && k < inner ? float(input[row * inner + k]) : 0.f;


        unsafe[row_tile] |= !isfinite(a) || !isfinite(b)
            || (a != 0.f && (fabsf(a) < 0x1p-60f || fabsf(a) > 0x1p60f))
            || (b != 0.f && (fabsf(b) < 0x1p-60f || fabsf(b) > 0x1p60f));
        const double ad = double(a), aa = fabs(ad);
        asm volatile("mma.sync.aligned.m8n8k4.row.col.rn.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+d"(sums[row_tile][0]), "+d"(sums[row_tile][1]) : "d"(ad), "d"(bd));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.rp.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+d"(absolutes[row_tile][0]), "+d"(absolutes[row_tile][1]) : "d"(aa), "d"(ab));
        }
    }
#else
    #pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile)
        unsafe[row_tile] = true;
#endif
    #pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
    const int tile_first_row = first_row + row_tile * 8;
    const bool unsafe_warp = __any_sync(0xffffffff, unsafe[row_tile]);
    bool failures[2];
    #pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        const int row = tile_first_row + (lane >> 2);
        const int column = first_column + (lane & 3) * 2 + slot;
        const double sum = sums[row_tile][slot];
        const double absolute = absolutes[row_tile][slot];
        bool accepted = false;
        if (!unsafe_warp && row < rows && column < columns
                && isfinite(absolute) && absolute > 0.
                && absolute <= double(FLT_MAX) * 0x1p-6
                && isfinite(radius_coefficient) && radius_coefficient > 0.) {
            const double radius = __dmul_ru(radius_coefficient, absolute);
            const __nv_bfloat16 low = __float2bfloat16_rn(__double2float_rn(__dsub_rd(sum, radius)));
            const __nv_bfloat16 high = __float2bfloat16_rn(__double2float_rn(__dadd_ru(sum, radius)));
            accepted = __bfloat16_as_ushort(low) == __bfloat16_as_ushort(high);
            if (accepted) output[row * columns + column] = low;
        }
        failures[slot] = row < rows && column < columns && !accepted;
    }
    const unsigned mask0 = __ballot_sync(0xffffffff, failures[0]);
    const unsigned mask1 = __ballot_sync(0xffffffff, failures[1]);

    for (int slot = 0; slot < 64; ++slot) {
        const unsigned mask = (slot & 1) ? mask1 : mask0;
        if (!(mask & (1u << (slot >> 1)))) continue;
        const int row = tile_first_row + slot / 8;
        const int column = first_column + slot % 8;
        float running = 0.f, correction = 0.f;
        bool exceptional = false;
        for (int index = lane; index < inner; index += 32) {
            const float a = float(input[row * inner + index]);
            const float b = float(weight[column * inner + index]);
            const float product = __fmul_rn(a, b);
            const float updated = __fadd_rn(running, product);
            const float virtual_product = __fsub_rn(updated, running);
            const float error = __fadd_rn(
                __fsub_rn(running, __fsub_rn(updated, virtual_product)),
                __fsub_rn(product, virtual_product));
            correction = __fadd_rn(correction, error);
            running = updated;
            exceptional |= !isfinite(product) || !isfinite(running) || !isfinite(correction)
                || (a != 0.f && b != 0.f && fabsf(product) < FLT_MIN);
        }
        double sum = __dadd_rn(double(running), double(correction));
        if (__any_sync(0xffffffff, exceptional)) {
            sum = 0.;
            for (int index = lane; index < inner; index += 32) {
                sum = __fma_rn(double(float(input[row * inner + index])),
                              double(float(weight[column * inner + index])), sum);
            }
        }
        for (int step = 16; step > 0; step >>= 1)
            sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
        if (lane == 0) output[row * columns + column] = __float2bfloat16_rn(__double2float_rn(sum));
    }
    }
}

extern "C" __global__ void qwen35_guarded_tensor_double_projection_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows, int inner, int columns,
    double radius_coefficient) {
    const int lane = threadIdx.x & 31;
    const int first_row = blockIdx.x * 8;
    const int first_column = blockIdx.y * 32 + (threadIdx.x >> 5) * 8;
    double sum0 = 0., sum1 = 0., absolute0 = 0., absolute1 = 0.;
    bool unsafe = false;
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900
    for (int first_k = 0; first_k < inner; first_k += 4) {
        const int row = first_row + (lane >> 2);
        const int column = first_column + (lane >> 2);
        const int k = first_k + (lane & 3);
        const float a = row < rows && k < inner ? float(input[row * inner + k]) : 0.f;
        const float b = column < columns && k < inner ? float(weight[column * inner + k]) : 0.f;


        unsafe |= !isfinite(a) || !isfinite(b)
            || (a != 0.f && (fabsf(a) < 0x1p-60f || fabsf(a) > 0x1p60f))
            || (b != 0.f && (fabsf(b) < 0x1p-60f || fabsf(b) > 0x1p60f));
        const double ad = double(a), bd = double(b);
        const double aa = fabs(ad), ab = fabs(bd);
        asm volatile("mma.sync.aligned.m8n8k4.row.col.rn.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+d"(sum0), "+d"(sum1) : "d"(ad), "d"(bd));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.rp.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+d"(absolute0), "+d"(absolute1) : "d"(aa), "d"(ab));
    }
#else
    unsafe = true;
#endif
    const bool unsafe_warp = __any_sync(0xffffffff, unsafe);
    bool failures[2];
    for (int slot = 0; slot < 2; ++slot) {
        const int row = first_row + (lane >> 2);
        const int column = first_column + (lane & 3) * 2 + slot;
        const double sum = slot == 0 ? sum0 : sum1;
        const double absolute = slot == 0 ? absolute0 : absolute1;
        bool accepted = false;
        if (!unsafe_warp && row < rows && column < columns
                && isfinite(absolute) && absolute > 0.
                && absolute <= double(FLT_MAX) * 0x1p-6
                && isfinite(radius_coefficient) && radius_coefficient > 0.) {
            const double radius = __dmul_ru(radius_coefficient, absolute);
            const __nv_bfloat16 low = __float2bfloat16_rn(__double2float_rn(__dsub_rd(sum, radius)));
            const __nv_bfloat16 high = __float2bfloat16_rn(__double2float_rn(__dadd_ru(sum, radius)));
            accepted = __bfloat16_as_ushort(low) == __bfloat16_as_ushort(high);
            if (accepted) output[row * columns + column] = low;
        }
        failures[slot] = row < rows && column < columns && !accepted;
    }
    const unsigned mask0 = __ballot_sync(0xffffffff, failures[0]);
    const unsigned mask1 = __ballot_sync(0xffffffff, failures[1]);

    for (int slot = 0; slot < 64; ++slot) {
        const unsigned mask = (slot & 1) ? mask1 : mask0;
        if (!(mask & (1u << (slot >> 1)))) continue;
        const int row = first_row + slot / 8;
        const int column = first_column + slot % 8;
        float running = 0.f, correction = 0.f;
        bool exceptional = false;
        for (int index = lane; index < inner; index += 32) {
            const float a = float(input[row * inner + index]);
            const float b = float(weight[column * inner + index]);
            const float product = __fmul_rn(a, b);
            const float updated = __fadd_rn(running, product);
            const float virtual_product = __fsub_rn(updated, running);
            const float error = __fadd_rn(
                __fsub_rn(running, __fsub_rn(updated, virtual_product)),
                __fsub_rn(product, virtual_product));
            correction = __fadd_rn(correction, error);
            running = updated;
            exceptional |= !isfinite(product) || !isfinite(running) || !isfinite(correction)
                || (a != 0.f && b != 0.f && fabsf(product) < FLT_MIN);
        }
        double sum = __dadd_rn(double(running), double(correction));
        if (__any_sync(0xffffffff, exceptional)) {
            sum = 0.;
            for (int index = lane; index < inner; index += 32) {
                sum = __fma_rn(double(float(input[row * inner + index])),
                              double(float(weight[column * inner + index])), sum);
            }
        }
        for (int step = 16; step > 0; step >>= 1)
            sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
        if (lane == 0) output[row * columns + column] = __float2bfloat16_rn(__double2float_rn(sum));
    }
}


extern "C" __global__ void qwen35_guarded_tensor_double_rows16_projection_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows, int inner, int columns,
    double radius_coefficient) {
    qwen35_guarded_tensor_double_projection_impl_bf16<2>(
        input, weight, output, rows, inner, columns, radius_coefficient);
}

__device__ __forceinline__ float qwen35_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    }
    return value;
}

extern "C" __global__ void qwen35_rms_norm_delta_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* delta_weight,
    __nv_bfloat16* output,
    int columns,
    float epsilon) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = float(input[row * columns + column]);
        sum += value * value;
    }
    sum = qwen35_warp_sum(sum);
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
    sum = qwen35_warp_sum(sum);
    const float scale = rsqrtf(sum / float(columns) + epsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        output[index] = __nv_bfloat16(
            scale * float(input[index]) * (1.f + float(delta_weight[column])));
    }
}




extern "C" __global__ void qwen35_residual_rms_norm_delta_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* mixed,
    const __nv_bfloat16* delta_weight,
    __nv_bfloat16* residual,
    __nv_bfloat16* normalized,
    int columns,
    float epsilon) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        const __nv_bfloat16 value = __hadd(input[index], mixed[index]);
        residual[index] = value;
        const float rounded = float(value);
        sum += rounded * rounded;
    }
    sum = qwen35_warp_sum(sum);
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
    sum = qwen35_warp_sum(sum);
    const float scale = rsqrtf(sum / float(columns) + epsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        normalized[index] = __nv_bfloat16(
            scale * float(residual[index]) * (1.f + float(delta_weight[column])));
    }
}

extern "C" __global__ void qwen35_post_finalize_bf16(
    const float* input,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* norm,
    __nv_bfloat16* output,
    int columns,
    float epsilon) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = float(__nv_bfloat16(input[row * columns + column]));
        sum += value * value;
    }
    sum = qwen35_warp_sum(sum);
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
    sum = qwen35_warp_sum(sum);
    const float scale = rsqrtf(sum / float(columns) + epsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        const float gate_value = float(gate[index]);
        const float gate_silu = gate_value / (1.f + expf(-gate_value));
        const float recurrent = float(__nv_bfloat16(input[index]));
        const float normalized = float(__nv_bfloat16(__fmul_rn(recurrent, scale)));
        const float weighted = float(__nv_bfloat16(__fmul_rn(normalized, float(norm[column]))));
        output[index] = __nv_bfloat16(__fmul_rn(weighted, gate_silu));
    }
}

template <typename T>
__device__ void qwen35_conv(
    const T* input,
    const T* weights,
    T* output,
    int count,
    int length,
    int channels,
    int kernel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int channel = index % channels;
    const int token = index / channels % length;
    const int batch = index / (channels * length);


    float value = 0.f;
    for (int tap = 0; tap < kernel; ++tap) {
        const int source_token = token + tap - kernel + 1;
        if (source_token >= 0) {
            const float product = __fmul_rn(
                float(input[(batch * length + source_token) * channels + channel]),
                float(weights[channel * kernel + tap]));
            value = __fadd_rn(value, product);
        }
    }

    const float x = float(T(value));
    output[index] = T(x / (1.f + expf(-x)));
}

extern "C" __global__ void qwen35_conv_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weights,
    __nv_bfloat16* output,
    int count,
    int length,
    int channels,
    int kernel) {
    qwen35_conv(input, weights, output, count, length, channels, kernel);
}

extern "C" __global__ void qwen35_conv_f32(
    const float* input,
    const float* weights,
    float* output,
    int count,
    int length,
    int channels,
    int kernel) {
    qwen35_conv(input, weights, output, count, length, channels, kernel);
}

extern "C" __global__ void qwen35_delta_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* gate,
    const float* beta,
    float* output,
    int length,
    int heads,
    int key_heads,
    int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (column >= value_dim) return;

    float state[4] = {0.f, 0.f, 0.f, 0.f};
    for (int token = 0; token < length; ++token) {
        const int token_head = (batch * length + token) * heads + head;
        const int qk_head = (batch * length + token) * key_heads + head / (heads / key_heads);
        const int output_offset = batch_head * length + token;
        const float decay = expf(gate[token_head]);
        const float weight = beta[token_head];
        const float* token_query = query + qk_head * 128;
        const float* token_key = key + qk_head * 128;
        const float* token_value = value + token_head * value_dim;
        float keys[4];
        float remembered = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            keys[part] = token_key[part * 32 + lane];
            state[part] *= decay;
            remembered += keys[part] * state[part];
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
            remembered += __shfl_down_sync(0xffffffff, remembered, step);
        }
        const float delta =
            (token_value[column] - __shfl_sync(0xffffffff, remembered, 0)) * weight;
        float answer = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            state[part] += keys[part] * delta;
            answer += token_query[part * 32 + lane] * state[part];
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
            answer += __shfl_down_sync(0xffffffff, answer, step);
        }
        if (lane == 0) {
            output[output_offset * value_dim + column] = answer;
        }
    }
}

extern "C" __global__ void qwen35_delta_subwarp16_f32(
    const float* query, const float* key, const float* value,
    const float* gate, const float* beta, float* output,
    int length, int heads, int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int lane = threadIdx.x & 15;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 4);
    if (column >= value_dim) return;
    const unsigned mask = __activemask();
    float state[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    for (int token = 0; token < length; ++token) {
        const int token_head = (batch * length + token) * heads + head;
        const int qk_head = (batch * length + token) * key_heads + head / (heads / key_heads);
        const int output_offset = batch_head * length + token;
        const float decay = expf(gate[token_head]);
        const float weight = beta[token_head];
        const float* token_query = query + qk_head * 128;
        const float* token_key = key + qk_head * 128;
        const float* token_value = value + token_head * value_dim;
        float keys[2][4];
        float remembered[2] = {0.f,0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                keys[half][part] = token_key[part * 32 + lane + half * 16];
                state[half][part] *= decay;
                remembered[half] += keys[half][part] * state[half][part];
            }
        }


        float total = remembered[0] + remembered[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        const float delta = (token_value[column] - __shfl_sync(mask, total, 0, 16)) * weight;
        float answer[2] = {0.f,0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                state[half][part] += keys[half][part] * delta;
                answer[half] += token_query[part * 32 + lane + half * 16] * state[half][part];
            }
        }
        total = answer[0] + answer[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        if (lane == 0) output[output_offset * value_dim + column] = total;
    }
}




extern "C" __global__ void qwen35_l2_128_f32(
    const float* input, float* output, unsigned int rows, float scale) {
    const unsigned int row = blockIdx.x * 4 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const unsigned int offset = row * 128 + lane * 4;
    float values[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) values[j] = input[offset + j];
    float sum = __fmul_rn(values[0], values[0]);
    #pragma unroll
    for (int j = 1; j < 4; ++j)
        sum = __fadd_rn(sum, __fmul_rn(values[j], values[j]));
    #pragma unroll
    for (int step = 16; step > 0; step >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, step));
    sum = __shfl_sync(0xffffffff, sum, 0);
    const float denom = __fsqrt_rn(__fadd_rn(sum, 1e-6f));
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float normalized = __fdiv_rn(values[j], denom);
        output[offset + j] = scale == 1.f ? normalized
            : __fmaf_rn(normalized, scale, 0.f);
    }
}


extern "C" __global__ void qwen35_preexp_gate_f32(
    const float* gate, float* decay, unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) decay[index] = expf(gate[index]);
}

extern "C" __global__ void qwen35_delta_subwarp16_preexp_f32(
    const float* query, const float* key, const float* value,
    const float* decay_values, const float* beta, float* output,
    int length, int heads, int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int lane = threadIdx.x & 15;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 4);
    if (column >= value_dim) return;
    const unsigned mask = __activemask();
    float state[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    for (int token = 0; token < length; ++token) {
        const int token_head = (batch * length + token) * heads + head;
        const int qk_head = (batch * length + token) * key_heads + head / (heads / key_heads);
        const int output_offset = batch_head * length + token;
        const float decay = decay_values[token_head];
        const float weight = beta[token_head];
        const float* token_query = query + qk_head * 128;
        const float* token_key = key + qk_head * 128;
        const float* token_value = value + token_head * value_dim;
        float keys[2][4];
        float remembered[2] = {0.f,0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                keys[half][part] = token_key[part * 32 + lane + half * 16];
                state[half][part] *= decay;
                remembered[half] += keys[half][part] * state[half][part];
            }
        }


        float total = remembered[0] + remembered[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        const float delta = (token_value[column] - __shfl_sync(mask, total, 0, 16)) * weight;
        float answer[2] = {0.f,0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                state[half][part] += keys[half][part] * delta;
                answer[half] += token_query[part * 32 + lane + half * 16] * state[half][part];
            }
        }
        total = answer[0] + answer[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        if (lane == 0) output[output_offset * value_dim + column] = total;
    }
}
