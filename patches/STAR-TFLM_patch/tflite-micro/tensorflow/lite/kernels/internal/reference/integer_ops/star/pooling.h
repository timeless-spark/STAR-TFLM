#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_STAR_POOLING_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_STAR_POOLING_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

template<typename out_type, int out_Nbit> 
inline void __attribute__((always_inline)) CastAndWrite_pool_STAR(int32_t acc, out_type* output_data, int output_offs);

template<>
inline void __attribute__((always_inline)) CastAndWrite_pool_STAR<int16_t, 16>(int32_t acc, int16_t* output_data, int output_offs)
{
  output_data[output_offs] = static_cast<int16_t>(acc);
  return;
}
template<>
inline void __attribute__((always_inline)) CastAndWrite_pool_STAR<int8_t, 8>(int32_t acc, int8_t* output_data, int output_offs)
{
  output_data[output_offs] = static_cast<int8_t>(acc);
  return;
}
template<>
inline void __attribute__((always_inline)) CastAndWrite_pool_STAR<int8_t, 4>(int32_t acc, int8_t* output_data, int output_offs)
{
  if (output_offs & 0x01)
  { // even (second 4bit element)
    output_offs = output_offs >> 1;
    output_data[output_offs] |= static_cast<int8_t>(acc) << 4;
  }
  else
  { // odd (first 4bit element)
    output_offs = output_offs >> 1;
    output_data[output_offs] = 0x00 | (0x0F & static_cast<int8_t>(acc));
  }
  return;
}

template <typename in_type, typename out_type, int out_Nbit>
bool __attribute__((noinline)) AveragePool_STAR(const PoolParams& params, const int32_t output_multiplier,
                        const int32_t output_shift, const RuntimeShape& input_shape,
                        const in_type* input_data, const RuntimeShape& output_shape,
                        out_type* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          int32_t acc = 0;
          int filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              acc +=
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              filter_count++;
            }
          }
          if (filter_count == 0) return false;

          acc += input_offset * filter_count;
          
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);

          scaled_acc = scaled_acc > 0 ? (scaled_acc + filter_count / 2) / filter_count
                        : (scaled_acc - filter_count / 2) / filter_count;

          scaled_acc += output_offset;

          scaled_acc = std::max(scaled_acc, params.quantized_activation_min);
          scaled_acc = std::min(scaled_acc, params.quantized_activation_max);

          int internal_output_offset = Offset(output_shape, batch, out_y, out_x, channel);
          CastAndWrite_pool_STAR<out_type, out_Nbit>(scaled_acc, output_data, internal_output_offset);
        }
      }
    }
  }
  return true;
}

template <typename in_type, typename out_type, int out_Nbit>
void __attribute__((noinline)) MaxPool_STAR(const PoolParams& params, const RuntimeShape& input_shape,
                    const in_type* input_data, const RuntimeShape& output_shape,
                    out_type* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          in_type max = std::numeric_limits<in_type>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
            }
          }
          max = std::max<in_type>(max, params.quantized_activation_min);
          max = std::min<in_type>(max, params.quantized_activation_max);

          int internal_output_offset = Offset(output_shape, batch, out_y, out_x, channel);
          CastAndWrite_pool_STAR<out_type, out_Nbit>(max, output_data, internal_output_offset);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_STAR_POOLING_H_
