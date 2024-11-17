#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_STAR_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_STAR_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "tensorflow/lite/micro/ibex_rv32im/ibex_utility.h"

#ifdef STAR_EMULATE
typedef int64_t acc_t;
#else
typedef int32_t acc_t;
#endif

namespace tflite {
namespace reference_integer_ops {

template<int OP_parall>
inline int32_t __attribute__((always_inline)) MakeOffsetArray_conv_STAR(int32_t input_offs);

template<>
inline int32_t __attribute__((always_inline)) MakeOffsetArray_conv_STAR<16>(int32_t input_offs)
{
  int32_t masked_offs = input_offs & 0x0000FFFF;
  return masked_offs | (masked_offs << 16);
}
template<>
inline int32_t __attribute__((always_inline)) MakeOffsetArray_conv_STAR<8>(int32_t input_offs)
{
  int32_t masked_offs = input_offs & 0x000000FF;
  return masked_offs | (masked_offs << 8) | (masked_offs << 16) | (masked_offs << 24);
}
template<>
inline int32_t __attribute__((always_inline)) MakeOffsetArray_conv_STAR<4>(int32_t input_offs)
{
  int32_t masked_offs = input_offs & 0x0000000F;
  return masked_offs | (masked_offs << 4) | (masked_offs << 8) | (masked_offs << 12) | (masked_offs << 16) | (masked_offs << 20) | (masked_offs << 24) | (masked_offs << 28);
}

template<typename out_type, int out_Nbit> 
inline void __attribute__((always_inline)) CastAndWrite_conv_STAR(int32_t& acc, out_type* output_data, int output_offs);

template<>
inline void __attribute__((always_inline)) CastAndWrite_conv_STAR<int16_t, 16>(int32_t& acc, int16_t* output_data, int output_offs)
{
  output_data[output_offs] = static_cast<int16_t>(acc);
  return;
}
template<>
inline void __attribute__((always_inline)) CastAndWrite_conv_STAR<int8_t, 8>(int32_t& acc, int8_t* output_data, int output_offs)
{
  output_data[output_offs] = static_cast<int8_t>(acc);
  return;
}
template<>
inline void __attribute__((always_inline)) CastAndWrite_conv_STAR<int8_t, 4>(int32_t& acc, int8_t* output_data, int output_offs)
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

template<int OP_parall>
inline void __attribute__((always_inline)) performMULST_conv_STAR(acc_t& acc, int32_t filter_val, int32_t input_val, int32_t in_offs_arr);

template<>
inline void __attribute__((always_inline)) performMULST_conv_STAR<16>(acc_t& acc, int32_t filter_val, int32_t input_val, int32_t in_offs_arr)
{
  #ifdef STAR_EMULATE

  int32_t fil_val_1 = ((filter_val) & 0x0000FFFF) < 0x8000 ? ((filter_val) & 0x0000FFFF) : ((filter_val) | 0xFFFF0000);
  int32_t fil_val_2 = filter_val>>16;
  int32_t in_val_1  = ((input_val) & 0x0000FFFF) < 0x8000 ? ((input_val) & 0x0000FFFF) : ((input_val) | 0xFFFF0000);
  int32_t in_val_2  = input_val>>16;
  int32_t off_val_1 = ((in_offs_arr) & 0x0000FFFF) < 0x8000 ? ((in_offs_arr) & 0x0000FFFF) : ((in_offs_arr) | 0xFFFF0000);
  int32_t off_val_2 = in_offs_arr>>16;

  acc += fil_val_1 * in_val_1;
  acc += fil_val_2 * in_val_2;

  acc += fil_val_1 * off_val_1;
  acc += fil_val_2 * off_val_2;

  #else
  
  asm volatile("mac16st %0, %1, %2\n":"=r"(acc):"r"(filter_val),"r"(input_val):"memory");
  asm volatile("mac16st %0, %1, %2\n":"=r"(acc):"r"(filter_val),"r"(in_offs_arr):"memory");
    
  #endif

  return;
}
template<>
inline void __attribute__((always_inline)) performMULST_conv_STAR<8>(acc_t& acc, int32_t filter_val, int32_t input_val, int32_t in_offs_arr)
{
  #ifdef STAR_EMULATE

  int32_t fil_val_1 = ((filter_val) & 0x000000FF) < 0x80 ? ((filter_val) & 0x000000FF) : ((filter_val) | 0xFFFFFF00);
  int32_t fil_val_2 = ((filter_val>>8) & 0x000000FF) < 0x80 ? ((filter_val>>8) & 0x000000FF) : ((filter_val>>8) | 0xFFFFFF00);
  int32_t fil_val_3 = ((filter_val>>16) & 0x000000FF) < 0x80 ? ((filter_val>>16) & 0x000000FF) : ((filter_val>>16) | 0xFFFFFF00);
  int32_t fil_val_4 = filter_val>>24;
  int32_t in_val_1  = ((input_val) & 0x000000FF) < 0x80 ? ((input_val) & 0x000000FF) : ((input_val) | 0xFFFFFF00);
  int32_t in_val_2  = ((input_val>>8) & 0x000000FF) < 0x80 ? ((input_val>>8) & 0x000000FF) : ((input_val>>8) | 0xFFFFFF00);
  int32_t in_val_3  = ((input_val>>16) & 0x000000FF) < 0x80 ? ((input_val>>16) & 0x000000FF) : ((input_val>>16) | 0xFFFFFF00);
  int32_t in_val_4  = input_val>>24;
  int32_t off_val_1 = ((in_offs_arr) & 0x000000FF) < 0x80 ? ((in_offs_arr) & 0x000000FF) : ((in_offs_arr) | 0xFFFFFF00);
  int32_t off_val_2 = ((in_offs_arr>>8) & 0x000000FF) < 0x80 ? ((in_offs_arr>>8) & 0x000000FF) : ((in_offs_arr>>8) | 0xFFFFFF00);
  int32_t off_val_3 = ((in_offs_arr>>16) & 0x000000FF) < 0x80 ? ((in_offs_arr>>16) & 0x000000FF) : ((in_offs_arr>>16) | 0xFFFFFF00);
  int32_t off_val_4 = in_offs_arr>>24;

  acc += fil_val_2 * in_val_1;
  acc += fil_val_1 * in_val_2;
  acc += fil_val_4 * in_val_3;
  acc += fil_val_3 * in_val_4;

  acc += fil_val_2 * off_val_1;
  acc += fil_val_1 * off_val_2;
  acc += fil_val_4 * off_val_3;
  acc += fil_val_3 * off_val_4;

  #else
  
  asm volatile("mac8st %0, %1, %2\n":"=r"(acc):"r"(filter_val),"r"(input_val):"memory");
  asm volatile("mac8st %0, %1, %2\n":"=r"(acc):"r"(filter_val),"r"(in_offs_arr):"memory");

  #endif

  return;
}
template<>
inline void __attribute__((always_inline)) performMULST_conv_STAR<4>(acc_t& acc, int32_t filter_val, int32_t input_val, int32_t in_offs_arr)
{
  #ifdef STAR_EMULATE

  int32_t fil_val_1 = ((filter_val) & 0x0000000F) < 0x8 ? ((filter_val) & 0x0000000F) : ((filter_val) | 0xFFFFFFF0);
  int32_t fil_val_2 = ((filter_val>>4) & 0x0000000F) < 0x8 ? ((filter_val>>4) & 0x0000000F) : ((filter_val>>4) | 0xFFFFFFF0);
  int32_t fil_val_3 = ((filter_val>>8) & 0x0000000F) < 0x8 ? ((filter_val>>8) & 0x0000000F) : ((filter_val>>8) | 0xFFFFFFF0);
  int32_t fil_val_4 = ((filter_val>>12) & 0x0000000F) < 0x8 ? ((filter_val>>12) & 0x0000000F) : ((filter_val>>12) | 0xFFFFFFF0);
  int32_t fil_val_5 = ((filter_val>>16) & 0x0000000F) < 0x8 ? ((filter_val>>16) & 0x0000000F) : ((filter_val>>16) | 0xFFFFFFF0);
  int32_t fil_val_6 = ((filter_val>>20) & 0x0000000F) < 0x8 ? ((filter_val>>20) & 0x0000000F) : ((filter_val>>20) | 0xFFFFFFF0);
  int32_t fil_val_7 = ((filter_val>>24) & 0x0000000F) < 0x8 ? ((filter_val>>24) & 0x0000000F) : ((filter_val>>24) | 0xFFFFFFF0);
  int32_t fil_val_8 = filter_val>>28;
  int32_t in_val_1  = ((input_val) & 0x0000000F) < 0x8 ? ((input_val) & 0x0000000F) : ((input_val) | 0xFFFFFFF0);
  int32_t in_val_2  = ((input_val>>4) & 0x0000000F) < 0x8 ? ((input_val>>4) & 0x0000000F) : ((input_val>>4) | 0xFFFFFFF0);
  int32_t in_val_3  = ((input_val>>8) & 0x0000000F) < 0x8 ? ((input_val>>8) & 0x0000000F) : ((input_val>>8) | 0xFFFFFFF0);
  int32_t in_val_4  = ((input_val>>12) & 0x0000000F) < 0x8 ? ((input_val>>12) & 0x0000000F) : ((input_val>>12) | 0xFFFFFFF0);
  int32_t in_val_5  = ((input_val>>16) & 0x0000000F) < 0x8 ? ((input_val>>16) & 0x0000000F) : ((input_val>>16) | 0xFFFFFFF0);
  int32_t in_val_6  = ((input_val>>20) & 0x0000000F) < 0x8 ? ((input_val>>20) & 0x0000000F) : ((input_val>>20) | 0xFFFFFFF0);
  int32_t in_val_7  = ((input_val>>24) & 0x0000000F) < 0x8 ? ((input_val>>24) & 0x0000000F) : ((input_val>>24) | 0xFFFFFFF0);
  int32_t in_val_8  = input_val>>28;
  int32_t off_val_1 = ((in_offs_arr) & 0x0000000F) < 0x8 ? ((in_offs_arr) & 0x0000000F) : ((in_offs_arr) | 0xFFFFFFF0);
  int32_t off_val_2 = ((in_offs_arr>>4) & 0x0000000F) < 0x8 ? ((in_offs_arr>>4) & 0x0000000F) : ((in_offs_arr>>4) | 0xFFFFFFF0);
  int32_t off_val_3 = ((in_offs_arr>>8) & 0x0000000F) < 0x8 ? ((in_offs_arr>>8) & 0x0000000F) : ((in_offs_arr>>8) | 0xFFFFFFF0);
  int32_t off_val_4 = ((in_offs_arr>>12) & 0x0000000F) < 0x8 ? ((in_offs_arr>>12) & 0x0000000F) : ((in_offs_arr>>12) | 0xFFFFFFF0);
  int32_t off_val_5 = ((in_offs_arr>>16) & 0x0000000F) < 0x8 ? ((in_offs_arr>>16) & 0x0000000F) : ((in_offs_arr>>16) | 0xFFFFFFF0);
  int32_t off_val_6 = ((in_offs_arr>>20) & 0x0000000F) < 0x8 ? ((in_offs_arr>>20) & 0x0000000F) : ((in_offs_arr>>20) | 0xFFFFFFF0);
  int32_t off_val_7 = ((in_offs_arr>>24) & 0x0000000F) < 0x8 ? ((in_offs_arr>>24) & 0x0000000F) : ((in_offs_arr>>24) | 0xFFFFFFF0);
  int32_t off_val_8 = in_offs_arr>>28;

  acc += fil_val_4 * in_val_1;
  acc += fil_val_3 * in_val_2;
  acc += fil_val_2 * in_val_3;
  acc += fil_val_1 * in_val_4;
  acc += fil_val_8 * in_val_5;
  acc += fil_val_7 * in_val_6;
  acc += fil_val_6 * in_val_7;
  acc += fil_val_5 * in_val_8;

  acc += fil_val_4 * off_val_1;
  acc += fil_val_3 * off_val_2;
  acc += fil_val_2 * off_val_3;
  acc += fil_val_1 * off_val_4;
  acc += fil_val_8 * off_val_5;
  acc += fil_val_7 * off_val_6;
  acc += fil_val_6 * off_val_7;
  acc += fil_val_5 * off_val_8;

  #else
  
  asm volatile("mac4st %0, %1, %2\n":"=r"(acc):"r"(filter_val),"r"(input_val):"memory");
  asm volatile("mac4st %0, %1, %2\n":"=r"(acc):"r"(filter_val),"r"(in_offs_arr):"memory");

  #endif

  return;
}

template<int OP_parall>
inline int32_t __attribute__((always_inline)) getAccAddBiasAndQuantize_conv_STAR(acc_t& acc, const int32_t* bias_data, int& out_c, const int32_t* output_multiplier, const int32_t* output_shift, const int32_t& output_offset, const int32_t& output_activation_min, const int32_t& output_activation_max);

template<>
inline int32_t __attribute__((always_inline)) getAccAddBiasAndQuantize_conv_STAR<16>(acc_t& acc, const int32_t* bias_data, int& out_c, const int32_t* output_multiplier, const int32_t* output_shift, const int32_t& output_offset, const int32_t& output_activation_min, const int32_t& output_activation_max)
{
  #ifdef STAR_EMULATE

  int64_t acc_64 = acc;
  
  #else

  int32_t ext_acc;
  asm volatile("mac16sth %0, x0, x0\n":"=r"(ext_acc)::"memory");
  int64_t acc_64 = (((int64_t) ext_acc) << 32) | (((int64_t) acc) & 0x00000000FFFFFFFF);

  #endif

  if (bias_data) {
      acc_64 += static_cast<int32_t>(bias_data[out_c]);
  }

  int32_t new_acc = MultiplyByQuantizedMultiplier(acc_64, output_multiplier[out_c],
                                                  output_shift[out_c]);
  new_acc += output_offset;
  new_acc = std::max(new_acc, output_activation_min);
  new_acc = std::min(new_acc, output_activation_max);

  return new_acc;
}

template<>
inline int32_t __attribute__((always_inline)) getAccAddBiasAndQuantize_conv_STAR<8>(acc_t& acc, const int32_t* bias_data, int& out_c, const int32_t* output_multiplier, const int32_t* output_shift, const int32_t& output_offset, const int32_t& output_activation_min, const int32_t& output_activation_max)
{
  
  int32_t new_acc = (int32_t)acc;

  if (bias_data) {
      new_acc += static_cast<int32_t>(bias_data[out_c]);
  }
  new_acc = MultiplyByQuantizedMultiplier(new_acc, output_multiplier[out_c],
                                      output_shift[out_c]);
  new_acc += output_offset;
  new_acc = std::max(new_acc, output_activation_min);
  new_acc = std::min(new_acc, output_activation_max);

  return new_acc;
}

template<>
inline int32_t __attribute__((always_inline)) getAccAddBiasAndQuantize_conv_STAR<4>(acc_t& acc, const int32_t* bias_data, int& out_c, const int32_t* output_multiplier, const int32_t* output_shift, const int32_t& output_offset, const int32_t& output_activation_min, const int32_t& output_activation_max)
{
  int32_t new_acc = (int32_t)acc;

  if (bias_data) {
      new_acc += static_cast<int32_t>(bias_data[out_c]);
  }
  new_acc = MultiplyByQuantizedMultiplier(new_acc, output_multiplier[out_c],
                                      output_shift[out_c]);
  new_acc += output_offset;
  new_acc = std::max(new_acc, output_activation_min);
  new_acc = std::min(new_acc, output_activation_max);

  return new_acc;
}

template<int OP_parall, typename out_type, int out_Nbit>
void __attribute__((noinline)) ConvPerChannel_STAR(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int32_t* input_data, const RuntimeShape& filter_shape,
    const int32_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    out_type* output_data) {
  #ifdef MEASURE_CNN
  enable_counter();
  #endif
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = filter_shape.Dims(0);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  int32_t input_offset_array = MakeOffsetArray_conv_STAR<OP_parall>(input_offset);

  constexpr int n_instr_parall = 32 / OP_parall;

  int norm_filter_input_depth = filter_input_depth / n_instr_parall;

  RuntimeShape int_input_shape(input_shape);
  int_input_shape.SetDim(3, int_input_shape.DimsData()[3] / n_instr_parall);
  RuntimeShape int_filter_shape(filter_shape);
  int_filter_shape.SetDim(3, int_filter_shape.DimsData()[3] / n_instr_parall);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          acc_t acc = 0;
          #ifndef STAR_EMULATE
          asm volatile("macrst %0, x0, x0\n":"=r"(acc)::"memory");
          #endif
          
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) { 
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < norm_filter_input_depth;
                   ++in_channel) {
                int internal_input_offset = Offset(int_input_shape, batch, in_y, in_x,
                                      in_channel + group * norm_filter_input_depth);
                int internal_filter_offset = Offset(int_filter_shape, out_channel, 
                                               filter_y, filter_x, in_channel);
                int32_t input_val =
                    input_data[internal_input_offset];
                int32_t filter_val = filter_data[internal_filter_offset];

                performMULST_conv_STAR<OP_parall>(acc, filter_val, input_val, input_offset_array);
              }
            }
          }

          int32_t quant_acc = getAccAddBiasAndQuantize_conv_STAR<OP_parall>(acc, bias_data, out_channel, output_multiplier, output_shift, output_offset, output_activation_min, output_activation_max);

          int internal_output_offset = Offset(output_shape, batch, out_y, out_x, out_channel);
          CastAndWrite_conv_STAR<out_type, out_Nbit>(quant_acc, output_data, internal_output_offset);
        }
      }
    }
  }
  #ifdef MEASURE_CNN
  disable_counter();
  #endif
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_STAR_CONV_H_
