#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_STAR_ADD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_STAR_ADD_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_integer_ops {

template<int Nbit>
inline void CheckOneArithmeticParam_add_STAR(const int32_t input_offset);

template<>
inline void CheckOneArithmeticParam_add_STAR<16>(const int32_t input_offset) {
  TFLITE_DCHECK_GE(-input_offset, std::numeric_limits<int16_t>::min());
  TFLITE_DCHECK_LE(-input_offset, std::numeric_limits<int16_t>::max());
}

template<>
inline void CheckOneArithmeticParam_add_STAR<8>(const int32_t input_offset) {
  TFLITE_DCHECK_GE(-input_offset, std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_LE(-input_offset, std::numeric_limits<int8_t>::max());
}

template<>
inline void CheckOneArithmeticParam_add_STAR<4>(const int32_t input_offset) {
  TFLITE_DCHECK_GE(-input_offset, -8);
  TFLITE_DCHECK_LE(-input_offset, +7);
}

template<int in1_Nbit, int in2_Nbit>
inline void CheckTwoArithmeticParams_add_STAR(const ArithmeticParams& params)
{
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  // Input offset is negative input zero point. Activation tensors are
  // asymmetric quantized so they span the full int8 range.
  CheckOneArithmeticParam_add_STAR<in1_Nbit>(params.input1_offset);
  CheckOneArithmeticParam_add_STAR<in2_Nbit>(params.input2_offset);
}

template<typename out_type, int out_Nbit> 
inline void __attribute__((always_inline)) CastAndWrite_add_STAR(int32_t acc, out_type* output_data, int output_offs);

template<>
inline void __attribute__((always_inline)) CastAndWrite_add_STAR<int16_t, 16>(int32_t acc, int16_t* output_data, int output_offs)
{
  output_data[output_offs] = static_cast<int16_t>(acc);
  return;
}
template<>
inline void __attribute__((always_inline)) CastAndWrite_add_STAR<int8_t, 8>(int32_t acc, int8_t* output_data, int output_offs)
{
  output_data[output_offs] = static_cast<int8_t>(acc);
  return;
}
template<>
inline void __attribute__((always_inline)) CastAndWrite_add_STAR<int8_t, 4>(int32_t acc, int8_t* output_data, int output_offs)
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

template <typename in1_type, typename in2_type, typename out_type, int out_Nbit>
inline void ElementWise_add_STAR(
    int size, const ArithmeticParams& params, const in1_type* input1_data,
    const in2_type* input2_data, out_type* output_data,
    void (*check_arithmetic_params)(const ArithmeticParams&),
    int32_t (*binary_func)(in1_type, in2_type, const ArithmeticParams&)) {
  check_arithmetic_params(params);
  for (int i = 0; i < size; ++i) {
    int32_t add_res = binary_func(input1_data[i], input2_data[i], params);
    CastAndWrite_add_STAR<out_type, out_Nbit>(add_res, output_data, i);
  }
}

template <typename in1_type, typename in2_type, typename out_type>
inline int32_t AddFunc_add_STAR(in1_type x, in2_type y, const ArithmeticParams& params) {
  const int32_t input1_val = params.input1_offset + x;
  const int32_t input2_val = params.input2_offset + y;
  const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
  const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
  const int32_t scaled_input1_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input1_val, params.input1_multiplier, params.input1_shift);
  const int32_t scaled_input2_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input2_val, params.input2_multiplier, params.input2_shift);
  const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
  const int32_t raw_output =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          raw_sum, params.output_multiplier, params.output_shift) +
      params.output_offset;
  const int32_t clamped_output =
      std::min(params.quantized_activation_max,
               std::max(params.quantized_activation_min, raw_output));
  return clamped_output;
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
template <typename in1_type, typename in2_type, typename out_type, int in1_Nbit, int in2_Nbit, int out_Nbit>
inline void AddElementwise_add_STAR(int size, const ArithmeticParams& params,
                           const in1_type* input1_data, const in2_type* input2_data,
                           out_type* output_data) {
  ElementWise_add_STAR<in1_type, in2_type, out_type, out_Nbit>(size, params, input1_data, input2_data, output_data,
              CheckTwoArithmeticParams_add_STAR<in1_Nbit, in2_Nbit>, AddFunc_add_STAR<in1_type, in2_type, out_type>);
}

template <typename in1_type, typename in2_type, typename out_type, int in1_Nbit, int in2_Nbit, int out_Nbit>
void __attribute__((noinline)) Add_STAR(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const in1_type* input1_data,
                const RuntimeShape& input2_shape, const in2_type* input2_data,
                const RuntimeShape& output_shape, out_type* output_data) {
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  AddElementwise_add_STAR<in1_type, in2_type, out_type, in1_Nbit, in2_Nbit, out_Nbit>(flat_size, params, input1_data, input2_data, output_data);
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_STAR_ADD_H_
