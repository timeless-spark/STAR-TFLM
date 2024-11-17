#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "tensorflow/lite/kernels/internal/reference/integer_ops/star/conv.h"

#include "tensorflow/lite/micro/ibex_rv32im/ibex_utility.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

TfLiteStatus Eval_STAR(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

  switch (output->type) {
    case kTfLiteInt16: {
      switch (input->type) {
        case kTfLiteInt16: {
          switch (filter->type) {
            case kTfLiteInt16: {
              // template<int OP_parall, typename out_type, int out_Nbit>
              reference_integer_ops::ConvPerChannel_STAR<16, int16_t, 16>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt8: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_filter_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_filter_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, op_data->filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int16_t, 16>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  reinterpret_cast<int32_t*>(unpacked_filter_data),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt4: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_filter_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_filter_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, op_data->filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int16_t, 16>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  reinterpret_cast<int32_t*>(unpacked_filter_data),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            default: {
              return kTfLiteError;
              break;
            }
          }
          break;
        }
        case kTfLiteInt8: {
          switch (filter->type) {
            case kTfLiteInt16: {
              // unpack to 16
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_input_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_input_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int16_t, 16>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  reinterpret_cast<int32_t*>(unpacked_input_data),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt8: {
              reference_integer_ops::ConvPerChannel_STAR<8, int16_t, 16>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt4: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_filter_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_filter_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<8, int16_t, 16>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  reinterpret_cast<int32_t*>(unpacked_filter_data),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            default: {
              return kTfLiteError;
              break;
            }
          }
          break;
        }
        case kTfLiteInt4: {
          switch (filter->type) {
            case kTfLiteInt16: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_input_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_input_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int16_t, 16>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  reinterpret_cast<int32_t*>(unpacked_input_data),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt8: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_input_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<8, int16_t, 16>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  reinterpret_cast<int32_t*>(unpacked_input_data),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt4: {
              reference_integer_ops::ConvPerChannel_STAR<4, int16_t, 16>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            default: {
              return kTfLiteError;
              break;
            }
          }
          break;
        }
        default: {
          return kTfLiteError;
          break;
        }
      }
      break;
    }
    case kTfLiteInt8: {
      switch (input->type) {
        case kTfLiteInt16: {
          switch (filter->type) {
            case kTfLiteInt16: {
              // template<int OP_parall, typename out_type, int out_Nbit>
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 8>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_filter_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_filter_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, op_data->filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 8>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  reinterpret_cast<int32_t*>(unpacked_filter_data),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_filter_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_filter_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, op_data->filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 8>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  reinterpret_cast<int32_t*>(unpacked_filter_data),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            default: {
              return kTfLiteError;
              break;
            }
          }
          break;
        }
        case kTfLiteInt8: {
          switch (filter->type) {
            case kTfLiteInt16: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_input_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_input_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 8>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  reinterpret_cast<int32_t*>(unpacked_input_data),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              reference_integer_ops::ConvPerChannel_STAR<8, int8_t, 8>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_filter_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_filter_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<8, int8_t, 8>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  reinterpret_cast<int32_t*>(unpacked_filter_data),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            default: {
              return kTfLiteError;
              break;
            }
          }
          break;
        }
        case kTfLiteInt4: {
          switch (filter->type) {
            case kTfLiteInt16: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_input_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_input_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 8>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  reinterpret_cast<int32_t*>(unpacked_input_data),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_input_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<8, int8_t, 8>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  reinterpret_cast<int32_t*>(unpacked_input_data),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              reference_integer_ops::ConvPerChannel_STAR<4, int8_t, 8>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            default: {
              return kTfLiteError;
              break;
            }
          }
          break;
        }
        default: {
          return kTfLiteError;
          break;
        }
      }
      break;
    }
    case kTfLiteInt4: {
      switch (input->type) {
        case kTfLiteInt16: {
          switch (filter->type) {
            case kTfLiteInt16: {
              // template<int OP_parall, typename out_type, int out_Nbit>
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 4>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_filter_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_filter_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, op_data->filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 4>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  reinterpret_cast<int32_t*>(unpacked_filter_data),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_filter_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_filter_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, op_data->filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 4>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  reinterpret_cast<int32_t*>(unpacked_filter_data),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            default: {
              return kTfLiteError;
              break;
            }
          }
          break;
        }
        case kTfLiteInt8: {
          switch (filter->type) {
            case kTfLiteInt16: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_input_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_input_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 4>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  reinterpret_cast<int32_t*>(unpacked_input_data),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              reference_integer_ops::ConvPerChannel_STAR<8, int8_t, 4>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_filter_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_filter_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<8, int8_t, 4>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  reinterpret_cast<int32_t*>(unpacked_filter_data),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            default: {
              return kTfLiteError;
              break;
            }
          }
          break;
        }
        case kTfLiteInt4: {
          switch (filter->type) {
            case kTfLiteInt16: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_input_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_input_data = static_cast<int16_t*>(
                  context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<16, int8_t, 4>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  reinterpret_cast<int32_t*>(unpacked_input_data),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input_data = nullptr;
              OpDataConv* op_data = static_cast<OpDataConv*>(node->user_data);
              unpacked_input_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::ConvPerChannel_STAR<8, int8_t, 4>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  reinterpret_cast<int32_t*>(unpacked_input_data),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              reference_integer_ops::ConvPerChannel_STAR<4, int8_t, 4>(
                  ConvParamsQuantized(params, data),
                  data.per_channel_output_multiplier, data.per_channel_output_shift,
                  tflite::micro::GetTensorShape(input),
                  tflite::micro::GetTensorData<int32_t>(input),
                  tflite::micro::GetTensorShape(filter),
                  tflite::micro::GetTensorData<int32_t>(filter),
                  tflite::micro::GetTensorShape(bias),
                  tflite::micro::GetOptionalTensorData<int32_t>(bias),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            default: {
              return kTfLiteError;
              break;
            }
          }
          break;
        }
        default: {
          return kTfLiteError;
          break;
        }
      }
      break;
    }
    default: {
      return kTfLiteError;
      break;
    }
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_CONV_2D_STAR() {
  return tflite::micro::RegisterOp(Init, ConvPrepare, Eval_STAR);
}

}  // namespace tflite
