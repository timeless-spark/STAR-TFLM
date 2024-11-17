#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "tensorflow/lite/kernels/internal/reference/integer_ops/star/fully_connected.h"

#include "tensorflow/lite/micro/ibex_rv32im/ibex_utility.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  
  // Dynamically allocate per-channel quantization parameters.
  const int num_channels = filter->dims->data[kFullyConnectedQuantizedDimension];
  data->per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  switch (input->type) {
    case kTfLiteInt16: {
      switch (filter->type) {
        case kTfLiteInt8: {
          int filter_size =
              RuntimeShape(filter->dims->size,
                          reinterpret_cast<const int32_t*>(filter->dims->data))
                  .FlatSize();
          int buffer_size = 2*filter_size;
          context->RequestScratchBufferInArena(context, buffer_size,
                                               &data->filter_buffer_index);
          break;
        }
        case kTfLiteInt4: {
          int filter_size =
              RuntimeShape(filter->dims->size,
                          reinterpret_cast<const int32_t*>(filter->dims->data))
                  .FlatSize();
          int buffer_size = 2*filter_size;
          context->RequestScratchBufferInArena(context, buffer_size,
                                               &data->filter_buffer_index);
          break;
        }
        default: {
          break;
        }
      }
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt16: {
          int input_size =
              RuntimeShape(input->dims->size,
                          reinterpret_cast<const int32_t*>(input->dims->data))
                  .FlatSize();
          int buffer_size = 2*input_size;
          context->RequestScratchBufferInArena(context, buffer_size,
                                               &data->input_buffer_index);
          break;
        }
        case kTfLiteInt4: {
          int filter_size =
              RuntimeShape(filter->dims->size,
                          reinterpret_cast<const int32_t*>(filter->dims->data))
                  .FlatSize();
          int buffer_size = filter_size;
          context->RequestScratchBufferInArena(context, buffer_size,
                                               &data->filter_buffer_index);
          break;
        }
        default: {
          break;
        }
      }
      break;
    }
    case kTfLiteInt4: {
      switch (filter->type) {
        case kTfLiteInt16: {
          int input_size =
              RuntimeShape(input->dims->size,
                          reinterpret_cast<const int32_t*>(input->dims->data))
                  .FlatSize();
          int buffer_size = 2*input_size;
          context->RequestScratchBufferInArena(context, buffer_size,
                                               &data->input_buffer_index);
          break;
        }
        case kTfLiteInt8: {
          int input_size =
              RuntimeShape(input->dims->size,
                          reinterpret_cast<const int32_t*>(input->dims->data))
                  .FlatSize();
          int buffer_size = input_size;
          context->RequestScratchBufferInArena(context, buffer_size,
                                               &data->input_buffer_index);
          break;
        }
        default: {
          break;
        }
      }
      break;
    }
    default: {
      break;
    }
  }

  TF_LITE_ENSURE_OK(context, CalculateOpDataFullyConnected(
                                 context, params->activation, input->type,
                                 input, filter, bias, output, data));

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus Eval_STAR(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);

  const auto& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));

  switch (output->type) {
    case kTfLiteInt16: {
      switch (input->type) {
        case kTfLiteInt16: {
          switch (filter->type) {
            case kTfLiteInt16: {
              // template<int OP_parall, typename out_type, int out_Nbit>
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int16_t, 16>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_filter_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int16_t, 16>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_filter_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int16_t, 16>(
                  FullyConnectedParamsQuantized(data),
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
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int16_t* unpacked_input_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int16_t, 16>(
                  FullyConnectedParamsQuantized(data),
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
              reference_integer_ops::FullyConnectedPerChannel_STAR<8, int16_t, 16>(
                  FullyConnectedParamsQuantized(data),
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
              int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<8, int16_t, 16>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_input_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int16_t, 16>(
                  FullyConnectedParamsQuantized(data),
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
              int8_t* unpacked_input_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<8, int16_t, 16>(
                  FullyConnectedParamsQuantized(data),
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
              reference_integer_ops::FullyConnectedPerChannel_STAR<4, int16_t, 16>(
                  FullyConnectedParamsQuantized(data),
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
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 8>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_filter_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 8>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_filter_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 8>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_input_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 8>(
                  FullyConnectedParamsQuantized(data),
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
              reference_integer_ops::FullyConnectedPerChannel_STAR<8, int8_t, 8>(
                  FullyConnectedParamsQuantized(data),
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
              int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<8, int8_t, 8>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_input_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 8>(
                  FullyConnectedParamsQuantized(data),
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
              int8_t* unpacked_input_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<8, int8_t, 8>(
                  FullyConnectedParamsQuantized(data),
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
              reference_integer_ops::FullyConnectedPerChannel_STAR<4, int8_t, 8>(
                  FullyConnectedParamsQuantized(data),
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
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 4>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_filter_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 4>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_filter_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 4>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_input_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt8IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 4>(
                  FullyConnectedParamsQuantized(data),
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
              reference_integer_ops::FullyConnectedPerChannel_STAR<8, int8_t, 4>(
                  FullyConnectedParamsQuantized(data),
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
              int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(filter), 
                  tflite::micro::GetTensorShape(filter).FlatSize(), 
                  unpacked_filter_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<8, int8_t, 4>(
                  FullyConnectedParamsQuantized(data),
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
              int16_t* unpacked_input_data = static_cast<int16_t*>(
              context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt16(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<16, int8_t, 4>(
                  FullyConnectedParamsQuantized(data),
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
              // unpack to 8
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.input_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input),
                  tflite::micro::GetTensorShape(input).FlatSize(), 
                  unpacked_input_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::FullyConnectedPerChannel_STAR<8, int8_t, 4>(
                  FullyConnectedParamsQuantized(data),
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
              reference_integer_ops::FullyConnectedPerChannel_STAR<4, int8_t, 4>(
                  FullyConnectedParamsQuantized(data),
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

TfLiteRegistration Register_FULLY_CONNECTED_STAR() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval_STAR);
}

}  // namespace tflite
