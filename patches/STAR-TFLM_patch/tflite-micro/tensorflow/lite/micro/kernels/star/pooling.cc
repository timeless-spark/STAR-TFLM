#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/padding.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "tensorflow/lite/kernels/internal/reference/integer_ops/star/pooling.h"

#include "tensorflow/lite/micro/ibex_rv32im/ibex_utility.h"

namespace tflite {

namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataPooling));
}

TfLiteStatus AverageEval_STAR(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPooling* data =
      static_cast<const OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  if (input->type == kTfLiteInt16 || input->type == kTfLiteInt8 || input->type == kTfLiteInt4) {
      AveragePoolingEvalQuantized_STAR(context, node, params, data, input, output);
  } else {
      MicroPrintf("Input type %s is not currently supported",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval_STAR(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPooling* data =
      static_cast<const OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);
    
  if (input->type != output->type) {
    MicroPrintf("Only config with the same Input and Output type are supported");
      return kTfLiteError;
  }

  if (input->type == kTfLiteInt16 || input->type == kTfLiteInt8 || input->type == kTfLiteInt4) {
      MaxPoolingEvalQuantized_STAR(context, node, params, data, input, output);
  } else {
      MicroPrintf("Input type %s is not currently supported",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

void AveragePoolingEvalQuantized_STAR(TfLiteContext* context, const TfLiteNode* node,
                                 const TfLitePoolParams* params,
                                 const OpDataPooling* data,
                                 const TfLiteEvalTensor* input,
                                 TfLiteEvalTensor* output) {
  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;
  op_params.input_offset = -data->input_zero_point;
  op_params.output_offset = data->output_zero_point;

  switch (output->type) {
    case kTfLiteInt16: {
      switch (input->type) {
        case kTfLiteInt16: {
            // template <typename in_type, typename out_type, int out_Nbit>
            reference_integer_ops::AveragePool_STAR<int16_t, int16_t, 16>(
                            op_params, data->output_multiplier, data->output_shift,
                            tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<int16_t>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int16_t>(output));
            break;
        }
        case kTfLiteInt8: {
            reference_integer_ops::AveragePool_STAR<int8_t, int16_t, 16>(
                            op_params, data->output_multiplier, data->output_shift,
                            tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<int8_t>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int16_t>(output));
            break;
        }
        case kTfLiteInt4: {
            #ifdef MEASURE_UNPACK
            enable_counter();
            #endif
            int8_t* unpacked_input_data = nullptr;
            OpDataPooling* op_data = static_cast<OpDataPooling*>(node->user_data);
            unpacked_input_data = static_cast<int8_t*>(
                            context->GetScratchBuffer(context, op_data->input_buffer_index));
            tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                            tflite::micro::GetTensorData<int8_t>(input), 
                            tflite::micro::GetTensorShape(input).FlatSize(), 
                            unpacked_input_data);
            #ifdef MEASURE_UNPACK
            disable_counter();
            #endif
            reference_integer_ops::AveragePool_STAR<int8_t, int16_t, 16>(
                            op_params, data->output_multiplier, data->output_shift,
                            tflite::micro::GetTensorShape(input),
                            reinterpret_cast<int8_t*>(unpacked_input_data),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int16_t>(output));
            break;
        }
        default: {
            break;
        }
      }
      break;
    }
    case kTfLiteInt8: {
      switch (input->type) {
        case kTfLiteInt16: {
            // template <typename in_type, typename out_type, int out_Nbit>
            reference_integer_ops::AveragePool_STAR<int16_t, int8_t, 8>(
                            op_params, data->output_multiplier, data->output_shift,
                            tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<int16_t>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int8_t>(output));
            break;
        }
        case kTfLiteInt8: {
            reference_integer_ops::AveragePool_STAR<int8_t, int8_t, 8>(
                            op_params, data->output_multiplier, data->output_shift,
                            tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<int8_t>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int8_t>(output));
            break;
        }
        case kTfLiteInt4: {
            #ifdef MEASURE_UNPACK
            enable_counter();
            #endif
            int8_t* unpacked_input_data = nullptr;
            OpDataPooling* op_data = static_cast<OpDataPooling*>(node->user_data);
            unpacked_input_data = static_cast<int8_t*>(
                            context->GetScratchBuffer(context, op_data->input_buffer_index));
            tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                            tflite::micro::GetTensorData<int8_t>(input), 
                            tflite::micro::GetTensorShape(input).FlatSize(), 
                            unpacked_input_data);
            #ifdef MEASURE_UNPACK
            disable_counter();
            #endif
            reference_integer_ops::AveragePool_STAR<int8_t, int8_t, 8>(
                            op_params, data->output_multiplier, data->output_shift,
                            tflite::micro::GetTensorShape(input),
                            reinterpret_cast<int8_t*>(unpacked_input_data),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int8_t>(output));
            break;
        }
        default: {
            break;
        }
      }
      break;
    }
    case kTfLiteInt4: {
      switch (input->type) {
        case kTfLiteInt16: {
            reference_integer_ops::AveragePool_STAR<int16_t, int8_t, 4>(
                            op_params, data->output_multiplier, data->output_shift,
                            tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<int16_t>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int8_t>(output));
            break;
        }
        case kTfLiteInt8: {
            reference_integer_ops::AveragePool_STAR<int8_t, int8_t, 4>(
                            op_params, data->output_multiplier, data->output_shift,
                            tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<int8_t>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int8_t>(output));
            break;
        }
        case kTfLiteInt4: {
            #ifdef MEASURE_UNPACK
            enable_counter();
            #endif
            int8_t* unpacked_input_data = nullptr;
            OpDataPooling* op_data = static_cast<OpDataPooling*>(node->user_data);
            unpacked_input_data = static_cast<int8_t*>(
                            context->GetScratchBuffer(context, op_data->input_buffer_index));
            tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                            tflite::micro::GetTensorData<int8_t>(input), 
                            tflite::micro::GetTensorShape(input).FlatSize(), 
                            unpacked_input_data);
            #ifdef MEASURE_UNPACK
            disable_counter();
            #endif
            reference_integer_ops::AveragePool_STAR<int8_t, int8_t, 4>(
                            op_params, data->output_multiplier, data->output_shift,
                            tflite::micro::GetTensorShape(input),
                            reinterpret_cast<int8_t*>(unpacked_input_data),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<int8_t>(output));
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
}

void MaxPoolingEvalQuantized_STAR(TfLiteContext* context, TfLiteNode* node,
                             TfLitePoolParams* params,
                             const OpDataPooling* data,
                             const TfLiteEvalTensor* input,
                             TfLiteEvalTensor* output) {
  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  switch (input->type) {
    case kTfLiteInt16: {
        // template <typename in_type, typename out_type, int out_Nbit>
        reference_integer_ops::MaxPool_STAR<int16_t, int16_t, 16>(
                        op_params, tflite::micro::GetTensorShape(input),
                        tflite::micro::GetTensorData<int16_t>(input),
                        tflite::micro::GetTensorShape(output),
                        tflite::micro::GetTensorData<int16_t>(output));
        break;
    }
    case kTfLiteInt8: {
        reference_integer_ops::MaxPool_STAR<int8_t, int8_t, 8>(
                        op_params, tflite::micro::GetTensorShape(input),
                        tflite::micro::GetTensorData<int8_t>(input),
                        tflite::micro::GetTensorShape(output),
                        tflite::micro::GetTensorData<int8_t>(output));
        break;
    }
    case kTfLiteInt4: {
        #ifdef MEASURE_UNPACK
        enable_counter();
        #endif
        int8_t* unpacked_input_data = nullptr;
        OpDataPooling* op_data = static_cast<OpDataPooling*>(node->user_data);
        unpacked_input_data = static_cast<int8_t*>(
                        context->GetScratchBuffer(context, op_data->input_buffer_index));
        tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                        tflite::micro::GetTensorData<int8_t>(input), 
                        tflite::micro::GetTensorShape(input).FlatSize(), 
                        unpacked_input_data);
        #ifdef MEASURE_UNPACK
        disable_counter();
        #endif
        // template <typename in_type, typename out_type, int out_Nbit>
        reference_integer_ops::MaxPool_STAR<int8_t, int8_t, 4>(
                        op_params, tflite::micro::GetTensorShape(input),
                        reinterpret_cast<int8_t*>(unpacked_input_data),
                        tflite::micro::GetTensorShape(output),
                        tflite::micro::GetTensorData<int8_t>(output));
        break;
    }
    default: {
        break;
    }
  }
}

TfLiteRegistration Register_AVERAGE_POOL_2D_STAR() {
  return tflite::micro::RegisterOp(Init, PoolingPrepare, AverageEval_STAR);
}

TfLiteRegistration Register_MAX_POOL_2D_STAR() {
  return tflite::micro::RegisterOp(Init, PoolingPrepare, MaxEval_STAR);
}

}  // namespace tflite
