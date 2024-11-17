#include "tensorflow/lite/kernels/internal/reference/add.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/add.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "tensorflow/lite/kernels/internal/reference/integer_ops/star/add.h"

#include "tensorflow/lite/micro/ibex_rv32im/ibex_utility.h"

namespace tflite {

void* AddInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataAdd));
}

TfLiteStatus AddEval_STAR(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteAddParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataAdd* data = static_cast<const OpDataAdd*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kAddInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kAddInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kAddOutputTensor);

  tflite::ArithmeticParams op_params;
  op_params.left_shift = data->left_shift;
  op_params.input1_offset = data->input1_offset;
  op_params.input1_multiplier = data->input1_multiplier;
  op_params.input1_shift = data->input1_shift;
  op_params.input2_offset = data->input2_offset;
  op_params.input2_multiplier = data->input2_multiplier;
  op_params.input2_shift = data->input2_shift;
  op_params.output_offset = data->output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  SetActivationParams(data->output_activation_min, data->output_activation_max,
                      &op_params);

  switch (output->type) {
    case kTfLiteInt16: {
      switch (input1->type) {
        case kTfLiteInt16: {
          switch (input2->type) {
            case kTfLiteInt16: {
              // in1  in2  out
              // 16 + 16 = 16
              reference_integer_ops::Add_STAR<int16_t, int16_t, int16_t, 16, 16, 16>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int16_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int16_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt8: {
              // 16 +  8 = 16
              reference_integer_ops::Add_STAR<int16_t, int8_t, int16_t, 16, 8, 16>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int16_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int8_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt4: {
              // 16 +  4 = 16
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input2_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input2_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input2_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input2), 
                  tflite::micro::GetTensorShape(input2).FlatSize(), 
                  unpacked_input2_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int16_t, int8_t, int16_t, 16, 4, 16>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  tflite::micro::GetTensorData<int16_t>(input1),
                  tflite::micro::GetTensorShape(input2),
                  unpacked_input2_data,
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
          switch (input2->type) {
            case kTfLiteInt16: {
              // in1  in2  out
              //  8 + 16 = 16
              reference_integer_ops::Add_STAR<int8_t, int16_t, int16_t, 8, 16, 16>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int8_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int16_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt8: {
              //  8 +  8 = 16
              reference_integer_ops::Add_STAR<int8_t, int8_t, int16_t, 8, 8, 16>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int8_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int8_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt4: {
              //  8 +  4 = 16
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input2_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input2_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input2_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input2), 
                  tflite::micro::GetTensorShape(input2).FlatSize(), 
                  unpacked_input2_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int8_t, int16_t, 8, 4, 16>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input2),
                  unpacked_input2_data,
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
          switch (input2->type) {
            case kTfLiteInt16: {
              // in1  in2  out
              //  4 + 16 = 16
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input1_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input1_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input1_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input1).FlatSize(), 
                  unpacked_input1_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int16_t, int16_t, 4, 16, 16>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  unpacked_input1_data,
                  tflite::micro::GetTensorShape(input2),
                  tflite::micro::GetTensorData<int16_t>(input2),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt8: {
              //  4 +  8 = 16
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input1_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input1_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input1_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input1).FlatSize(), 
                  unpacked_input1_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int8_t, int16_t, 4, 8, 16>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  unpacked_input1_data,
                  tflite::micro::GetTensorShape(input2),
                  tflite::micro::GetTensorData<int8_t>(input2),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int16_t>(output));
              break;
            }
            case kTfLiteInt4: {
              //  4 +  4 = 16
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input1_data = nullptr;
              int8_t* unpacked_input2_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input1_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input1_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input1).FlatSize(), 
                  unpacked_input1_data);
              unpacked_input2_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input2_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input2), 
                  tflite::micro::GetTensorShape(input2).FlatSize(), 
                  unpacked_input2_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int8_t, int16_t, 4, 4, 16>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  unpacked_input1_data,
                  tflite::micro::GetTensorShape(input2),
                  unpacked_input2_data,
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
      switch (input1->type) {
        case kTfLiteInt16: {
          switch (input2->type) {
            case kTfLiteInt16: {
              // in1  in2  out
              // 16 + 16 =  8
              reference_integer_ops::Add_STAR<int16_t, int16_t, int8_t, 16, 16, 8>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int16_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int16_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              // 16 +  8 =  8
              reference_integer_ops::Add_STAR<int16_t, int8_t, int8_t, 16, 8, 8>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int16_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int8_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              // 16 +  4 =  8
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input2_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input2_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input2_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input2), 
                  tflite::micro::GetTensorShape(input2).FlatSize(), 
                  unpacked_input2_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int16_t, int8_t, int8_t, 16, 4, 8>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  tflite::micro::GetTensorData<int16_t>(input1),
                  tflite::micro::GetTensorShape(input2),
                  unpacked_input2_data,
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
          switch (input2->type) {
            case kTfLiteInt16: {
              // in1  in2  out
              //  8 + 16 =  8
              reference_integer_ops::Add_STAR<int8_t, int16_t, int8_t, 8, 16, 8>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int8_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int16_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              //  8 +  8 =  8
              reference_integer_ops::Add_STAR<int8_t, int8_t, int8_t, 8, 8, 8>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int8_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int8_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              //  8 +  4 =  8
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input2_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input2_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input2_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input2), 
                  tflite::micro::GetTensorShape(input2).FlatSize(), 
                  unpacked_input2_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int8_t, int8_t, 8, 4, 8>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input2),
                  unpacked_input2_data,
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
          switch (input2->type) {
            case kTfLiteInt16: {
              // in1  in2  out
              //  4 + 16 =  8
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input1_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input1_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input1_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input1).FlatSize(), 
                  unpacked_input1_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int16_t, int8_t, 4, 16, 8>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  unpacked_input1_data,
                  tflite::micro::GetTensorShape(input2),
                  tflite::micro::GetTensorData<int16_t>(input2),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              //  4 +  8 =  8
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input1_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input1_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input1_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input1).FlatSize(), 
                  unpacked_input1_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int8_t, int8_t, 4, 8, 8>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  unpacked_input1_data,
                  tflite::micro::GetTensorShape(input2),
                  tflite::micro::GetTensorData<int8_t>(input2),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              //  4 +  4 =  8
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input1_data = nullptr;
              int8_t* unpacked_input2_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input1_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input1_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input1).FlatSize(), 
                  unpacked_input1_data);
              unpacked_input2_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input2_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input2), 
                  tflite::micro::GetTensorShape(input2).FlatSize(), 
                  unpacked_input2_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int8_t, int8_t, 4, 4, 8>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  unpacked_input1_data,
                  tflite::micro::GetTensorShape(input2),
                  unpacked_input2_data,
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
      switch (input1->type) {
        case kTfLiteInt16: {
          switch (input2->type) {
            case kTfLiteInt16: {
              // in1  in2  out
              // 16 + 16 =  4
              reference_integer_ops::Add_STAR<int16_t, int16_t, int8_t, 16, 16, 4>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int16_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int16_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              // 16 +  8 =  4
              reference_integer_ops::Add_STAR<int16_t, int8_t, int8_t, 16, 8, 4>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int16_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int8_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              // 16 +  4 =  4
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input2_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input2_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input2_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input2), 
                  tflite::micro::GetTensorShape(input2).FlatSize(), 
                  unpacked_input2_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int16_t, int8_t, int8_t, 16, 4, 4>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  tflite::micro::GetTensorData<int16_t>(input1),
                  tflite::micro::GetTensorShape(input2),
                  unpacked_input2_data,
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
          switch (input2->type) {
            case kTfLiteInt16: {
              // in1  in2  out
              //  8 + 16 =  4
              reference_integer_ops::Add_STAR<int8_t, int16_t, int8_t, 8, 16, 4>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int8_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int16_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              //  8 +  8 =  4
              reference_integer_ops::Add_STAR<int8_t, int8_t, int8_t, 8, 8, 4>(
                      op_params, tflite::micro::GetTensorShape(input1),
                      tflite::micro::GetTensorData<int8_t>(input1),
                      tflite::micro::GetTensorShape(input2),
                      tflite::micro::GetTensorData<int8_t>(input2),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              //  8 +  4 =  4
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input2_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input2_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input2_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input2), 
                  tflite::micro::GetTensorShape(input2).FlatSize(), 
                  unpacked_input2_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int8_t, int8_t, 8, 4, 4>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input2),
                  unpacked_input2_data,
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
          switch (input2->type) {
            case kTfLiteInt16: {
              // in1  in2  out
              //  4 + 16 =  4
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input1_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input1_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input1_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input1).FlatSize(),
                  unpacked_input1_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int16_t, int8_t, 4, 16, 4>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  unpacked_input1_data,
                  tflite::micro::GetTensorShape(input2),
                  tflite::micro::GetTensorData<int16_t>(input2),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt8: {
              //  4 +  8 =  4
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input1_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input1_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input1_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input1).FlatSize(), 
                  unpacked_input1_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int8_t, int8_t, 4, 8, 4>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  unpacked_input1_data,
                  tflite::micro::GetTensorShape(input2),
                  tflite::micro::GetTensorData<int8_t>(input2),
                  tflite::micro::GetTensorShape(output),
                  tflite::micro::GetTensorData<int8_t>(output));
              break;
            }
            case kTfLiteInt4: {
              //  4 +  4 =  4
              #ifdef MEASURE_UNPACK
              enable_counter();
              #endif
              int8_t* unpacked_input1_data = nullptr;
              int8_t* unpacked_input2_data = nullptr;
              OpDataAdd* op_data = static_cast<OpDataAdd*>(node->user_data);
              unpacked_input1_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input1_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input1),
                  tflite::micro::GetTensorShape(input1).FlatSize(), 
                  unpacked_input1_data);
              unpacked_input2_data = static_cast<int8_t*>(
                  context->GetScratchBuffer(context, op_data->input2_buffer_index));
              tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                  tflite::micro::GetTensorData<int8_t>(input2), 
                  tflite::micro::GetTensorShape(input2).FlatSize(), 
                  unpacked_input2_data);
              #ifdef MEASURE_UNPACK
              disable_counter();
              #endif
              reference_integer_ops::Add_STAR<int8_t, int8_t, int8_t, 4, 4, 4>(
                  op_params, tflite::micro::GetTensorShape(input1),
                  unpacked_input1_data,
                  tflite::micro::GetTensorShape(input2),
                  unpacked_input2_data,
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

TfLiteRegistration Register_ADD_STAR() {
  return tflite::micro::RegisterOp(AddInit, AddPrepare, AddEval_STAR);
}

}  // namespace tflite
