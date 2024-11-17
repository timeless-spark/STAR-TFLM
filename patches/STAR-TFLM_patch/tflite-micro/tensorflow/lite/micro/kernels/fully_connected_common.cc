/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

const int kFullyConnectedInputTensor = 0;
const int kFullyConnectedWeightsTensor = 1;
const int kFullyConnectedBiasTensor = 2;
const int kFullyConnectedOutputTensor = 0;

#if defined(STAR) || defined(FC_SCALE_PER_CHANNEL)
const int kFullyConnectedQuantizedDimension = 0;
#endif

FullyConnectedParams FullyConnectedParamsQuantized(
    const OpDataFullyConnected& op_data) {
  FullyConnectedParams op_params;
  op_params.input_offset = -op_data.input_zero_point;
  op_params.weights_offset = -op_data.filter_zero_point;
  op_params.output_offset = op_data.output_zero_point;
  op_params.output_multiplier = op_data.output_multiplier;
  op_params.output_shift = op_data.output_shift;
  op_params.quantized_activation_min = op_data.output_activation_min;
  op_params.quantized_activation_max = op_data.output_activation_max;
  return op_params;
}

FullyConnectedParams FullyConnectedParamsFloat(
    TfLiteFusedActivation activation) {
  FullyConnectedParams op_params;
  CalculateActivationRange(activation, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  return op_params;
}

TfLiteStatus CalculateOpDataFullyConnected(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    OpDataFullyConnected* data) {
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;

#if defined(STAR) || defined(FC_SCALE_PER_CHANNEL)

int num_channels = filter->dims->data[kFullyConnectedQuantizedDimension];

// Check data type.
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  const bool is_per_channel = affine_quantization->scale->size > 1;
  if (is_per_channel) {
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size, num_channels);
    TF_LITE_ENSURE_EQ(
        context, num_channels,
        filter->dims->data[affine_quantization->quantized_dimension]);
  }

  // Populate multiplier and shift using affine quantization.
  const float input_scale = input->params.scale;
  const float output_scale = output->params.scale;
  const float* filter_scales = affine_quantization->scale->data;
  for (int i = 0; i < num_channels; ++i) {
    // If per-tensor quantization parameter is specified, broadcast it along the
    // quantization dimension (channels_out).
    const float scale = is_per_channel ? filter_scales[i] : filter_scales[0];
    const double filter_scale = static_cast<double>(scale);
    const double effective_output_scale = static_cast<double>(input_scale) *
                                          filter_scale /
                                          static_cast<double>(output_scale);
    int32_t significand;
    int channel_shift;
    QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
    data->per_channel_output_multiplier[i] = significand;
    data->per_channel_output_shift[i] = channel_shift;
  }

  data->input_zero_point = input->params.zero_point;
  TFLITE_DCHECK(filter->params.zero_point == 0);
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8 ||
      input->type == kTfLiteInt16 || input->type == kTfLiteInt4) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }

#else
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);

    data->input_zero_point = input->params.zero_point;
    // Filter weights will always be symmetric quantized since we only support
    // int8 quantization. See
    // https://github.com/tensorflow/tensorflow/issues/44912 for additional
    // context.
    TFLITE_DCHECK(filter->params.zero_point == 0);
    data->filter_zero_point = filter->params.zero_point;
    data->output_zero_point = output->params.zero_point;

    return CalculateActivationRangeQuantized(context, activation, output,
                                             &data->output_activation_min,
                                             &data->output_activation_max);
#endif
  }
  return kTfLiteOk;
}

}  // namespace tflite
