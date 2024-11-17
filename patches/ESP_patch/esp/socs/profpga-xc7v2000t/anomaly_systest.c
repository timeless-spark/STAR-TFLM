#include "anomaly_model_data.h"
#include "anomaly_test_data.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "ibex_utility.h"
#include "input_quantize.h"

extern "C"
{
void* __dso_handle = NULL;
}

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;
int correct_count = 0;
uint64_t average_cycles = 0;

constexpr int kTensorArenaSize = 256 * 1024;
alignas(32) uint8_t tensor_arena[kTensorArenaSize];

constexpr int skip_first_inferences = 5;

int local_test_samples = 15;

}  // namespace

void setup() {
  
  model = tflite::GetModel(anomaly_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return;
  }

  static tflite::MicroMutableOpResolver<1> micro_op_resolver;
#ifdef STAR
  micro_op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_STAR());
#else
  micro_op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8_PER_CHANNEL());
#endif

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);

  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

void __attribute__((noinline)) __attribute__((optimize("O0"))) loop() {

  if (input->type == kTfLiteInt16)
    GenInput_16(inference_count, input_size, (int16_t*)input->data.data, input->params.scale, input->params.zero_point);
  else if (input->type == kTfLiteInt8)
    GenInput_8( inference_count, input_size, (int8_t*)input->data.data, input->params.scale, input->params.zero_point);
  else if (input->type == kTfLiteInt4)
    GenInput_4( inference_count, input_size, (int8_t*)input->data.data, input->params.scale, input->params.zero_point);

  disable_counter();

  reset_counter();

  #ifdef MEASURE_GLOBAL
  enable_counter();
  #endif

  TfLiteStatus invoke_status = interpreter->Invoke();

  #ifdef MEASURE_GLOBAL
  disable_counter();
  #endif

  uint64_t cycle_count = read_counter();

  if (invoke_status != kTfLiteOk) {
    return;
  }

  // Obtain the quantized output from model's output tensor
  int16_t y_q[640];
  if (output->type == kTfLiteInt4)
  {
    for (int i=0; i<320; ++i)
    {
      int8_t data_i = static_cast<int8_t*>(output->data.data)[i];
      y_q[2*i    ] = static_cast<int16_t>((int8_t)((data_i & 0x0F) < 0x08 ? (data_i & 0x0F) : (data_i | 0xF0)));
      y_q[2*i + 1] = static_cast<int16_t>((int8_t)(data_i >> 4));
    }
  }
  else
  {
    for (int i=0; i<640; ++i)
    {
      if (output->type == kTfLiteInt16)
        y_q[i] = static_cast<int16_t*>(output->data.data)[i];
      else
        y_q[i] = static_cast<int8_t*>(output->data.data)[i];
    }
  }

  // Dequantize the output from integer to floating-point
  double y[640];
  for (int i=0; i<640; ++i)
  {
    y[i] = (y_q[i] - output->params.zero_point) * output->params.scale;
  }

  if (inference_count == local_test_samples-1)
    printf("\n n_cycles (last) = %llu\n", cycle_count);

  if (inference_count>=skip_first_inferences) {
    uint64_t cycle_added = cycle_count;
    average_cycles += cycle_added;
  }

  ++inference_count;

  return;
}

int main() {

  printf("Test-set inference\n");

  setup();

  for (int i=0; i<local_test_samples; ++i) {
    loop();
  }

  printf("\naverage_cycles final = %llu\n", (uint64_t)((double)(average_cycles) / (double)(local_test_samples-skip_first_inferences)));

  printf("\n-------------------------------------------------------------\n");

  return 0;
}
