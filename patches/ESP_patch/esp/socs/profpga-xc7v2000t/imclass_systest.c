#include "imclass_model_data.h"
#include "imclass_test_data.h"

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

constexpr int kTensorArenaSize = 128 * 1024;
alignas(32) uint8_t tensor_arena[kTensorArenaSize];

constexpr int skip_first_inferences = 5;

int local_test_samples = 15;

}  // namespace

void setup() {
  
  model = tflite::GetModel(imclass_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return;
  }

  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
#ifdef STAR
  micro_op_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_STAR());
  micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_STAR());
  micro_op_resolver.AddAdd(tflite::Register_ADD_STAR());
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_STAR());
  micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX());
#else
  micro_op_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
  micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
  micro_op_resolver.AddAdd(tflite::Register_ADD_INT8());
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8_PER_CHANNEL());
  micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX());
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
  int16_t y_q[10];
  y_q[0] = static_cast<int16_t*>(output->data.data)[0];
  y_q[1] = static_cast<int16_t*>(output->data.data)[1];
  y_q[2] = static_cast<int16_t*>(output->data.data)[2];
  y_q[3] = static_cast<int16_t*>(output->data.data)[3];
  y_q[4] = static_cast<int16_t*>(output->data.data)[4];
  y_q[5] = static_cast<int16_t*>(output->data.data)[5];
  y_q[6] = static_cast<int16_t*>(output->data.data)[6];
  y_q[7] = static_cast<int16_t*>(output->data.data)[7];
  y_q[8] = static_cast<int16_t*>(output->data.data)[8];
  y_q[9] = static_cast<int16_t*>(output->data.data)[9];
  // Dequantize the output from integer to floating-point
  double y[10];
  y[0] = (y_q[0] - output->params.zero_point) * output->params.scale;
  y[1] = (y_q[1] - output->params.zero_point) * output->params.scale;
  y[2] = (y_q[2] - output->params.zero_point) * output->params.scale;
  y[3] = (y_q[3] - output->params.zero_point) * output->params.scale;
  y[4] = (y_q[4] - output->params.zero_point) * output->params.scale;
  y[5] = (y_q[5] - output->params.zero_point) * output->params.scale;
  y[6] = (y_q[6] - output->params.zero_point) * output->params.scale;
  y[7] = (y_q[7] - output->params.zero_point) * output->params.scale;
  y[8] = (y_q[8] - output->params.zero_point) * output->params.scale;
  y[9] = (y_q[9] - output->params.zero_point) * output->params.scale;

  // MicroPrintf("\n\n prob. 0 = %f%%\n", (y[0]*100));
  // MicroPrintf(" prob. 1 = %f%%\n",     (y[1]*100));
  // MicroPrintf(" prob. 2 = %f%%\n",     (y[2]*100));
  // MicroPrintf(" prob. 3 = %f%%\n",     (y[3]*100));
  // MicroPrintf(" prob. 4 = %f%%\n",     (y[4]*100));
  // MicroPrintf(" prob. 5 = %f%%\n",     (y[5]*100));
  // MicroPrintf(" prob. 6 = %f%%\n",     (y[6]*100));
  // MicroPrintf(" prob. 7 = %f%%\n",     (y[7]*100));
  // MicroPrintf(" prob. 8 = %f%%\n",     (y[8]*100));
  // MicroPrintf(" prob. 9 = %f%%\n",     (y[9]*100));

  if (inference_count == local_test_samples-1)
    printf("\n n_cycles (last) = %llu\n", cycle_count);

  if (inference_count>=skip_first_inferences) {
    uint64_t cycle_added = cycle_count;
    average_cycles += cycle_added;
  }

  int pred = 0;
  for (int c=1; c<10; ++c) {
    if (y[c] > y[pred])
      pred = c;
  }

  if (pred == data_vect[inference_count].lable) {
    ++correct_count;
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

  printf("\n\naccuracy = %d%%\n", (int)(((double)correct_count/(double)inference_count)*1000));
  printf("\naverage_cycles final = %llu\n", (uint64_t)((double)(average_cycles) / (double)(local_test_samples-skip_first_inferences)));

  printf("\n-------------------------------------------------------------\n");

  return 0;
}
