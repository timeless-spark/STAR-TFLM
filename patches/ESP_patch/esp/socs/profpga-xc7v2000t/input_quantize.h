// rounding functions
#include <cmath>

int32_t QuantizeEle(float data, float scale, int zero_point, int32_t data_max, int32_t data_min) {
  double div = data/scale;
  int round_val = (div > 0) - (div < 0);
  int32_t q_data = static_cast<int32_t>(trunc(div + round_val*0.5) + zero_point);
//  int32_t q_data = static_cast<int32_t>(round(div) + zero_point);
  if (q_data > data_max)
    q_data = data_max;
  if (q_data < data_min)
    q_data = data_min;

  return q_data;
}

void GenInput_16(int sample_idx, int size, int16_t* data_buf, float in_scale, int in_zero_point) {
  int i = 0;
  for (; i < size; ++i) {
    data_buf[i] = static_cast<int16_t>(QuantizeEle(data_vect[sample_idx].data[i], in_scale, in_zero_point, (1<<15)-1, -((1<<15)-1)));
  }
}

void GenInput_8(int sample_idx, int size, int8_t* data_buf, float in_scale, int in_zero_point) {
  int i = 0;
  for (; i < size; ++i) {
    data_buf[i] = static_cast<int8_t>(QuantizeEle(data_vect[sample_idx].data[i], in_scale, in_zero_point, (1<<7)-1, -((1<<7)-1)));
  }
}

void GenInput_4(int sample_idx, int size, int8_t* data_buf, float in_scale, int in_zero_point) {
  int i = 0;
  for (; i < size; ++i) {
    int8_t data = static_cast<int8_t>(QuantizeEle(data_vect[sample_idx].data[i], in_scale, in_zero_point, (1<<3)-1, -((1<<3)-1)));
    if (i & 0x01)
    { // è pari (secondo elemento da 4bit)
      data_buf[i>>1] |= static_cast<int8_t>(data) << 4;
    }
    else
    { // è dispari (primo elemento da 4bit)
      data_buf[i>>1] = 0x00 | (0x0F & static_cast<int8_t>(data));
    }
  }
}