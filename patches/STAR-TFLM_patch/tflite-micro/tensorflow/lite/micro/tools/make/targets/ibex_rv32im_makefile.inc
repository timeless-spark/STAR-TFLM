# Settings for RISCV 32-bit MCU toolchain.
TARGET_ARCH := riscv32_mcu
TARGET_TOOLCHAIN_PREFIX := /home/edward.manca/riscv32imc_star/bin/riscv32-unknown-elf-

FC_SCALE_PER_CHANNEL :=

MEASURE_FC :=
MEASURE_CNN :=
MEASURE_DWCNN :=
MEASURE_UNPACK :=

MANUAL_UNROLL :=

export PATH := $(MAKEFILE_DIR)/downloads/riscv_toolchain/bin/:$(PATH)


PLATFORM_FLAGS = \
  -march=rv32im \
  -mabi=ilp32 \
  -mcmodel=medany \
  -fno-builtin-printf \
  -DTF_LITE_MCU_DEBUG_LOG \
  -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
  -funsigned-char \
  -fno-delete-null-pointer-checks \
  -fomit-frame-pointer

CXXFLAGS += $(PLATFORM_FLAGS) \
  -fpermissive \
  -fno-use-cxa-atexit \
  -DTF_LITE_USE_GLOBAL_MIN \
  -DTF_LITE_USE_GLOBAL_MAX 

CXXFLAGS += -mexplicit-relocs
#CXXFLAGS += -mno-explicit-relocs
CXXFLAGS += -static
CXXFLAGS += -mrelax
#CXXFLAGS += -mno-relax
CXXFLAGS += -mstrict-align
CXXFLAGS += -falign-functions=4
CXXFLAGS += -falign-loops=4
CXXFLAGS += -falign-labels=4
CXXFLAGS += -falign-jumps=4

CXXFLAGS += -DTFLITE_SINGLE_ROUNDING
CXXFLAGS += -DTFLITE_EMULATE_FLOAT

#------------------------------------------

CCFLAGS += $(PLATFORM_FLAGS)

CCFLAGS += -mexplicit-relocs
#CCFLAGS += -mno-explicit-relocs
CCFLAGS += -static
CCFLAGS += -mrelax
#CCFLAGS += -mno-relax
CCFLAGS += -mstrict-align
CCFLAGS += -falign-functions=4
CCFLAGS += -falign-loops=4
CCFLAGS += -falign-labels=4
CCFLAGS += -falign-jumps=4

CCFLAGS += -DTFLITE_SINGLE_ROUNDING
CCFLAGS += -DTFLITE_EMULATE_FLOAT

#CCFLAGS += -ffast-math

BUILD_TYPE := micro

# MICROLITE_CC_SRCS += \
#   $(wildcard tensorflow/lite/micro/riscv32_mcu/*.cc)

ifneq ($(FC_SCALE_PER_CHANNEL),)
  CXXFLAGS += -DFC_SCALE_PER_CHANNEL
  CCFLAGS += -DFC_SCALE_PER_CHANNEL
endif

ifneq ($(MEASURE_FC),)
  CXXFLAGS += -DMEASURE_FC
  CCFLAGS += -DMEASURE_FC
endif

ifneq ($(MEASURE_CNN),)
  CXXFLAGS += -DMEASURE_CNN
  CCFLAGS += -DMEASURE_CNN
endif

ifneq ($(MEASURE_DWCNN),)
  CXXFLAGS += -DMEASURE_DWCNN
  CCFLAGS += -DMEASURE_DWCNN
endif

ifneq ($(MEASURE_UNPACK),)
  CXXFLAGS += -DMEASURE_UNPACK
  CCFLAGS += -DMEASURE_UNPACK
endif

ifneq ($(MANUAL_UNROLL),)
  CXXFLAGS += -DMANUAL_UNROLL
  CCFLAGS += -DMANUAL_UNROLL
endif

