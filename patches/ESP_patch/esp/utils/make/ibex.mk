# Copyright (c) 2011-2022 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0

# overwrite these variables to select which test to perform
TFLM_TYPE := standard
#         := standard_unroll
#         := star
#         := star_emulate

TINY_NET :=
#        := imclass
#        := keyword
#        := mobilenet
#        := anomaly

QUANT_TYPE := int8
#          := int4
#          := int16
#          := MPQ

MEASURE_TYPE :=
#            := global
#            := kernel
#            := only_fc
#            := only_cnn
#            := only_dwcnn

IBEX ?= $(ESP_ROOT)/rtl/cores/ibex/ibex

# SOFT is "esp/soft/ibex"

RISCV_TESTS = $(SOFT)/riscv-tests
RISCV_PK = $(SOFT)/riscv-pk

soft: $(SOFT_BUILD)/prom.srec $(SOFT_BUILD)/ram.srec $(SOFT_BUILD)/prom.bin $(SOFT_BUILD)/systest.bin 

soft-clean:
	$(QUIET_CLEAN)$(RM) 			\
		$(SOFT_BUILD)/prom.srec 	\
		$(SOFT_BUILD)/ram.srec		\
		$(SOFT_BUILD)/prom.exe		\
		$(SOFT_BUILD)/systest.exe	\
		$(SOFT_BUILD)/prom.bin		\
		$(SOFT_BUILD)/riscv.dtb		\
		$(SOFT_BUILD)/startup.o		\
		$(SOFT_BUILD)/main.o		\
		$(SOFT_BUILD)/uart.o		\
		$(SOFT)/common/syscalls.o   \
		$(SOFT_BUILD)/systest.bin

soft-tflm-clean:
	$(MAKE) -j16 -C ./TFLM_restore/tflite-micro -f ./tensorflow/lite/micro/tools/make/Makefile clean
	$(QUIET_CLEAN)$(RM) $(SOFT_BUILD)/tensorflow-microlite.a

.PHONY: soft-tflm-clean	

soft-distclean: soft-clean soft-tflm-clean

$(SOFT_BUILD)/riscv.dtb: $(ESP_CFG_BUILD)/riscv.dts $(ESP_CFG_BUILD)/socmap.vhd
	$(QUIET_BUILD) mkdir -p $(SOFT_BUILD)
	$(QUIET_BUILD) dtc -I dts $< -O dtb -o $@

$(SOFT_BUILD)/startup.o: $(BOOTROM_PATH)/startup.S $(SOFT_BUILD)/riscv.dtb
	$(QUIET_BUILD) mkdir -p $(SOFT_BUILD)
	$(QUIET_CC)cd $(SOFT_BUILD);  $(CROSS_COMPILE_ELF)gcc \
		-Os \
		-Wall -Werror \
		-mcmodel=medany -mexplicit-relocs \
		-march=rv32imc -mabi=ilp32 \
		-mstrict-align \
		-I$(DESIGN_PATH)/$(ESP_CFG_BUILD) \
		-I$(BOOTROM_PATH) \
		-c $< -o $@

$(SOFT_BUILD)/main.o: $(BOOTROM_PATH)/main.c $(ESP_CFG_BUILD)/esplink.h
	$(QUIET_BUILD) mkdir -p $(SOFT_BUILD)
	$(QUIET_CC) $(CROSS_COMPILE_ELF)gcc \
		-Os \
		-Wall -Werror \
		-mcmodel=medany -mexplicit-relocs \
		-march=rv32imc -mabi=ilp32 \
		-mstrict-align \
		-I$(BOOTROM_PATH) \
		-I$(DESIGN_PATH)/$(ESP_CFG_BUILD) \
		-c $< -o $@

$(SOFT_BUILD)/uart.o: $(BOOTROM_PATH)/uart.c $(ESP_CFG_BUILD)/esplink.h
	$(QUIET_BUILD) mkdir -p $(SOFT_BUILD)
	$(QUIET_CC) $(CROSS_COMPILE_ELF)gcc \
		-Os \
		-Wall -Werror \
		-mcmodel=medany -mexplicit-relocs \
		-march=rv32imc -mabi=ilp32 \
		-mstrict-align \
		-I$(BOOTROM_PATH) \
		-I$(DESIGN_PATH)/$(ESP_CFG_BUILD) \
		-c $< -o $@

$(SOFT_BUILD)/prom.exe: $(SOFT_BUILD)/startup.o $(SOFT_BUILD)/uart.o $(SOFT_BUILD)/main.o $(BOOTROM_PATH)/linker.lds
	$(QUIET_CC) $(CROSS_COMPILE_ELF)gcc \
		-Os \
		-Wall -Werror \
		-mcmodel=medany -mexplicit-relocs \
		-march=rv32imc -mabi=ilp32 \
		-mstrict-align \
		-I$(BOOTROM_PATH) \
		-I$(DESIGN_PATH)/$(ESP_CFG_BUILD) \
		-nodefaultlibs -nostartfiles \
		-T$(BOOTROM_PATH)/linker.lds \
		$(SOFT_BUILD)/startup.o $(SOFT_BUILD)/uart.o $(SOFT_BUILD)/main.o -lgcc\
		-o $@

#####################################################################################
$(SOFT)/common/syscalls.o: $(SOFT)/common/syscalls.c
	$(QUIET_CC) $(CROSS_COMPILE_ELF)gcc \
		-Os \
		-mcmodel=medany -mexplicit-relocs \
		-march=rv32imc -mabi=ilp32 \
		-mstrict-align \
	-I$(DESIGN_PATH)/$(ESP_CFG_BUILD) \
	-I$(RISCV_TESTS)/env -I$(BOOTROM_PATH) \
	-I$(RISCV_TESTS)/benchmarks/common \
	-c $< -o $@
#####################################################################################

$(SOFT_BUILD)/prom.srec: $(SOFT_BUILD)/prom.exe
	$(QUIET_OBJCP)$(CROSS_COMPILE_ELF)objcopy -O srec $< $@

$(SOFT_BUILD)/prom.bin: $(SOFT_BUILD)/prom.exe
	$(QUIET_OBJCP) $(CROSS_COMPILE_ELF)objcopy -O binary $< $@


# arch definition
RISCV_CFLAGS +=	-march=rv32im -mabi=ilp32
RISCV_CFLAGS += -std=c++11

# global optimization flag
RISCV_CFLAGS += -O2
# RISCV_CFLAGS += -O0

# gcc output
RISCV_CFLAGS = -mcmodel=medany
#RISCV_CFLAGS = -mcmodel=medlow

# use relocations and or relax
RISCV_CFLAGS += -mexplicit-relocs
#RISCV_CFLAGS += -mno-explicit-relocs
RISCV_CFLAGS += -mrelax
#RISCV_CFLAGS += -mno-relax

# startup and printf
RISCV_CFLAGS += -fno-builtin-printf
RISCV_CFLAGS += -nostartfiles

#alignment
RISCV_CFLAGS +=	-mstrict-align
RISCV_CFLAGS += -falign-functions=4
RISCV_CFLAGS += -falign-loops=4
RISCV_CFLAGS += -falign-labels=4
RISCV_CFLAGS += -falign-jumps=4

# code opt funcs and data
#RISCV_CFLAGS += -ffunction-sections ## it could lead to problems
#RISCV_CFLAGS += -fdata-sections ## it could lead to problems

# remove thread variables initialization
RISCV_CFLAGS += -fno-threadsafe-statics

# remove run-time type identification (i.e. dynamic_cast)
RISCV_CFLAGS += -fno-rtti

# remove exception hanling in the code
RISCV_CFLAGS += -fno-exceptions
RISCV_CFLAGS += -fno-unwind-tables

# a char is a byte
RISCV_CFLAGS += -funsigned-char

# checks for null pointer
RISCV_CFLAGS += -fno-delete-null-pointer-checks

# remove frame pointer
RISCV_CFLAGS += -fomit-frame-pointer

# strict C for destructors
RISCV_CFLAGS += -fno-use-cxa-atexit

#Warnings
RISCV_CFLAGS += -fpermissive
RISCV_CFLAGS += -Werror
RISCV_CFLAGS += -Wall
RISCV_CFLAGS += -Wextra
RISCV_CFLAGS += -Wnon-virtual-dtor
RISCV_CFLAGS += -Wsign-compare
RISCV_CFLAGS += -Wdouble-promotion
RISCV_CFLAGS += -Wshadow
RISCV_CFLAGS += -Wunused-function
RISCV_CFLAGS += -Wswitch
RISCV_CFLAGS += -Wvla
RISCV_CFLAGS += -Wmissing-field-initializers
RISCV_CFLAGS += -Wstrict-aliasing
RISCV_CFLAGS += -Wno-unused-parameter
#RISCV_CFLAGS += -Wunused-variable
RISCV_CFLAGS += -Wno-unused-variable
RISCV_CFLAGS += -Wno-unused-but-set-variable

# linker-related flags
#LDFLAGS += -nostdlib
LDFLAGS += -lm -lgcc -lc_nano
LDFLAGS += -L$(SOFT_BUILD) -ltensorflow-microlite
LDFLAGS += -Wl,--fatal-warnings
LDFLAGS += -Xlinker -Map=output.map

# libraries position
RISCV_CFLAGS += -I$(RISCV_TESTS)/env
RISCV_CFLAGS += -I$(RISCV_TESTS)/benchmarks/common
RISCV_CFLAGS += -I$(SOFT)/common
RISCV_CFLAGS += -I$(BOOTROM_PATH)
RISCV_CFLAGS += -I$(DESIGN_PATH)/$(ESP_CFG_BUILD)
RISCV_CFLAGS += -I./TFLM_restore/tflite-micro
RISCV_CFLAGS += -I./TFLM_restore/tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
RISCV_CFLAGS += -I./TFLM_restore/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp
RISCV_CFLAGS += -I./TFLM_restore/tflite-micro/tensorflow/lite/micro/tools/make/downloads/ruy
RISCV_CFLAGS += -I./TFLM_restore/tflite-micro/tensorflow/lite/kernels/internal/reference/integer_ops/star

# STAR flags
ifeq ("$(TFLM_TYPE)", "star")
	ifeq ("$(QUANT_TYPE)", "int8")
		TEST_REPO = ./test_data/star/int8
		FLATBUF_REPO = ./model_data/star/int8
		RISCV_CFLAGS += -I./model_data/star/int8 -I./test_data/star/int8
	else ifeq ("$(QUANT_TYPE)", "int4")
		TEST_REPO = ./test_data/star/int4
		FLATBUF_REPO = ./model_data/star/int4
		RISCV_CFLAGS += -I./model_data/star/int4 -I./test_data/star/int4
	else ifeq ("$(QUANT_TYPE)", "int16")
		TEST_REPO = ./test_data/star/int16
		FLATBUF_REPO = ./model_data/star/int16
		RISCV_CFLAGS += -I./model_data/star/int16 -I./test_data/star/int16
	else ifeq ("$(QUANT_TYPE)", "MPQ")
		TEST_REPO = ./test_data/star/MPQ
		FLATBUF_REPO = ./model_data/star/MPQ
		RISCV_CFLAGS += -I./model_data/star/MPQ -I./test_data/star/MPQ
	else ifeq ("$(QUANT_TYPE)", "MPQ_UNPACK")
		TEST_REPO = ./test_data/star/MPQ_UNPACK
		FLATBUF_REPO = ./model_data/star/MPQ_UNPACK
		RISCV_CFLAGS += -I./model_data/star/MPQ_UNPACK -I./test_data/star/MPQ_UNPACK
	endif
	TFLM_MAKE_FLAGS = OPTIMIZED_KERNEL_DIR=star
	RISCV_CFLAGS += -DSTAR
else ifeq ("$(TFLM_TYPE)", "star_emulate")
	ifeq ("$(QUANT_TYPE)", "int8")
		TEST_REPO = ./test_data/star/int8
		FLATBUF_REPO = ./model_data/star/int8
		RISCV_CFLAGS += -I./model_data/star/int8 -I./test_data/star/int8
	else ifeq ("$(QUANT_TYPE)", "int4")
		TEST_REPO = ./test_data/star/int4
		FLATBUF_REPO = ./model_data/star/int4
		RISCV_CFLAGS += -I./model_data/star/int4 -I./test_data/star/int4
	else ifeq ("$(QUANT_TYPE)", "int16")
		TEST_REPO = ./test_data/star/int16
		FLATBUF_REPO = ./model_data/star/int16
		RISCV_CFLAGS += -I./model_data/star/int16 -I./test_data/star/int16
	else ifeq ("$(QUANT_TYPE)", "MPQ")
		TEST_REPO = ./test_data/star/MPQ
		FLATBUF_REPO = ./model_data/star/MPQ
		RISCV_CFLAGS += -I./model_data/star/MPQ -I./test_data/star/MPQ
	else ifeq ("$(QUANT_TYPE)", "MPQ_UNPACK")
		TEST_REPO = ./test_data/star/MPQ_UNPACK
		FLATBUF_REPO = ./model_data/star/MPQ_UNPACK
		RISCV_CFLAGS += -I./model_data/star/MPQ_UNPACK -I./test_data/star/MPQ_UNPACK
	endif
	TFLM_MAKE_FLAGS = OPTIMIZED_KERNEL_DIR=star STAR_EMULATE=true
	RISCV_CFLAGS += -DSTAR -DSTAR_EMULATE
else ifeq ("$(TFLM_TYPE)", "standard")
	TEST_REPO = ./test_data/standard/int8
	FLATBUF_REPO = ./model_data/standard/int8
	TFLM_MAKE_FLAGS = FC_SCALE_PER_CHANNEL=true
	RISCV_CFLAGS += -I./model_data/standard/int8 -I./test_data/standard/int8 -DFC_SCALE_PER_CHANNEL
else ifeq ("$(TFLM_TYPE)", "standard_unroll")
	TEST_REPO = ./test_data/standard/int8
	FLATBUF_REPO = ./model_data/standard/int8
	TFLM_MAKE_FLAGS = FC_SCALE_PER_CHANNEL=true MANUAL_UNROLL=true
	RISCV_CFLAGS += -I./model_data/standard/int8 -I./test_data/standard/int8 -DFC_SCALE_PER_CHANNEL -DMANUAL_UNROLL
endif

ifeq ("$(MEASURE_TYPE)", "global")
	RISCV_CFLAGS += -DMEASURE_GLOBAL
else ifeq ("$(MEASURE_TYPE)", "kernel")
	TFLM_MAKE_FLAGS += MEASURE_FC=true MEASURE_CNN=true MEASURE_DWCNN=true
	RISCV_CFLAGS += -DMEASURE_FC -DMEASURE_CNN -DMEASURE_DWCNN
else ifeq ("$(MEASURE_TYPE)", "only_fc")
	TFLM_MAKE_FLAGS += MEASURE_FC=true
	RISCV_CFLAGS += -DMEASURE_FC
else ifeq ("$(MEASURE_TYPE)", "only_cnn")
	TFLM_MAKE_FLAGS += MEASURE_CNN=true
	RISCV_CFLAGS += -DMEASURE_CNN
else ifeq ("$(MEASURE_TYPE)", "only_dwcnn")
	TFLM_MAKE_FLAGS += MEASURE_DWCNN=true
	RISCV_CFLAGS += -DMEASURE_DWCNN
else ifeq ("$(MEASURE_TYPE)", "unpacking")
	TFLM_MAKE_FLAGS += MEASURE_UNPACK=true
	RISCV_CFLAGS += -DMEASURE_UNPACK
endif

# TFLM flags
RISCV_CFLAGS += -DTF_LITE_STATIC_MEMORY
RISCV_CFLAGS += -DTF_LITE_USE_GLOBAL_MIN
RISCV_CFLAGS += -DTF_LITE_USE_GLOBAL_MAX
RISCV_CFLAGS += -DTF_LITE_MCU_DEBUG_LOG
RISCV_CFLAGS += -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS
RISCV_CFLAGS += -DTFLITE_SINGLE_ROUNDING
RISCV_CFLAGS += -DTFLITE_EMULATE_FLOAT

# compile TFLM microlib
# ##################################################################################### for debug change flag BUILD_TYPE=debug in the following makefile
$(SOFT_BUILD)/tensorflow-microlite.a:
	$(MAKE) -j16 -C ./TFLM_restore/tflite-micro -f ./tensorflow/lite/micro/tools/make/Makefile TARGET=ibex_rv32im TARGET_ARCH=ibex_rv32im BUILD_TYPE=release $(TFLM_MAKE_FLAGS) microlite
	cp ./TFLM_restore/tflite-micro/gen/ibex_rv32im_ibex_rv32im_release/lib/libtensorflow-microlite.a $(SOFT_BUILD)/libtensorflow-microlite.a

# compile systest
# #####################################################################################
$(SOFT_BUILD)/systest.exe: systest.c $(SOFT_BUILD)/uart.o $(SOFT)/common/syscalls.o
	$(CROSS_COMPILE_ELF)g++ $(RISCV_CFLAGS) \
	$(RISCV_TESTS)/benchmarks/common/crt.S \
	$(SOFT)/common/syscalls.o $(SOFT_BUILD)/uart.o model_data.c $< \
	-T $(RISCV_TESTS)/benchmarks/common/test.ld \
	$(LDFLAGS) -o $@

$(SOFT_BUILD)/systest_imclass.exe: imclass_systest.c $(SOFT_BUILD)/uart.o $(SOFT)/common/syscalls.o $(SOFT_BUILD)/tensorflow-microlite.a
	$(CROSS_COMPILE_ELF)g++ $(RISCV_CFLAGS) \
	$(RISCV_TESTS)/benchmarks/common/crt.S \
	$(SOFT)/common/syscalls.o $(SOFT_BUILD)/uart.o $(FLATBUF_REPO)/imclass_model_data.c $(TEST_REPO)/imclass_test_data.c $< \
	-T $(RISCV_TESTS)/benchmarks/common/test.ld \
	$(LDFLAGS) -o $@

$(SOFT_BUILD)/systest_mobilenet.exe: mobilenet_systest.c $(SOFT_BUILD)/uart.o $(SOFT)/common/syscalls.o $(SOFT_BUILD)/tensorflow-microlite.a
	$(CROSS_COMPILE_ELF)g++ $(RISCV_CFLAGS) \
	$(RISCV_TESTS)/benchmarks/common/crt.S \
	$(SOFT)/common/syscalls.o $(SOFT_BUILD)/uart.o $(FLATBUF_REPO)/mobilenet_model_data.c $(TEST_REPO)/mobilenet_test_data.c $< \
	-T $(RISCV_TESTS)/benchmarks/common/test.ld \
	$(LDFLAGS) -o $@

$(SOFT_BUILD)/systest_keyword.exe: keyword_systest.c $(SOFT_BUILD)/uart.o $(SOFT)/common/syscalls.o $(SOFT_BUILD)/tensorflow-microlite.a
	$(CROSS_COMPILE_ELF)g++ $(RISCV_CFLAGS) \
	$(RISCV_TESTS)/benchmarks/common/crt.S \
	$(SOFT)/common/syscalls.o $(SOFT_BUILD)/uart.o $(FLATBUF_REPO)/keyword_model_data.c $(TEST_REPO)/keyword_test_data.c $< \
	-T $(RISCV_TESTS)/benchmarks/common/test.ld \
	$(LDFLAGS) -o $@

$(SOFT_BUILD)/systest_anomaly.exe: anomaly_systest.c $(SOFT_BUILD)/uart.o $(SOFT)/common/syscalls.o $(SOFT_BUILD)/tensorflow-microlite.a
	$(CROSS_COMPILE_ELF)g++ $(RISCV_CFLAGS) \
	$(RISCV_TESTS)/benchmarks/common/crt.S \
	$(SOFT)/common/syscalls.o $(SOFT_BUILD)/uart.o $(FLATBUF_REPO)/anomaly_model_data.c $(TEST_REPO)/anomaly_test_data.c $< \
	-T $(RISCV_TESTS)/benchmarks/common/test.ld \
	$(LDFLAGS) -o $@

# #####################################################################################

ifeq ($(TINY_NET),)
SYSTEST_EXT =
else
SYSTEST_EXT = _$(TINY_NET)
endif

# generate binaries
$(SOFT_BUILD)/systest.bin: $(SOFT_BUILD)/systest$(SYSTEST_EXT).exe
	$(QUIET_OBJCP) $(CROSS_COMPILE_ELF)objcopy -O binary $< $@

$(SOFT_BUILD)/ram.srec: $(SOFT_BUILD)/systest$(SYSTEST_EXT).exe
	$(QUIET_OBJCP) $(CROSS_COMPILE_ELF)objcopy -O srec --gap-fill 0 $< $@


sysroot:

sysroot.files:

sysroot.cpio:

linux-build/.config:

linux-build/vmlinux:

pk-build:

pk-build/bbl:

linux.bin:

linux:

linux-clean:

linux-distclean:


### Flags

## Modelsim
VLOGOPT +=
VLOGOPT += -incr
VLOGOPT += -64
VLOGOPT += -nologo
VLOGOPT += -suppress 13262
VLOGOPT += -suppress 2286
VLOGOPT += -permissive
VLOGOPT += +define+WT_DCACHE
ifneq ($(filter $(TECHLIB),$(FPGALIBS)),)
# use Xilinx-based primitives for FPGA
VLOGOPT += +define+PRIM_DEFAULT_IMPL=prim_pkg::ImplXilinx
endif
VLOGOPT += -pedanticerrors
VLOGOPT += -suppress 2583
VLOGOPT += -suppress 13314

VLOGOPT += +define+STAR

## Xcelium
XMLOGOPT +=
XMLOGOPT += -UNCLOCKEDSVA

#Vivado flags
VIVOPT +=
VIVOPT += STAR=STAR

### Incdir and RTL
ifeq ("$(CPU_ARCH)", "ibex")
INCDIR  += $(IBEX)/vendor/lowrisc_ip/ip/prim/rtl
VERILOG_IBEX += $(foreach f, $(shell strings $(FLISTS)/ibex_vlog.flist), $(IBEX)/$(f))
VERILOG_IBEX += $(DESIGN_PATH)/$(ESP_CFG_BUILD)/plic_regmap.sv
THIRDPARTY_VLOG += $(VERILOG_IBEX)
endif
