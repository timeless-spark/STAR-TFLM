# start sequence of the bootloader
#
#
#include <smp.h>
#include "platform.h"
#include "esplink.h"

  .section .text.init
  .option norvc
  .globl _prog_start
_prog_start:
  smp_pause(s1, s2)

  li sp, SP_START

  call main

  smp_resume(s1, s2)
#  csrr a0, mhartid
#  la a1, _dtb
#  li s1, DRAM_BASE
#  jr s1

  .section .dtb
  .globl _dtb
  .align 4, 0
_dtb:
  .incbin "riscv.dtb"
