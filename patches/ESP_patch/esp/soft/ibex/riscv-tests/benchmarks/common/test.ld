/*======================================================================*/
/* Proxy kernel linker script                                           */
/*======================================================================*/
/* This is the linker script used when building the proxy kernel. */

/*----------------------------------------------------------------------*/
/* Setup                                                                */
/*----------------------------------------------------------------------*/

/* The OUTPUT_ARCH command specifies the machine architecture where the
   argument is one of the names used in the BFD library. More
   specifically one of the entires in bfd/cpu-mips.c */

OUTPUT_ARCH( "riscv" )
ENTRY(_start)

/*----------------------------------------------------------------------*/
/* Sections                                                             */
/*----------------------------------------------------------------------*/

MEMORY
{
  RAM (xrwal) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
  /* text: test code section */
  .text.init :
  { 
    /* . = 0x80000000; */
    *(.text.init)
  } > RAM

  .tohost :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.tohost)
  } > RAM

  .syscalls :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.syscalls)
  } > RAM

  .riscv.attributes :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.riscv.attributes)
    . = ALIGN(0x4);
    *(.rela.text .rela.text.* .rela.gnu.linkonce.t.*)
    *(.rela.rodata .rela.rodata.* .rela.gnu.linkonce.r.*)
    *(.rela.sdata .rela.sdata.* .rela.gnu.linkonce.s.*)
    *(.rela.sbss .rela.sbss.* .rela.gnu.linkonce.sb.*)
    *(.rela.sdata2 .rela.sdata2.* .rela.gnu.linkonce.s2.*)
    *(.rela.sbss2 .rela.sbss2.* .rela.gnu.linkonce.sb2.*)
  } > RAM
  
  .text :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.text.unlikely .text.*_unlikely .text.unlikely.*)
    *(.text.exit .text.exit.*)
    *(.text.startup .text.startup.*)
    *(.text.hot .text.hot.*)
    *(.text .stub .text.* .gnu.linkonce.t.*)
  } > RAM

  .comp.kern :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.comp.kern .comp.kern.*)
  }

  .text.fini :
  { 
    *(.text.fini)
  } > RAM

  /* data segment */

  .rodata :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.rodata .rodata.* .gnu.linkonce.r.*)
  } > RAM
  .rodata1 : 
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.rodata1)
  } > RAM

  .sdata2 :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)
  } > RAM
  
  .sbss2 : 
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.sbss2 .sbss2.* .gnu.linkonce.sb2.*)
  } > RAM

  .eh_frame_hdr :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.eh_frame_hdr) *(.eh_frame_entry .eh_frame_entry.*)
  } > RAM
  .eh_frame :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    KEEP (*(.eh_frame)) *(.eh_frame.*)
  } > RAM
  .gnu_extab :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.gnu_extab)
  } > RAM
  .gcc_except_table :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.gcc_except_table .gcc_except_table.*)
  } > RAM
  .exception_ranges :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.exception_ranges*)
  } > RAM

  .init_array    :
  {
    __init_array_start = .;
    KEEP (*(.init_array.*))
    KEEP (*(.init_array))
    __init_array_end = .;
  } > RAM
  .fini_array    :
  {
    __fini_array_start = .;
    KEEP (*(.fini_array.*))
    KEEP (*(.fini_array))
    __fini_array_end = .;
  } > RAM

  .data  : 
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.data .data.* .gnu.linkonce.d.*)
  } > RAM
  .data1 : 
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.data1)
  } > RAM

  .sdata : {
    . = . + 0x10;
    . = ALIGN(0x4);
    __global_pointer$ = . + 0x800;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  } > RAM
  
  .edata : {
    . = . + 0x10;
    . = ALIGN(0x4);
    _edata = .; PROVIDE (edata = .);
  } > RAM

  /* bss segment */
  .sbss           :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.dynsbss)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
  } > RAM
  .bss : 
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.dynbss)
    *(.bss .bss.* .gnu.linkonce.b.*) 
  } > RAM

  /* thread-local data segment */
  .tdata :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    _tls_data = .;
    *(.tdata.begin)
    *(.tdata .tdata*)
    *(.tdata.end)
  } > RAM
  .tbss :
  {
    . = . + 0x10;
    . = ALIGN(0x4);
    *(.tbss)
    *(.tbss.end)
  } > RAM

  .tp.space :
  {
    . = . + 0x200;
    . = ALIGN(0x40);
    _tp_start = .;
    . = . + 0x100000;
  } > RAM
  .stack.space :
  {
    . = . + 0x100000;
    _stack_top = .;
  } > RAM

  /* heap starts here */
  .heap :
  {
    . = . + 0x200;
    . = ALIGN(0x40);
    end = .;
    _end = .;
    _heap_end = .;
  } > RAM

}

