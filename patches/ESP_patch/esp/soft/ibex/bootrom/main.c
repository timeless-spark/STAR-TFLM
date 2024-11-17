#include "uart.h"
#include "platform.h"

int main()
{
	init_uart();

	// // jump to the address
	// __asm__ volatile(
	// 	"li s0, 0x81000000;"
	// 	"la a1, _dtb;"
	// 	"jr s0");

	// jump to the address
	__asm__ volatile(
		"li s0, 0x80000000;"
		"la a1, _dtb;"
		"jr s0");

	while (1)
	{
		// do nothing
	}
}

void handle_trap(void)
{
  print_uart("handle_trap\n\0");
}
