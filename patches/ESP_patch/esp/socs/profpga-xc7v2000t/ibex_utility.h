#ifndef IBEX_UTILITY_H_
#define IBEX_UTILITY_H_

static inline uint32_t __attribute__((always_inline)) read_counter_status()
{
    uint32_t reg_val;
 
    asm volatile ("csrr t0, 0x320\n":::"memory");
    asm volatile ("mv %0, t0\n":"=r"(reg_val)::"memory");

    return reg_val;
}

static inline void __attribute__((always_inline)) disable_counter()
{
    asm volatile ("csrr t0, 0x320\n":::"memory");
    asm volatile ("ori t0, t0, 0x01\n":::"memory");
    asm volatile ("csrw 0x320, t0\n":::"memory");

    return;
}

static inline void __attribute__((always_inline)) enable_counter()
{
    asm volatile ("csrr t0, 0x320\n":::"memory");
    asm volatile ("andi t0, t0, 0xfffffffe\n":::"memory");
    asm volatile ("csrw 0x320, t0\n":::"memory");

    return;
}

static inline void __attribute__((always_inline)) reset_counter()
{
    asm volatile ("csrw 0xb00, x0\n":::"memory");
    asm volatile ("csrw 0xb80, x0\n":::"memory");

    return;
}

static inline uint32_t __attribute__((always_inline)) read_counter_low()
{
    uint32_t counter_l;
 
    asm volatile ("li t0, 0\n":::"memory");
    asm volatile ("csrr t0, 0xb00\n":::"memory");
    asm volatile ("mv %0, t0\n":"=r"(counter_l)::"memory");

    return counter_l;
}

static inline uint32_t __attribute__((always_inline)) read_counter_high()
{
    uint32_t counter_h;
 
    asm volatile ("li t0, 0\n":::"memory");
    asm volatile ("csrr t0, 0xb80\n":::"memory");
    asm volatile ("mv %0, t0\n":"=r"(counter_h)::"memory");

    return counter_h;
}

static inline uint64_t __attribute__((always_inline)) __attribute__((optimize("O0"))) read_counter()
{
    uint64_t counter;

    uint32_t count_h_b = read_counter_high();
    uint32_t count_l = read_counter_low();
    uint32_t count_h_a = read_counter_high();
    
    if (count_h_b != count_h_a)
        count_l = read_counter_low();

    counter = (((uint64_t)count_h_a) << 32) | count_l;

    return counter;
}

#endif