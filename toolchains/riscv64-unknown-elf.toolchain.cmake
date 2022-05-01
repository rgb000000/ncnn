set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(DEFINED ENV{RISCV_ROOT_PATH})
    file(TO_CMAKE_PATH $ENV{RISCV_ROOT_PATH} RISCV_ROOT_PATH)
else()
    message(FATAL_ERROR "RISCV_ROOT_PATH env must be defined")
endif()

set(RISCV_ROOT_PATH ${RISCV_ROOT_PATH} CACHE STRING "root path to riscv toolchain")

set(CMAKE_C_COMPILER "${RISCV_ROOT_PATH}/bin/riscv64-unknown-elf-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_ROOT_PATH}/bin/riscv64-unknown-elf-g++")

set(CMAKE_FIND_ROOT_PATH "${RISCV_ROOT_PATH}/riscv64-unknown-elf")

add_compile_options(
        -march=rv64ima -mabi=lp64 -mcmodel=medany
        -nostartfiles
        -Wall -c
        -ffunction-sections
        -nostdlib
        -fno-use-cxa-atexit)
# SET(CMAKE_CXX_FLAGS "-march=rv64ima -mabi=lp64 -mcmodel=medany -nostartfiles -Wall -nostdlib" )
# SET(CMAKE_C_FLAGS   "-march=rv64ima -mabi=lp64 -mcmodel=medany -nostartfiles -Wall -nostdlib" )

# add_link_options(
#                  -L/opt/rv64ima/lib/gcc/riscv64-unknown-elf/11.1.0
#                  -L/opt/rv64ima/riscv64-unknown-elf/lib
#                  -Wl,-nostdlib
#                  -Wl,--start-group -l:libc.a -l:libm.a -l:libgcc.a -Wl,--end-group
#                  -Wl,--gc-sections --verbose
#         )

# SET(CMAKE_EXE_LINKER_FLAGS " -e _mystart                              \
#                              -nostdlib                                \
#                              -T /home/PRJ/mycpu/ncnn/src/internal.ld  \
#                              /home/PRJ/mycpu/newlib_test/mycrt0.o     \
#                              /home/PRJ/mycpu/newlib_test/stubs.o      \
#                              -L/opt/rv64ima/lib/gcc/riscv64-unknown-elf/11.1.0 \
#                              -L/opt/rv64ima/riscv64-unknown-elf/lib            \
#                              -Wl,-\( -l:libc.a -l:libm.a -l:libgcc.a -Wl,-\)   \
#                              -Wl,--gc-sections --verbose")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
