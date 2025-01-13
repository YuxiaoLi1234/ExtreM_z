#!/bin/bash

# 设置安装路径为用户主目录中的 ExtreM_install

INSTALL_PREFIX=${HOME}

# 创建构建目录
cd build
# 运行 CMake，生成构建文件
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} ..

# 编译并安装库
make -j$(nproc) && make install

# 检查库是否成功编译并安装
if [ $? -eq 0 ]; then
    echo "Build and installation completed successfully."
    echo "Library installed at: ${INSTALL_PREFIX}/lib"
    echo "Header files installed at: ${INSTALL_PREFIX}/include/api"
else
    echo "Build or installation failed."
    exit 1
fi

# 编译 main.cpp，生成可执行文件
echo "Compiling main.cpp..."
cd ..

g++ -std=c++17 ./main.cpp -o main_program -I${INSTALL_PREFIX}/include/include/api -L${INSTALL_PREFIX}/lib64 -lExtreMz -lsz -lzstd -fopenmp

if [ $? -eq 0 ]; then
    echo "main_program compiled successfully."
    echo "Run it using: ./main_program"
else
    echo "Failed to compile main_program."
    exit 1
fi
