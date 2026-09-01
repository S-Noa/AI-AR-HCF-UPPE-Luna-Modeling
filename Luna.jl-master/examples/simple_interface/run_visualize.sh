#!/bin/bash
# 运行 visualize_hdf5.jl 的辅助脚本
# 自动处理 Julia 环境激活和依赖安装

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Luna.jl HDF5 可视化工具 ==="
echo "当前目录: $(pwd)"
echo ""

# 检查 Julia 是否可用
if ! command -v julia &> /dev/null; then
    echo "错误: 未找到 Julia 命令"
    echo "请确保 Julia 已安装并添加到 PATH"
    exit 1
fi

# 检查 Project.toml 是否存在
if [ ! -f "Project.toml" ]; then
    echo "警告: 未找到 Project.toml，使用默认 Julia 环境"
    echo ""
    julia --project=. visualize_hdf5.jl
else
    echo "找到 Project.toml，激活项目环境..."
    echo ""

    # 启动 Julia 并运行脚本
    # 使用 --project=. 来激活当前目录的项目环境
    julia --project=. -e '
        using Pkg

        # 如果还没有安装依赖，则安装
        if !haskey(Pkg.dependencies(), Base.UUID("f67ccb88-3c26-57ae-aae1-a51bef35a3f9"))  # HDF5 UUID
            println("正在安装 HDF5 包...")
            Pkg.add("HDF5")
        end

        if !haskey(Pkg.dependencies(), Base.UUID("b8c40bc3-8e24-4f07-9af9-72e86c2c11fc"))  # CairoMakie UUID (如果需要)
            println("正在安装 CairoMakie 包...")
            Pkg.add("CairoMakie")
        end

        println("运行可视化脚本...")
        println("")

        # 运行主脚本
        include("visualize_hdf5.jl")
    '
fi

echo ""
echo "=== 脚本执行完成 ==="