using HDF5
using Luna

# 检查HDF5文件的脚本
# 用于验证模拟数据的完整性和格式正确性

# 要检查的文件列表
files_to_check = [
    "sample_000001.h5",
    "sample_000002.h5",
    "sample_000003.h5",
    "sample_000004.h5",
    "sample_000005.h5"
]

# 输出目录
output_dir = "training_data"

# 检查文件函数
function check_file(filepath)
    println("\n=== 检查文件: $filepath ===")
    
    if !isfile(filepath)
        println("❌ 文件不存在")
        return false
    end
    
    try
        h5open(filepath, "r") do file
            # 检查基本结构
            println("✅ 文件打开成功")
            
            # 检查根级键
            root_keys = keys(file)
            println("根级键: $root_keys")
            
            # 检查必要的键
            required_keys = ["Eω", "z", "stats", "meta"]
            for key in required_keys
                if haskey(file, key)
                    println("✅ 包含键: $key")
                else
                    println("❌ 缺少键: $key")
                end
            end
            
            # 检查物理特征
            if haskey(file, "physics_features")
                println("✅ 包含物理特征")
                physics_keys = keys(file["physics_features"])
                println("物理特征键: $physics_keys")
                
                # 检查物理特征值
                required_physics = ["beta2", "gamma", "N", "L0", "gamma_K", "P_ratio", "Aeff", "neff"]
                for key in required_physics
                    if haskey(file["physics_features"], key)
                        value = read(file["physics_features"][key])
                        println("  ✅ $key: $value")
                    else
                        println("  ❌ 缺少物理特征: $key")
                    end
                end
            else
                println("❌ 缺少物理特征组")
            end
            
            # 检查Eω数据
            if haskey(file, "Eω")
                Eω = file["Eω"]
                println("✅ Eω数据形状: $(size(Eω))")
            end
            
            # 检查z数据
            if haskey(file, "z")
                z = file["z"]
                println("✅ z数据形状: $(size(z))")
                println("  z范围: $(minimum(z)) to $(maximum(z))")
            end
            
            # 检查统计数据
            if haskey(file, "stats")
                stats_keys = keys(file["stats"])
                println("✅ 统计数据键: $stats_keys")
            end
            
            # 检查元数据
            if haskey(file, "meta")
                println("✅ 包含元数据")
                meta_keys = keys(file["meta"])
                println("元数据键: $meta_keys")
            end
        end
        
        println("✅ 文件检查完成")
        return true
    catch e
        println("❌ 文件检查失败: $e")
        return false
    end
end

# 主函数
function main()
    println("开始检查HDF5文件...")
    
    success_count = 0
    total_count = length(files_to_check)
    
    for filename in files_to_check
        filepath = joinpath(output_dir, filename)
        if check_file(filepath)
            success_count += 1
        end
    end
    
    println("\n=== 检查结果 ===")
    println("成功: $success_count/$total_count")
    
    if success_count == total_count
        println("✅ 所有文件检查通过")
    else
        println("❌ 部分文件检查失败")
    end
end

# 运行主函数
main()
