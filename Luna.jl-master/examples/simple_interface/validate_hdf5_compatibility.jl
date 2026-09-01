using HDF5
using Luna

function analyze_generation_structure()
    println("=== data_generation.jl 生成的数据结构 ===")
    structure = Dict()
    
    structure["主数据集"] = [
        Dict("名称" => "z", "描述" => "传播距离数组", "类型" => "Float64", "维度" => "一维数组"),
        Dict("名称" => "Eω", "描述" => "频域电场数据", "类型" => "ComplexF64", "维度" => "二维数组 (波长点数 × 传播距离点数)"),
        Dict("名称" => "Et", "描述" => "时域电场数据", "类型" => "ComplexF64", "维度" => "二维数组 (时间点数 × 传播距离点数)")
    ]
    
    structure["grid 组"] = [
        Dict("名称" => "λ", "描述" => "波长网格", "类型" => "Float64", "维度" => "一维数组"),
        Dict("名称" => "t", "描述" => "时间网格", "类型" => "Float64", "维度" => "一维数组"),
        Dict("名称" => "ω", "描述" => "角频率网格", "类型" => "Float64", "维度" => "一维数组")
    ]
    
    structure["physics_features 组"] = [
        Dict("名称" => "beta2", "描述" => "二阶色散系数", "类型" => "Float64", "维度" => "标量"),
        Dict("名称" => "gamma", "描述" => "非线性系数", "类型" => "ComplexF64", "维度" => "标量"),
        Dict("名称" => "N", "描述" => "孤子阶数", "类型" => "ComplexF64", "维度" => "标量"),
        Dict("名称" => "L0", "描述" => "色散长度", "类型" => "Float64", "维度" => "标量"),
        Dict("名称" => "gamma_K", "描述" => "电离参数", "类型" => "Float64", "维度" => "标量"),
        Dict("名称" => "P_ratio", "描述" => "峰值功率与临界功率比", "类型" => "ComplexF64", "维度" => "标量"),
        Dict("名称" => "Aeff", "描述" => "有效面积", "类型" => "Float64", "维度" => "标量"),
        Dict("名称" => "neff", "描述" => "有效折射率", "类型" => "Float64", "维度" => "标量")
    ]
    
    println("1. 主数据集:")
    for ds in structure["主数据集"]
        println("   - $(ds["名称"]): $(ds["描述"]) [$(ds["类型"]), $(ds["维度"])]")
    end
    
    println("\n2. grid 组:")
    for ds in structure["grid 组"]
        println("   - $(ds["名称"]): $(ds["描述"]) [$(ds["类型"]), $(ds["维度"])]")
    end
    
    println("\n3. physics_features 组:")
    for ds in structure["physics_features"]
        println("   - $(ds["名称"]): $(ds["描述"]) [$(ds["类型"]), $(ds["维度"])]")
    end
    
    return structure
end

function analyze_visualization_structure()
    println("\n=== visualize_hdf5.jl 读取的数据结构 ===")
    structure = Dict()
    
    structure["主数据集"] = [
        Dict("名称" => "z", "描述" => "传播距离", "类型" => "Any", "维度" => "一维"),
        Dict("名称" => "Eω", "描述" => "频域数据", "类型" => "Any", "维度" => "二维"),
        Dict("名称" => "Et", "描述" => "时域数据", "类型" => "Any", "维度" => "二维")
    ]
    
    structure["grid 组"] = [
        Dict("名称" => "λ", "描述" => "波长", "类型" => "Any", "维度" => "一维"),
        Dict("名称" => "t", "描述" => "时间", "类型" => "Any", "维度" => "一维")
    ]
    
    structure["physics_features 组"] = [
        Dict("名称" => "beta2", "描述" => "二阶色散", "类型" => "Any", "维度" => "标量"),
        Dict("名称" => "gamma", "描述" => "非线性系数", "类型" => "Any", "维度" => "标量"),
        Dict("名称" => "N", "描述" => "孤子阶数", "类型" => "Any", "维度" => "标量"),
        Dict("名称" => "L0", "描述" => "色散长度", "类型" => "Any", "维度" => "标量"),
        Dict("名称" => "gamma_K", "描述" => "电离参数", "类型" => "Any", "维度" => "标量"),
        Dict("名称" => "P_ratio", "描述" => "功率比", "类型" => "Any", "维度" => "标量"),
        Dict("名称" => "Aeff", "描述" => "有效面积", "类型" => "Any", "维度" => "标量"),
        Dict("名称" => "neff", "描述" => "有效折射率", "类型" => "Any", "维度" => "标量")
    ]
    
    println("1. 主数据集:")
    for ds in structure["主数据集"]
        println("   - $(ds["名称"]): $(ds["描述"]) [$(ds["类型"]), $(ds["维度"])]")
    end
    
    println("\n2. grid 组:")
    for ds in structure["grid 组"]
        println("   - $(ds["名称"]): $(ds["描述"]) [$(ds["类型"]), $(ds["维度"])]")
    end
    
    println("\n3. physics_features 组:")
    for ds in structure["physics_features"]
        println("   - $(ds["名称"]): $(ds["描述"]) [$(ds["类型"]), $(ds["维度"])]")
    end
    
    return structure
end

function validate_actual_file(file_path)
    println("\n=== 实际 HDF5 文件验证: $file_path ===")
    
    if !isfile(file_path)
        println("错误: 文件不存在!")
        return
    end
    
    h5open(file_path, "r") do fid
        println("\n1. 文件根目录内容:")
        for name in keys(fid)
            obj = fid[name]
            obj_type = typeof(obj)
            println("   - $name: $obj_type")
        end
        
        if haskey(fid, "grid")
            println("\n2. grid 组内容:")
            grid = fid["grid"]
            for name in keys(grid)
                ds = grid[name]
                dtype = eltype(ds)
                dims = size(ds)
                println("   - $name: 类型=$dtype, 维度=$dims")
            end
        else
            println("\n2. grid 组: 不存在")
        end
        
        if haskey(fid, "physics_features")
            println("\n3. physics_features 组内容:")
            pf = fid["physics_features"]
            for name in keys(pf)
                val = read(pf, name)
                dtype = typeof(val)
                println("   - $name: 值=$val, 类型=$dtype")
            end
        else
            println("\n3. physics_features 组: 不存在")
        end
        
        if haskey(fid, "Eω")
            Eω = fid["Eω"]
            println("\n4. Eω 数据集:")
            println("   - 类型: $(eltype(Eω))")
            println("   - 维度: $(size(Eω))")
            println("   - 数据范围: 实部 [$(minimum(real(Eω))) , $(maximum(real(Eω)))], 虚部 [$(minimum(imag(Eω))) , $(maximum(imag(Eω)))]")
        end
        
        if haskey(fid, "Et")
            Et = fid["Et"]
            println("\n5. Et 数据集:")
            println("   - 类型: $(eltype(Et))")
            println("   - 维度: $(size(Et))")
        else
            println("\n5. Et 数据集: 不存在")
        end
        
        if haskey(fid, "z")
            z = fid["z"]
            println("\n6. z 数据集:")
            println("   - 类型: $(eltype(z))")
            println("   - 维度: $(size(z))")
            println("   - 范围: [$(z[1]) , $(z[end])]")
        end
    end
end

function compare_structures(gen_struct, vis_struct)
    println("\n=== 结构兼容性对比 ===")
    
    all_matched = true
    
    println("\n1. 主数据集对比:")
    gen_names = [ds["名称"] for ds in gen_struct["主数据集"]]
    vis_names = [ds["名称"] for ds in vis_struct["主数据集"]]
    
    matched = Set(gen_names) == Set(vis_names)
    if matched
        println("   ✓ 数据集名称完全匹配")
        println("   生成: $gen_names")
        println("   读取: $vis_names")
    else
        println("   ✗ 数据集名称不匹配")
        println("   生成: $gen_names")
        println("   读取: $vis_names")
        all_matched = false
    end
    
    println("\n2. grid 组对比:")
    gen_grid_names = [ds["名称"] for ds in gen_struct["grid 组"]]
    vis_grid_names = [ds["名称"] for ds in vis_struct["grid 组"]]
    
    gen_set = Set(gen_grid_names)
    vis_set = Set(vis_grid_names)
    
    if gen_set == vis_set
        println("   ✓ grid 组数据集名称完全匹配")
    else
        missing_in_vis = gen_set - vis_set
        extra_in_vis = vis_set - gen_set
        if !isempty(missing_in_vis)
            println("   ⚠ 生成中有但读取中没有: $missing_in_vis")
        end
        if !isempty(extra_in_vis)
            println("   ⚠ 读取中有但生成中没有: $extra_in_vis")
        end
        all_matched = false
    end
    
    println("\n3. physics_features 组对比:")
    gen_pf_names = [ds["名称"] for ds in gen_struct["physics_features"]]
    vis_pf_names = [ds["名称"] for ds in vis_struct["physics_features"]]
    
    if Set(gen_pf_names) == Set(vis_pf_names)
        println("   ✓ physics_features 组数据集名称完全匹配")
        println("   包含: $gen_pf_names")
    else
        println("   ✗ physics_features 组数据集名称不匹配")
        println("   生成: $gen_pf_names")
        println("   读取: $vis_pf_names")
        all_matched = false
    end
    
    println("\n=== 兼容性总结 ===")
    if all_matched
        println("✓ 所有数据结构完全兼容!")
    else
        println("✗ 存在兼容性问题，请检查上述警告和错误")
    end
    
    return all_matched
end

gen_struct = analyze_generation_structure()
vis_struct = analyze_visualization_structure()

sample_files = [
    joinpath(@__DIR__, "sample_000030.h5"),
    joinpath(@__DIR__, "sample_000048.h5"),
    joinpath(@__DIR__, "sample_000001.h5")
]

for file in sample_files
    if isfile(file)
        validate_actual_file(file)
        break
    end
end

compare_structures(gen_struct, vis_struct)