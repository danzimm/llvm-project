// RUN: mlir-translate -mlir-to-llvmir %s -o %t.ll
// RUN: FileCheck %s --check-prefix=IR < %t.ll

// IR-LABEL: @rocdl_hint_kernel(
// IR: [[A:%.*]] = load i32
// IR: br i1 {{.*}}, label %[[LOAD:[^, ]+]], label %[[JOIN:[^, ]+]]
// IR: [[LOAD]]:
// IR: load i32
// IR: store i32
// IR: call void @llvm.amdgcn.pad.counter(i32 0, i32 1)
// IR: br label %[[JOIN]]
// IR: [[JOIN]]:
// IR: store i32 [[A]]

module attributes {llvm.target_triple = "amdgpu9.50-amd-amdhsa"} {
  llvm.func @rocdl_hint_kernel(%data: !llvm.ptr<1>, %out: !llvm.ptr<1>,
                             %condition: i32) attributes {rocdl.kernel} {
    %index = rocdl.workitem.id.x : i32
    %wide = llvm.zext %index : i32 to i64
    %a_ptr = llvm.getelementptr %data[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %a = llvm.load %a_ptr : !llvm.ptr<1> -> i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %cond = llvm.icmp "ne" %condition, %zero : i32
    llvm.cond_br %cond, ^load, ^join
  ^load:
    %offset = llvm.mlir.constant(64 : i64) : i64
    %b_index = llvm.add %wide, %offset : i64
    %b_ptr = llvm.getelementptr %data[%b_index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %b = llvm.load %b_ptr : !llvm.ptr<1> -> i32
    %b_out = llvm.getelementptr %out[%b_index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %b, %b_out : i32, !llvm.ptr<1>
    rocdl.pad.counter 0, 1
    llvm.br ^join
  ^join:
    %out_ptr = llvm.getelementptr %out[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %a, %out_ptr : i32, !llvm.ptr<1>
    llvm.return
  }
}
