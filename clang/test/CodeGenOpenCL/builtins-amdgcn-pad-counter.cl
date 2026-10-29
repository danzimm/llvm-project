// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -cl-std=CL2.0 -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// REQUIRES: amdgpu-registered-target

// The void hint must survive optimization in the conditional load arm.
// IR-LABEL: @hinted_kernel(
// IR: [[A:%.*]] = load i32
// IR: br i1 {{.*}}, label %[[JOIN:[^, ]+]], label %[[LOAD:[^, ]+]]
// IR: [[LOAD]]:
// IR: load i32
// IR: store i32
// IR-NEXT: {{(tail )?}}call void @llvm.amdgcn.pad.counter(i32 0, i32 1)
// IR: br label %[[JOIN]]
// IR: [[JOIN]]:
// IR: store i32 [[A]]

kernel void hinted_kernel(global int *data, global int *out,
                          int condition) {
  unsigned index = __builtin_amdgcn_workitem_id_x();
  int a = data[index];
  if (condition) {
    out[index + 64] = data[index + 64];
    __builtin_amdgcn_pad_counter(0, 1);
  }
  out[index] = a;
}
