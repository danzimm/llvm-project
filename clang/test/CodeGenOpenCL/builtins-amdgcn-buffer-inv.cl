// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu9.42-amd-amdhsa -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -O2 -cl-std=CL2.0 -triple amdgpu9.42-amd-amdhsa -S \
// RUN:   -mllvm -structurizecfg-skip-uniform-regions -o - %s | FileCheck --check-prefix=ASM %s

// Preserve the uniform branch so this test isolates explicit source-level
// padding. The default StructurizeCFG flow region conservatively requires
// vmcnt(0) at the join even when the short path contains buffer_inv.

// CHECK-LABEL: @test_buffer_inv(
// CHECK: call void @llvm.amdgcn.buffer.inv(i32 0)
// CHECK: call void @llvm.amdgcn.buffer.inv(i32 1)
// CHECK: call void @llvm.amdgcn.buffer.inv(i32 16)
// CHECK: call void @llvm.amdgcn.buffer.inv(i32 17)
void test_buffer_inv(void) {
  __builtin_amdgcn_buffer_inv(0);
  __builtin_amdgcn_buffer_inv(1);
  __builtin_amdgcn_buffer_inv(16);
  __builtin_amdgcn_buffer_inv(17);
}

kernel void short_path_unpadded(global int *in, global int *out,
                                int take_long_path) {
  unsigned id = __builtin_amdgcn_workitem_id_x();
  int common = in[id];
  if (take_long_path)
    (void)*(volatile global int *)(in + id + 1);
  out[id] = common + 1;
}

// ASM-LABEL: short_path_unpadded:
// ASM:       global_load_dword
// ASM-NOT:   buffer_inv
// ASM:       ; %__clang_ocl_kern_imp_short_path_unpadded.exit
// ASM-NEXT:  s_waitcnt vmcnt(0)
// ASM-NEXT:  v_add_{{.*}}32

kernel void short_path_padded(global int *in, global int *out,
                              int take_long_path) {
  unsigned id = __builtin_amdgcn_workitem_id_x();
  int common = in[id];
  if (take_long_path)
    (void)*(volatile global int *)(in + id + 1);
  else
    __builtin_amdgcn_buffer_inv(0);
  out[id] = common + 1;
}

// ASM-LABEL: short_path_padded:
// ASM:       global_load_dword
// ASM:       buffer_inv
// ASM:       ; %__clang_ocl_kern_imp_short_path_padded.exit
// ASM-NEXT:  s_waitcnt vmcnt(1)
// ASM-NEXT:  v_add_{{.*}}32
