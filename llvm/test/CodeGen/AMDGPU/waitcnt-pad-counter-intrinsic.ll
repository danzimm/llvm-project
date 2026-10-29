; RUN: llc -O2 -mtriple=amdgpu9.50-amd-amdhsa -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -O2 -global-isel=1 -global-isel-abort=1 -mtriple=amdgpu9.50-amd-amdhsa -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -O2 -mtriple=amdgpu9.0a-amd-amdhsa -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=NO-PAD --implicit-check-not=buffer_inv

; This real LLVM input exercises both instruction selectors and branch-arm
; association. The hint is not a counted event beside the conditional store.
; CHECK-LABEL: hint_counter_intrinsic:
; CHECK: global_load_dword
; CHECK: s_cbranch_scc{{[01]}} [[PAD:\.LBB[0-9]+_[0-9]+]]
; CHECK: global_load_dword
; CHECK: global_store_dword
; CHECK-NOT: buffer_inv
; CHECK: [[JOIN:\.LBB[0-9]+_[0-9]+]]:
; CHECK: s_waitcnt vmcnt(1)
; CHECK: global_store_dword
; CHECK: s_endpgm
; CHECK: [[PAD]]:
; CHECK: buffer_inv
; CHECK: s_branch [[JOIN]]
; NO-PAD-LABEL: hint_counter_intrinsic:
; NO-PAD: s_waitcnt vmcnt(0)
define amdgpu_kernel void @hint_counter_intrinsic(ptr addrspace(1) %data,
                                                ptr addrspace(1) %out,
                                                i32 %condition) {
entry:
  %index = call i32 @llvm.amdgcn.workitem.id.x()
  %wide = zext i32 %index to i64
  %a.ptr = getelementptr i32, ptr addrspace(1) %data, i64 %wide
  %a = load i32, ptr addrspace(1) %a.ptr, align 4
  %cond = icmp ne i32 %condition, 0
  br i1 %cond, label %long, label %join
long:
  %offset = add i64 %wide, 64
  %b.ptr = getelementptr i32, ptr addrspace(1) %data, i64 %offset
  %b = load i32, ptr addrspace(1) %b.ptr, align 4
  %b.out = getelementptr i32, ptr addrspace(1) %out, i64 %offset
  store i32 %b, ptr addrspace(1) %b.out, align 4
  call void @llvm.amdgcn.pad.counter(i32 0, i32 1)
  br label %join
join:
  %a.out = getelementptr i32, ptr addrspace(1) %out, i64 %wide
  store i32 %a, ptr addrspace(1) %a.out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.pad.counter(i32 immarg, i32 immarg)
