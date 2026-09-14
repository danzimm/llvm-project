; RUN: opt -passes='simplifycfg<sink-common-insts>' -S %s | FileCheck %s

; BUFFER_INV increments a wave-level counter, so calls reached by different
; lane sets must not be sunk and merged after control-flow reconvergence.

declare void @llvm.amdgcn.buffer.inv(i32 immarg)
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @side_then() nounwind
declare void @side_else() nounwind

; CHECK: declare void @llvm.amdgcn.buffer.inv(i32 immarg) #[[ATTR:[0-9]+]]

define amdgpu_kernel void @do_not_sink_buffer_inv() {
; CHECK-LABEL: define amdgpu_kernel void @do_not_sink_buffer_inv(
; CHECK:       then:
; CHECK-NEXT:    call void @side_then()
; CHECK-NEXT:    call void @llvm.amdgcn.buffer.inv(i32 0)
; CHECK-NEXT:    br label %join
; CHECK:       else:
; CHECK-NEXT:    call void @side_else()
; CHECK-NEXT:    call void @llvm.amdgcn.buffer.inv(i32 0)
; CHECK-NEXT:    br label %join
; CHECK:       join:
; CHECK-NEXT:    ret void
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp eq i32 %id, 0
  br i1 %cond, label %then, label %else

then:
  call void @side_then()
  call void @llvm.amdgcn.buffer.inv(i32 0)
  br label %join

else:
  call void @side_else()
  call void @llvm.amdgcn.buffer.inv(i32 0)
  br label %join

join:
  ret void
}

; CHECK: attributes #[[ATTR]] = { convergent
