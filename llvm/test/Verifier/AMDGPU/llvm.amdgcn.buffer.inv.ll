; RUN: split-file %s %t
; RUN: not llvm-as %t/nonconstant.ll -disable-output 2>&1 | FileCheck --check-prefix=NONCONSTANT %s
; RUN: not llvm-as %t/invalid-cache-policy.ll -disable-output 2>&1 | FileCheck --check-prefix=INVALID-POLICY %s

;--- nonconstant.ll
declare void @llvm.amdgcn.buffer.inv(i32 immarg)

define void @nonconstant(i32 %cpol) {
  ; NONCONSTANT: immarg operand has non-immediate parameter
  ; NONCONSTANT-NEXT: i32 %cpol
  ; NONCONSTANT-NEXT: call void @llvm.amdgcn.buffer.inv(i32 %cpol)
  call void @llvm.amdgcn.buffer.inv(i32 %cpol)
  ret void
}

;--- invalid-cache-policy.ll
declare void @llvm.amdgcn.buffer.inv(i32 immarg)

define void @invalid_cache_policy() {
  ; INVALID-POLICY: immarg value 2 for arg 0 out of range set
  ; INVALID-POLICY-NEXT: call void @llvm.amdgcn.buffer.inv(i32 2)
  call void @llvm.amdgcn.buffer.inv(i32 2)
  ret void
}
