// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu9.42-amd-amdhsa -DTEST_NONCONSTANT -fsyntax-only -verify %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu9.42-amd-amdhsa -DTEST_INVALID_POLICY -fsyntax-only -verify %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu9.0a-amd-amdhsa -DTEST_UNSUPPORTED -emit-llvm -o /dev/null -verify %s

#ifdef TEST_NONCONSTANT
void test_nonconstant(int cpol) {
  __builtin_amdgcn_buffer_inv(cpol); // expected-error {{argument to '__builtin_amdgcn_buffer_inv' must be a constant integer}}
}
#endif

#ifdef TEST_INVALID_POLICY
void test_invalid_policy(void) {
  __builtin_amdgcn_buffer_inv(2); // expected-error {{argument to '__builtin_amdgcn_buffer_inv' must be 0, 1, 16, or 17}}
}
#endif

#ifdef TEST_UNSUPPORTED
void test_unsupported(void) {
  __builtin_amdgcn_buffer_inv(0); // expected-error {{'__builtin_amdgcn_buffer_inv' needs target feature gfx940-insts}}
}
#endif
