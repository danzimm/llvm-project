//===- SIWaitcntPadding.h - Balance waitcnts across branches ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SIWAITCNTPADDING_H
#define LLVM_LIB_TARGET_AMDGPU_SIWAITCNTPADDING_H

namespace llvm {

class MachineDominatorTree;
class MachineFunction;
class MachineLoopInfo;
class MachinePostDominatorTree;

namespace AMDGPU {

struct HardwareLimits;

struct CounterPaddingResult {
  /// Padding was inserted.
  bool Changed = false;
  /// At least one edge was split to hold padding.
  bool ChangedCFG = false;
};

/// Materialize padding for memory operations annotated with
/// !amdgpu.balance.waits (MOBalanceWaits) in \p MF: each annotated instruction
/// in a supported conditional arm adds one counter event on the opposite edge,
/// so that waitcnt insertion analyzes it like any other counter event. The
/// subtarget must support padding. \p MLI, \p PDT and, if non-null, \p MDT are
/// kept up to date.
CounterPaddingResult expandCounterPaddingHints(MachineFunction &MF,
                                               MachineLoopInfo &MLI,
                                               MachineDominatorTree *MDT,
                                               MachinePostDominatorTree &PDT,
                                               const HardwareLimits &Limits);

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SIWAITCNTPADDING_H
