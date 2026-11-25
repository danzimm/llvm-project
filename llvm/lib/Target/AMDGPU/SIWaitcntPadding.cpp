//===- SIWaitcntPadding.cpp - Balance waitcnts across branches -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pad the opposite edge of conditional branches guarding memory operations
// annotated with !amdgpu.balance.waits, before waitcnt insertion analysis.
//
//===----------------------------------------------------------------------===//

#include "SIWaitcntPadding.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "si-insert-waitcnts"

namespace {

[[maybe_unused]] void printIgnored(const MachineBasicBlock &Arm,
                                   StringRef Reason) {
  dbgs() << "Ignoring amdgpu-balance-waits in " << printMBBReference(Arm)
         << " of " << Arm.getParent()->getName() << ": " << Reason << '\n';
}

struct PaddingEdge {
  // When padding a triangle then PadBlockOrPred is the pred of the arm
  // requesting the padding. Otherwise it is the block to pad.
  MachineBasicBlock *PadBlockOrPred;
  MachineBasicBlock *Join;
};

std::optional<PaddingEdge> getCounterPaddingEdge(MachineBasicBlock &Arm,
                                                 MachineLoopInfo &MLI,
                                                 const SIInstrInfo &TII) {
  auto Reject = [&Arm](StringRef Reason) -> std::optional<PaddingEdge> {
    LLVM_DEBUG(printIgnored(Arm, Reason));
    return std::nullopt;
  };

  // We're looking for either a triangle:
  //
  //     Branch
  //      |   \
  //      |    \
  //      |     Arm
  //      |    /
  //      |   /
  //     Join
  //
  // or a diamond:
  //
  //      Branch
  //      /   \
  //     /     \
  // Other      Arm
  //     \     /
  //      \   /
  //       Join

  MachineBasicBlock *Branch = Arm.getSinglePredecessor();
  MachineBasicBlock *Join = Arm.getSingleSuccessor();
  if (!Branch || !Join)
    return Reject("arm lacks a single predecessor and successor");
  if (Branch->succ_size() != 2 || Join->pred_size() != 2)
    return Reject("branch or join does not have exactly two edges");
  if (Arm.isEHPad() || Join->isEHPad())
    return Reject("arm or join is an EH pad");

  auto Preds = Join->pred_begin();
  MachineBasicBlock *Other = *Preds != &Arm ? *Preds : *std::next(Preds);
  // Other is Branch from a triangle
  if (Other != Branch && (Other->getSinglePredecessor() != Branch ||
                          Other->getSingleSuccessor() != Join))
    return Reject("not a triangle or diamond");

  // If this is a triangle make sure we can insert a new block
  if (Other == Branch && !Branch->canSplitCriticalEdge(Join, &MLI))
    return Reject("branch-to-join edge cannot be split");
  return PaddingEdge{Other, Join};
}

} // end anonymous namespace

AMDGPU::CounterPaddingResult AMDGPU::expandCounterPaddingHints(
    MachineFunction &MF, MachineLoopInfo &MLI, MachineDominatorTree *MDT,
    MachinePostDominatorTree &PDT, const HardwareLimits &Limits) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo &TII = *ST.getInstrInfo();

  // In future this should encode the padding instruction, too, but for now
  // we only support `buffer_inv 0`.
  struct PaddingRequest {
    MachineBasicBlock *Join;
    unsigned Count;
    DebugLoc DL;
  };
  MapVector<MachineBasicBlock *, PaddingRequest> Requests;
  CounterPaddingResult Result;
  for (MachineBasicBlock &MBB : MF) {
    // The edge to pad only depends on the arm, so look it up lazily, once.
    std::optional<std::optional<PaddingEdge>> Edge;
    for (const MachineInstr &MI : MBB.instrs()) {
      if (!any_of(MI.memoperands(), [](const MachineMemOperand *MMO) {
            return MMO->getFlags() & MOBalanceWaits;
          }))
        continue;
      // Each annotated instruction accounts for one padding event, so that the
      // padding follows however many instructions the access was lowered to.
      // For now only LOAD/VM counter events are balanced.
      if (!SIInstrInfo::usesVM_CNT(MI)) {
        LLVM_DEBUG(printIgnored(MBB, "access does not use the VM counter"));
        continue;
      }
      if (!Edge)
        Edge = getCounterPaddingEdge(MBB, MLI, TII);
      if (!*Edge)
        continue;
      auto [It, Inserted] = Requests.try_emplace(
          (*Edge)->PadBlockOrPred,
          PaddingRequest{(*Edge)->Join, 1, MI.getDebugLoc()});
      if (!Inserted)
        ++It->second.Count;
    }
  }

  MachineDomTreeUpdater MDTU(MDT, &PDT,
                             MachineDomTreeUpdater::UpdateStrategy::Eager);
  for (const auto &[PadBlockOrPred, R] : Requests) {
    MachineBasicBlock *PadBlock = PadBlockOrPred;
    // When the matched CFG is a triangle PadBlock will be the pred and have 2
    // succ.
    if (PadBlock->succ_size() != 1) {
      MachineBasicBlock::SplitCriticalEdgeAnalyses Analyses{};
      Analyses.MLI = &MLI;
      PadBlock = PadBlock->SplitCriticalEdge(R.Join, Analyses, &MDTU);
      if (!PadBlock) {
        LLVM_DEBUG(printIgnored(*PadBlockOrPred,
                                "failed to split branch-to-join edge"));
        continue;
      }
      Result.ChangedCFG = true;
    }
    // Padding beyond the counter's wait range can't relax any further waits.
    unsigned Count = std::min(R.Count, Limits.LoadcntMax - 1);
    if (Count != R.Count)
      LLVM_DEBUG(dbgs() << "Clamping padding of "
                        << printMBBReference(*PadBlock) << " from " << R.Count
                        << " to " << Count << '\n');
    auto InsertPt = PadBlock->getFirstTerminator();
    for (unsigned I = 0; I != Count; ++I)
      BuildMI(*PadBlock, InsertPt, R.DL, TII.get(AMDGPU::BUFFER_INV)).addImm(0);
    Result.Changed = true;
    LLVM_DEBUG(dbgs() << "Padded " << printMBBReference(*PadBlock) << " with "
                      << Count << " BUFFER_INV for amdgpu-balance-waits in "
                      << MF.getName() << '\n');
  }
  return Result;
}
