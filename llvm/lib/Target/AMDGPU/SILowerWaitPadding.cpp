//===- SILowerWaitPadding.cpp - Lower wait counter padding hints --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Materialize exact-count padding hints before waitcnt insertion analyzes the
// machine function. The marker itself contributes no counter event and is
// always removed, including when its optional padding is skipped.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include <optional>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "si-lower-wait-padding"

namespace {

std::optional<std::pair<MachineBasicBlock *, MachineBasicBlock *>>
getCounterPaddingEdge(MachineBasicBlock &Arm, MachineLoopInfo &MLI,
                      const SIInstrInfo &TII) {
  MachineBasicBlock *Branch = Arm.getSinglePredecessor();
  MachineBasicBlock *Join = Arm.getSingleSuccessor();
  if (!Branch || !Join || Branch->succ_size() != 2 || Join->pred_size() != 2 ||
      Arm.isEHPad() || Join->isEHPad())
    return std::nullopt;

  MachineBasicBlock *Other = nullptr;
  for (MachineBasicBlock *Pred : Join->predecessors())
    if (Pred != &Arm)
      Other = Pred;
  if (!Other || (Other != Branch && (Other->getSinglePredecessor() != Branch ||
                                     Other->getSingleSuccessor() != Join)))
    return std::nullopt;

  MachineLoop *Loop = MLI.getLoopFor(Join);
  if (MLI.getLoopFor(Branch) != Loop || MLI.getLoopFor(&Arm) != Loop ||
      MLI.getLoopFor(Other) != Loop ||
      (Loop && (Loop->getHeader() == Join || Loop->isLoopLatch(Join))))
    return std::nullopt;

  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (TII.analyzeBranch(*Branch, TBB, FBB, Cond) || Cond.empty())
    return std::nullopt;
  for (MachineBasicBlock *MBB : {&Arm, Other}) {
    if (MBB == Branch)
      continue;
    for (const MachineInstr &MI : MBB->terminators())
      if (MI.getOpcode() != AMDGPU::S_BRANCH)
        return std::nullopt;
  }
  if (Other == Branch && !Other->canSplitCriticalEdge(Join, &MLI))
    return std::nullopt;
  return std::make_pair(Other, Join);
}

bool lowerWaitPadding(MachineFunction &MF, MachineLoopInfo &MLI,
                      MachineDominatorTree *MDT, MachinePostDominatorTree *PDT,
                      bool EnablePadding) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo &TII = *ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII.getRegisterInfo();
  const AMDGPU::HardwareLimits Limits(AMDGPU::getIsaVersion(ST.getCPU()));
  EnablePadding &= ST.hasBufferInvInst() &&
                   MF.getTarget().getOptLevel() != CodeGenOptLevel::None;

  struct Request {
    MachineBasicBlock *Branch;
    MachineBasicBlock *Pred;
    MachineBasicBlock *Join;
    unsigned Count;
    DebugLoc DL;
  };
  SmallVector<Request, 8> Requests;
  DenseSet<MachineBasicBlock *> SeenBranches, RepeatedBranches;
  bool Modified = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB.instrs())) {
      if (MI.getOpcode() != AMDGPU::PAD_COUNTER)
        continue;
      int64_t Counter = MI.getOperand(0).getImm();
      int64_t Count = MI.getOperand(1).getImm();
      if (EnablePadding && Counter == 0 && Count > 0 &&
          uint64_t(Count) <= Limits.LoadcntMax - 1) {
        if (auto Edge = getCounterPaddingEdge(MBB, MLI, TII)) {
          MachineBasicBlock *Branch = MBB.getSinglePredecessor();
          if (!SeenBranches.insert(Branch).second)
            RepeatedBranches.insert(Branch);
          Requests.push_back({Branch, Edge->first, Edge->second,
                              unsigned(Count), MI.getDebugLoc()});
        }
      }
      // An ignored hint grants no permission to weaken a wait and must not
      // leave a target pseudo behind when its optimization is skipped.
      MI.eraseFromParent();
      Modified = true;
    }
  }

  MachineDomTreeUpdater MDTU(MDT, PDT,
                             MachineDomTreeUpdater::UpdateStrategy::Eager);
  for (const Request &R : Requests) {
    if (RepeatedBranches.contains(R.Branch))
      continue;
    MachineBasicBlock *PadBlock = R.Pred;
    if (R.Pred->succ_size() != 1) {
      bool WasLayoutEdge = R.Pred->isLayoutSuccessor(R.Join);
      MachineBasicBlock::SplitCriticalEdgeAnalyses Analyses{};
      Analyses.MLI = &MLI;
      PadBlock = R.Pred->SplitCriticalEdge(R.Join, Analyses, &MDTU);
      if (!PadBlock)
        continue;
      PadBlock->setSectionID(R.Pred->getSectionID());
      if (!WasLayoutEdge) {
        // The splitter repairs branches for its initial layout. Move a taken
        // edge out of line and repair the predecessor's fallthrough again.
        MF.splice(MF.end(), PadBlock);
        R.Pred->updateTerminator(PadBlock);
      }
    }
    PadBlock->addLiveIn(TRI.getExec());
    PadBlock->sortUniqueLiveIns();
    auto InsertPt = PadBlock->getFirstTerminator();
    for (unsigned I = 0; I != R.Count; ++I)
      BuildMI(*PadBlock, InsertPt, R.DL, TII.get(AMDGPU::BUFFER_INV)).addImm(0);
  }
  return Modified;
}

class SILowerWaitPaddingLegacy : public MachineFunctionPass {
public:
  static char ID;

  SILowerWaitPaddingLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    auto *MDTWrapper =
        getAnalysisIfAvailable<MachineDominatorTreeWrapperPass>();
    auto *MDT = MDTWrapper ? &MDTWrapper->getDomTree() : nullptr;
    auto *PDTWrapper =
        getAnalysisIfAvailable<MachinePostDominatorTreeWrapperPass>();
    auto *PDT = PDTWrapper ? &PDTWrapper->getPostDomTree() : nullptr;
    // Only the padding is optional; marker removal is required lowering.
    return lowerWaitPadding(MF, MLI, MDT, PDT, !skipFunction(MF.getFunction()));
  }

  StringRef getPassName() const override { return "SI Lower Wait Padding"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addUsedIfAvailable<MachineDominatorTreeWrapperPass>();
    AU.addUsedIfAvailable<MachinePostDominatorTreeWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachinePostDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SILowerWaitPaddingLegacy, DEBUG_TYPE,
                      "SI Lower Wait Padding", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(SILowerWaitPaddingLegacy, DEBUG_TYPE,
                    "SI Lower Wait Padding", false, false)

char SILowerWaitPaddingLegacy::ID = 0;
char &llvm::SILowerWaitPaddingID = SILowerWaitPaddingLegacy::ID;

FunctionPass *llvm::createSILowerWaitPaddingPass() {
  return new SILowerWaitPaddingLegacy();
}

PreservedAnalyses
SILowerWaitPaddingPass::run(MachineFunction &MF,
                            MachineFunctionAnalysisManager &MFAM) {
  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  auto *MDT = MFAM.getCachedResult<MachineDominatorTreeAnalysis>(MF);
  auto *PDT = MFAM.getCachedResult<MachinePostDominatorTreeAnalysis>(MF);
  const Function &F = MF.getFunction();
  bool EnablePadding =
      !F.hasOptNone() && !shouldSkipOptimizationForOptBisect(F);
  if (!lowerWaitPadding(MF, MLI, MDT, PDT, EnablePadding))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses()
      .preserve<MachineLoopAnalysis>()
      .preserve<MachineDominatorTreeAnalysis>()
      .preserve<MachinePostDominatorTreeAnalysis>();
}
