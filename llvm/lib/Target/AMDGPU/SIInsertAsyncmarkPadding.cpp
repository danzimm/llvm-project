//===- SIInsertAsyncmarkPadding.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Guide eager branch padding with asyncmarks.
//
// This pass runs immediately before SIInsertWaitcnts. It predicts, from
// asyncmark history, that a later async wait can select an inherited
// checkpoint across a branch join, and records the LOAD-counter imbalance
// between the two joining edges as a PAD_COUNTER hint on the longer edge.
// SIInsertWaitcnts expands accepted hints into real BUFFER_INV padding
// before its ordinary solver runs, so this pass never splits edges or
// emits padding itself.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUHWEvents.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include <optional>

using namespace llvm;

using HWEvents = AMDGPU::HWEvents;

#define DEBUG_TYPE "si-insert-asyncmark-padding"

namespace {

// Match the asyncmark history cap used when merging score brackets in
// SIInsertWaitcnts. This analysis only asks whether a wait could select an
// inherited checkpoint; the normal waitcnt analysis determines its
// completion requirements.
constexpr unsigned AsyncmarkHistoryLimit = 16;

unsigned countRemainingAsyncmarks(const MachineBasicBlock &MBB,
                                  unsigned Count) {
  for (const MachineInstr &MI : MBB) {
    if (MI.getOpcode() == AMDGPU::ASYNCMARK)
      Count = std::min(Count + 1, AsyncmarkHistoryLimit);
    else if (MI.getOpcode() == AMDGPU::WAIT_ASYNCMARK)
      Count = std::min(Count, unsigned(MI.getOperand(0).getImm()));
  }
  return Count;
}

// A maximum over paths deliberately admits possible inherited history, rather
// than requiring a single static ASYNCMARK to dominate the branch. In
// particular, an asyncmark in a loop's suffix can be inherited on the next
// iteration. Saturating before a wait with N >= 16 can overestimate the
// remaining history; candidate waits below the cap use the usual recency
// interpretation.
DenseSet<MachineBasicBlock *> findAsyncmarkPaddingHeaders(MachineFunction &MF) {
  DenseMap<MachineBasicBlock *, unsigned> IncomingMarks;
  SmallVector<MachineBasicBlock *> Worklist{&MF.front()};
  IncomingMarks[&MF.front()] = 0;
  while (!Worklist.empty()) {
    MachineBasicBlock *MBB = Worklist.pop_back_val();
    unsigned Count = countRemainingAsyncmarks(*MBB, IncomingMarks.lookup(MBB));
    for (MachineBasicBlock *Succ : MBB->successors()) {
      auto [It, Inserted] = IncomingMarks.try_emplace(Succ, Count);
      if (Inserted || Count > It->second) {
        It->second = Count;
        Worklist.push_back(Succ);
      }
    }
  }

  struct NextWait {
    const MachineInstr *MI;
    unsigned NewMarks;
  };
  DenseSet<MachineBasicBlock *> Headers;
  for (MachineBasicBlock &Header : MF) {
    if (Header.succ_size() != 2 || !IncomingMarks.contains(&Header))
      continue;
    unsigned InheritedMarks =
        countRemainingAsyncmarks(Header, IncomingMarks.lookup(&Header));
    if (!InheritedMarks)
      continue;

    // Scan only to a common first async wait. Entering a block before its
    // result is known detects a cycle before that wait and rejects the
    // corridor.
    DenseMap<MachineBasicBlock *, std::optional<NextWait>> NextWaits;
    NextWaits[&Header] = std::nullopt;
    auto FindWait = [&](auto &&Self,
                        MachineBasicBlock *MBB) -> std::optional<NextWait> {
      auto [It, Inserted] = NextWaits.try_emplace(MBB, std::nullopt);
      if (!Inserted)
        return It->second;

      unsigned NewMarks = 0;
      for (const MachineInstr &MI : *MBB) {
        if (MI.getOpcode() == AMDGPU::ASYNCMARK) {
          if (++NewMarks == AsyncmarkHistoryLimit)
            return std::nullopt;
        } else if (MI.getOpcode() == AMDGPU::WAIT_ASYNCMARK) {
          return NextWaits[MBB] = NextWait{&MI, NewMarks};
        }
      }

      std::optional<NextWait> Result;
      for (MachineBasicBlock *Succ : MBB->successors()) {
        auto Other = Self(Self, Succ);
        if (!Other || (Result && (Other->MI != Result->MI ||
                                  Other->NewMarks != Result->NewMarks)))
          return std::nullopt;
        Result = Other;
      }
      if (!Result || Result->NewMarks + NewMarks >= AsyncmarkHistoryLimit)
        return std::nullopt;
      Result->NewMarks += NewMarks;
      return NextWaits[MBB] = Result;
    };

    std::optional<NextWait> Wait;
    bool Supported = true;
    for (MachineBasicBlock *Succ : Header.successors()) {
      auto Other = FindWait(FindWait, Succ);
      if (!Other || (Wait && (Other->MI != Wait->MI ||
                              Other->NewMarks != Wait->NewMarks))) {
        Supported = false;
        break;
      }
      Wait = Other;
    }
    if (!Supported)
      continue;

    // No outside path may introduce a different marker count before the wait.
    for (const auto &[MBB, Result] : NextWaits) {
      if (MBB == &Header)
        continue;
      for (MachineBasicBlock *Pred : MBB->predecessors())
        Supported &= NextWaits.contains(Pred);
    }
    if (!Supported)
      continue;

    unsigned N = Wait->MI->getOperand(0).getImm();
    if (N >= AsyncmarkHistoryLimit || N < Wait->NewMarks)
      continue;
    if (InheritedMarks > N - Wait->NewMarks)
      Headers.insert(&Header);
  }
  return Headers;
}

class SIInsertAsyncmarkPadding {
  MachineLoopInfo &MLI;
  MachineFunction &MF;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const bool TgSplit;
  const bool IsExpertMode;
  HWEvents LoadEventMask;
  unsigned LoadLimit;

public:
  SIInsertAsyncmarkPadding(MachineLoopInfo &MLI, MachineFunction &MF)
      : MLI(MLI), MF(MF), ST(MF.getSubtarget<GCNSubtarget>()),
        TII(*ST.getInstrInfo()),
        TgSplit(ST.hasTgSplitSupport() &&
                AMDGPU::isTgSplitEnabled(MF.getFunction())),
        IsExpertMode(ST.hasExtendedWaitCounts() &&
                     AMDGPU::isExpertSchedulingMode(ST, MF.getFunction())) {
    if (ST.hasExtendedWaitCounts()) {
      LoadEventMask = HWEvents::VMEM_READ_ACCESS | HWEvents::VMEM_INV_ACCESS;
    } else {
      LoadEventMask =
          HWEvents::VMEM_READ_ACCESS | HWEvents::VMEM_SAMPLER_READ_ACCESS |
          HWEvents::VMEM_BVH_READ_ACCESS | HWEvents::VMEM_INV_ACCESS;
      if (!ST.hasVscnt())
        LoadEventMask |=
            HWEvents::VMEM_WRITE_ACCESS | HWEvents::SCRATCH_WRITE_ACCESS;
    }
    LoadLimit = AMDGPU::HardwareLimits(AMDGPU::getIsaVersion(ST.getCPU()))
                    .get(AMDGPU::LOAD_CNT);
  }

  bool run();

private:
  struct JoinShape {
    MachineBasicBlock *Branch;
    MachineBasicBlock *ArmA;
    MachineBasicBlock *ArmB;
  };

  std::optional<JoinShape>
  getJoinShape(MachineBasicBlock &Join,
               const DenseSet<MachineBasicBlock *> &Headers) const;
  std::optional<unsigned> countLoadEvents(MachineBasicBlock &Arm) const;
};

class SIInsertAsyncmarkPaddingLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIInsertAsyncmarkPaddingLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.setPreservesCFG();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "SI Insert Asyncmark Padding";
  }
};

std::optional<SIInsertAsyncmarkPadding::JoinShape>
SIInsertAsyncmarkPadding::getJoinShape(
    MachineBasicBlock &Join,
    const DenseSet<MachineBasicBlock *> &Headers) const {
  if (Join.pred_size() != 2 || Join.isEHPad())
    return std::nullopt;

  MachineLoop *Loop = MLI.getLoopFor(&Join);
  if (Loop && (Loop->getHeader() == &Join || Loop->isLoopLatch(&Join)))
    return std::nullopt;

  auto Pred = Join.pred_begin();
  MachineBasicBlock *A = *Pred++;
  MachineBasicBlock *B = *Pred;
  auto GetArmPredecessor = [&](MachineBasicBlock *Arm) {
    return Arm->getSingleSuccessor() == &Join ? Arm->getSinglePredecessor()
                                              : nullptr;
  };
  MachineBasicBlock *PredA = GetArmPredecessor(A);
  MachineBasicBlock *PredB = GetArmPredecessor(B);
  MachineBasicBlock *Branch = nullptr;
  if (PredA == B || PredB == A)
    Branch = PredA == B ? B : A;
  else if (PredA && PredA == PredB)
    Branch = PredA;

  if (!Branch || Branch->succ_size() != 2 || MLI.getLoopFor(Branch) != Loop ||
      MLI.getLoopFor(A) != Loop || MLI.getLoopFor(B) != Loop)
    return std::nullopt;

  // Only split ordinary direct branches. The nonempty condition also excludes
  // two-successor blocks whose transfer is not represented by a conditional
  // branch that can be retargeted.
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (TII.analyzeBranch(*Branch, TBB, FBB, Cond) || Cond.empty())
    return std::nullopt;
  if ((Branch == A || Branch == B) &&
      !Branch->canSplitCriticalEdge(&Join, &MLI))
    return std::nullopt;

  for (MachineBasicBlock *Arm : {A, B}) {
    if (Arm == Branch)
      continue;
    for (const MachineInstr &MI : Arm->terminators())
      if (MI.getOpcode() != AMDGPU::S_BRANCH)
        return std::nullopt;
  }
  if (!Headers.contains(Branch))
    return std::nullopt;
  return JoinShape{Branch, A, B};
}

// Count LOAD-counter events issued by a single-block arm. The edge taken
// directly from the branch block issues none. Bail out on anything that
// can drain or otherwise redefine the counter mid-arm, where this simple
// count would no longer match the waitcnt solver's outstanding count.
std::optional<unsigned>
SIInsertAsyncmarkPadding::countLoadEvents(MachineBasicBlock &Arm) const {
  unsigned Count = 0;
  for (const MachineInstr &MI : Arm) {
    switch (MI.getOpcode()) {
    case AMDGPU::PAD_COUNTER:
    case AMDGPU::S_WAITCNT:
    case AMDGPU::S_WAIT_LOADCNT:
    case AMDGPU::S_WAITCNT_DEPCTR:
    case AMDGPU::WAIT_ASYNCMARK:
      return std::nullopt;
    default:
      break;
    }
    if (MI.isCall())
      return std::nullopt;
    Count +=
        (AMDGPU::getEventsFor(MI, ST, IsExpertMode, TgSplit) & LoadEventMask)
            .size();
  }
  return Count;
}

bool SIInsertAsyncmarkPadding::run() {
  // BUFFER_INV 0 is the only supported padding operation, and padding is
  // an optimization only. SIInsertWaitcnts revalidates every hint, so a
  // hint emitted here can never affect correctness on its own.
  if (!ST.hasBufferInvInst() || MF.getFunction().hasOptNone() ||
      MF.getTarget().getOptLevel() == CodeGenOptLevel::None)
    return false;

  DenseSet<MachineBasicBlock *> Headers = findAsyncmarkPaddingHeaders(MF);
  if (Headers.empty())
    return false;

  bool Modified = false;
  DenseSet<MachineBasicBlock *> PaddedBranches;
  for (MachineBasicBlock &Join : MF) {
    auto Shape = getJoinShape(Join, Headers);
    if (!Shape || !PaddedBranches.insert(Shape->Branch).second)
      continue;

    // The direct branch edge issues no arm events; a single-block arm is
    // counted in full. An existing user hint already claims this branch.
    auto CountFor = [&](MachineBasicBlock *Arm) -> std::optional<unsigned> {
      if (Arm == Shape->Branch)
        return 0;
      return countLoadEvents(*Arm);
    };
    auto CountA = CountFor(Shape->ArmA);
    auto CountB = CountFor(Shape->ArmB);
    if (!CountA || !CountB || *CountA == *CountB)
      continue;

    MachineBasicBlock *Longer = *CountA > *CountB ? Shape->ArmA : Shape->ArmB;
    unsigned Amount = std::min(
        std::max(*CountA, *CountB) - std::min(*CountA, *CountB), LoadLimit - 1);
    if (!Amount || Longer == Shape->Branch)
      continue;

    BuildMI(*Longer, Longer->getFirstTerminator(), DebugLoc(),
            TII.get(AMDGPU::PAD_COUNTER))
        .addImm(0)
        .addImm(Amount);
    Modified = true;
    LLVM_DEBUG({
      dbgs() << "Asyncmark branch padding: ";
      Longer->printName(dbgs());
      dbgs() << " -> ";
      Join.printName(dbgs());
      dbgs() << " by " << Amount << '\n';
    });
  }
  return Modified;
}

} // namespace

bool SIInsertAsyncmarkPaddingLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  return SIInsertAsyncmarkPadding(MLI, MF).run();
}

PreservedAnalyses
SIInsertAsyncmarkPaddingPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  // Inserting a hint does not change the CFG or any cached analysis.
  SIInsertAsyncmarkPadding(MLI, MF).run();
  return PreservedAnalyses::all();
}

char SIInsertAsyncmarkPaddingLegacy::ID = 0;

char &llvm::SIInsertAsyncmarkPaddingID = SIInsertAsyncmarkPaddingLegacy::ID;

INITIALIZE_PASS_BEGIN(SIInsertAsyncmarkPaddingLegacy, DEBUG_TYPE,
                      "SI Insert Asyncmark Padding", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SIInsertAsyncmarkPaddingLegacy, DEBUG_TYPE,
                    "SI Insert Asyncmark Padding", false, false)

FunctionPass *llvm::createSIInsertAsyncmarkPaddingPass() {
  return new SIInsertAsyncmarkPaddingLegacy();
}
