//===- SIWaitcntPadding.cpp - Counter padding hints for waitcnts ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower waitcnt padding hints to actual padding before waitcnt insertion
// analysis. Branches guarded by asyncmark history are padded the same way
// without a hint.
//
//===----------------------------------------------------------------------===//

#include "SIWaitcntPadding.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/Support/AMDGPUAsyncStages.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "si-insert-waitcnts"

namespace {

[[maybe_unused]] void printIgnored(const MachineBasicBlock &Arm,
                                   StringRef Reason) {
  dbgs() << "Ignoring PAD_WAITCNT in " << printMBBReference(Arm) << " of "
         << Arm.getParent()->getName() << ": " << Reason << '\n';
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

// Match the asyncmark history cap used when merging score brackets in
// SIInsertWaitcnts. This analysis only asks whether a wait could select an
// inherited checkpoint; the normal waitcnt analysis determines its completion
// requirements.
constexpr unsigned AsyncmarkHistoryLimit = 16;

// Return the stages named by an ASYNCMARK or WAIT_ASYNCMARK, or none for any
// other instruction.
AMDGPU::AsyncStages getAsyncmarkStages(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AMDGPU::ASYNCMARK:
    return AMDGPU::asyncStagesFromMaskOperand(MI.getOperand(0).getImm());
  case AMDGPU::WAIT_ASYNCMARK:
    return AMDGPU::asyncStagesFromMaskOperand(MI.getOperand(1).getImm());
  default:
    return AMDGPU::AsyncStages::NONE;
  }
}

bool isAsyncmarkFor(const MachineInstr &MI, AMDGPU::AsyncStages Stage) {
  return MI.getOpcode() == AMDGPU::ASYNCMARK &&
         any(getAsyncmarkStages(MI) & Stage);
}

bool isAsyncWaitFor(const MachineInstr &MI, AMDGPU::AsyncStages Stage) {
  return MI.getOpcode() == AMDGPU::WAIT_ASYNCMARK &&
         any(getAsyncmarkStages(MI) & Stage);
}

unsigned countRemainingAsyncmarks(const MachineBasicBlock &MBB,
                                  AMDGPU::AsyncStages Stage, unsigned Count) {
  for (const MachineInstr &MI : MBB) {
    if (isAsyncmarkFor(MI, Stage))
      Count = std::min(Count + 1, AsyncmarkHistoryLimit);
    else if (isAsyncWaitFor(MI, Stage))
      Count = std::min(Count, unsigned(MI.getOperand(0).getImm()));
  }
  return Count;
}

// Add to \p Branches the branch blocks after which a later async wait on
// \p Stage can select an inherited checkpoint. Marks and waits are tracked per
// stage, as in SIInsertWaitcnts.
//
// A maximum over paths deliberately admits possible inherited history, rather
// than requiring a single static ASYNCMARK to dominate the branch. In
// particular, an asyncmark in a loop's suffix can be inherited on the next
// iteration. Saturating before a wait with N >= 16 can overestimate the
// remaining history; candidate waits below the cap use the usual recency
// interpretation.
void findAsyncmarkPaddingBranches(MachineFunction &MF,
                                  AMDGPU::AsyncStages Stage,
                                  DenseSet<MachineBasicBlock *> &Branches) {
  DenseMap<MachineBasicBlock *, unsigned> IncomingMarks;
  SmallVector<MachineBasicBlock *> Worklist{&MF.front()};
  IncomingMarks[&MF.front()] = 0;
  while (!Worklist.empty()) {
    MachineBasicBlock *MBB = Worklist.pop_back_val();
    unsigned Count =
        countRemainingAsyncmarks(*MBB, Stage, IncomingMarks.lookup(MBB));
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
  for (MachineBasicBlock &Branch : MF) {
    if (Branch.succ_size() != 2 || !IncomingMarks.contains(&Branch))
      continue;
    unsigned InheritedMarks =
        countRemainingAsyncmarks(Branch, Stage, IncomingMarks.lookup(&Branch));
    if (!InheritedMarks)
      continue;

    // Scan only to a common first async wait. Entering a block before its
    // result is known detects a cycle before that wait and rejects the
    // corridor.
    DenseMap<MachineBasicBlock *, std::optional<NextWait>> NextWaits;
    NextWaits[&Branch] = std::nullopt;
    auto FindWait = [&](auto &&Self,
                        MachineBasicBlock *MBB) -> std::optional<NextWait> {
      auto [It, Inserted] = NextWaits.try_emplace(MBB, std::nullopt);
      if (!Inserted)
        return It->second;

      unsigned NewMarks = 0;
      for (const MachineInstr &MI : *MBB) {
        if (isAsyncmarkFor(MI, Stage)) {
          if (++NewMarks == AsyncmarkHistoryLimit)
            return std::nullopt;
        } else if (isAsyncWaitFor(MI, Stage)) {
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
    for (MachineBasicBlock *Succ : Branch.successors()) {
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
      if (MBB == &Branch)
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
      Branches.insert(&Branch);
  }
}

DenseSet<MachineBasicBlock *>
findAsyncmarkPaddingBranches(MachineFunction &MF) {
  AMDGPU::AsyncStages Marked = AMDGPU::AsyncStages::NONE;
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB)
      if (MI.getOpcode() == AMDGPU::ASYNCMARK)
        Marked |= getAsyncmarkStages(MI);

  DenseSet<MachineBasicBlock *> Branches;
  for (AMDGPU::AsyncStages Stage : AMDGPU::asyncStages(Marked))
    findAsyncmarkPaddingBranches(MF, Stage, Branches);
  return Branches;
}

struct JoinShape {
  MachineBasicBlock *Branch;
  MachineBasicBlock *ArmA;
  MachineBasicBlock *ArmB;
};

// Match a triangle or diamond ending at \p Join whose branch is in \p Branches.
// In a triangle one of the arms is the branch block itself.
std::optional<JoinShape>
getAsyncmarkJoinShape(MachineBasicBlock &Join,
                      const DenseSet<MachineBasicBlock *> &Branches,
                      MachineLoopInfo &MLI, const SIInstrInfo &TII) {
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

  if (!Branch || !Branches.contains(Branch) || Branch->succ_size() != 2 ||
      MLI.getLoopFor(Branch) != Loop || MLI.getLoopFor(A) != Loop ||
      MLI.getLoopFor(B) != Loop)
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
  return JoinShape{Branch, A, B};
}

// Count LOAD counter events issued by a single-block arm. Bail out on anything
// that can drain or otherwise redefine the counter mid-arm, where this simple
// count would no longer match the waitcnt solver's outstanding count.
std::optional<unsigned> countLoadEvents(const MachineBasicBlock &Arm,
                                        const GCNSubtarget &ST,
                                        AMDGPU::HWEvents LoadEvents,
                                        bool IsExpertMode, bool TgSplit) {
  unsigned Count = 0;
  for (const MachineInstr &MI : Arm) {
    switch (MI.getOpcode()) {
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
    Count += (AMDGPU::getEventsFor(MI, ST, IsExpertMode, TgSplit) & LoadEvents)
                 .size();
  }
  return Count;
}

} // end anonymous namespace

AMDGPU::CounterPaddingResult AMDGPU::expandCounterPaddingHints(
    MachineFunction &MF, MachineLoopInfo &MLI, MachineDominatorTree *MDT,
    MachinePostDominatorTree &PDT, const HardwareLimits &Limits,
    HWEvents LoadEvents, bool IsExpertMode, bool TgSplit) {
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
  DenseSet<const MachineBasicBlock *> HintedBlocks;
  CounterPaddingResult Result;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB.instrs())) {
      if (MI.getOpcode() != AMDGPU::PAD_WAITCNT)
        continue;
      HintedBlocks.insert(&MBB);
      int64_t Counter = MI.getOperand(0).getImm();
      int64_t Count = MI.getOperand(1).getImm();
      if (Counter != 0) {
        LLVM_DEBUG(printIgnored(MBB, "unsupported counter selector"));
      } else if (Count <= 0 || uint64_t(Count) > Limits.LoadcntMax - 1) {
        LLVM_DEBUG(printIgnored(MBB, "count is out of range"));
      } else if (auto PE = getCounterPaddingEdge(MBB, MLI, TII)) {
        auto [It, Inserted] = Requests.try_emplace(
            PE->PadBlockOrPred,
            PaddingRequest{PE->Join, unsigned(Count), MI.getDebugLoc()});
        if (!Inserted)
          It->second.Count += Count;
      }
      // An ignored hint grants no permission to weaken a wait.
      MI.eraseFromParent();
      Result.Changed = true;
    }
  }

  // Pad the shorter edge of each unbalanced join that asyncmark history
  // guards. An arm with a hint, even an ignored one, is left to the user.
  DenseSet<MachineBasicBlock *> Branches = findAsyncmarkPaddingBranches(MF);
  DenseSet<MachineBasicBlock *> PaddedBranches;
  for (MachineBasicBlock &Join : MF) {
    auto Shape = getAsyncmarkJoinShape(Join, Branches, MLI, TII);
    if (!Shape || !PaddedBranches.insert(Shape->Branch).second)
      continue;

    // The direct edge of a triangle issues no events.
    auto CountFor = [&](MachineBasicBlock *Arm) -> std::optional<unsigned> {
      if (Arm == Shape->Branch)
        return 0;
      if (HintedBlocks.contains(Arm))
        return std::nullopt;
      return countLoadEvents(*Arm, ST, LoadEvents, IsExpertMode, TgSplit);
    };
    std::optional<unsigned> CountA = CountFor(Shape->ArmA);
    std::optional<unsigned> CountB = CountFor(Shape->ArmB);
    if (!CountA || !CountB || *CountA == *CountB)
      continue;

    MachineBasicBlock *Shorter = *CountA < *CountB ? Shape->ArmA : Shape->ArmB;
    unsigned Count =
        std::min(std::max(*CountA, *CountB) - std::min(*CountA, *CountB),
                 Limits.LoadcntMax - 1);
    auto [It, Inserted] =
        Requests.try_emplace(Shorter, PaddingRequest{&Join, Count, DebugLoc()});
    if (!Inserted)
      It->second.Count += Count;
    LLVM_DEBUG(dbgs() << "Asyncmark padding " << printMBBReference(*Shorter)
                      << " -> " << printMBBReference(Join) << " by " << Count
                      << " in " << MF.getName() << '\n');
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
    auto InsertPt = PadBlock->getFirstTerminator();
    for (unsigned I = 0; I != R.Count; ++I)
      BuildMI(*PadBlock, InsertPt, R.DL, TII.get(AMDGPU::BUFFER_INV)).addImm(0);
    Result.Changed = true;
    LLVM_DEBUG(dbgs() << "Padded " << printMBBReference(*PadBlock) << " with "
                      << R.Count << " BUFFER_INV in " << MF.getName() << '\n');
  }
  return Result;
}
