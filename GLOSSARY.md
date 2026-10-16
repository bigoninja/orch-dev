# Agent Dev Orchestrator

A LangGraph orchestrator that drives OpenCode sessions to implement software tasks in
parallel: dependency-aware scheduling, per-task review, and a single integration branch
for human promotion. First target repo: shopdash.

## Artifacts

**Plan**:
The task DAG written as `plan.yaml` by the Planner from a feature brief; the human approves it before execution begins.
_Avoid_: roadmap, phase list, backlog

**Task**:
A bounded unit of work: id, dependencies, files it may modify, acceptance criteria, and Checks.
_Avoid_: job, ticket, workstream

**Repair task**:
A Task spawned automatically from an integration failure that is clearly local to one Task; capped at one per failure event, and reviewed like any other Task.
_Avoid_: fix task, hotfix, patch

**Checks**:
The explicit shell commands (test / lint / typecheck) that produce a Task's deterministic pass/fail evidence.
_Avoid_: validation, quality gates

## Task lifecycle

**Approved**:
A Task whose diff and evidence an independent Reviewer has accepted.
_Avoid_: passed, complete, done

**Integrated**:
An Approved Task whose commits are merged into the Integration branch and whose post-merge Checks passed.
_Avoid_: merged, landed, shipped

**Ready**:
A Task whose dependencies are all Integrated.
_Avoid_: unblocked, available, enabled

## Roles

**Planner**: Turns a feature brief and the Target repo into a Plan.

**Worker**: Implements exactly one Task inside its own worktree.
_Avoid_: implementer, executor, subagent, coding agent

**Reviewer**: Independently judges one Task's diff and evidence in a fresh session.

## Concepts

**Session**:
One OpenCode conversation answering one role-step; Workers and Reviewers must never share one (a fresh session is what makes review independent).
_Avoid_: thread, chat, context

**Run**:
One execution of the orchestrator over a Plan — its own run-id, artifacts directory, and checkpoints.
_Avoid_: job, execution, attempt

**Target repo**:
The repository the orchestrator operates on — initially shopdash.
_Avoid_: product, project, codebase

**Integration branch**:
The branch where Approved Tasks are merged automatically; promoting it to the Target repo's main branch is always a human action.
_Avoid_: master, main, trunk

**Escalation**:
The orchestrator stopping to request a human decision — retries exhausted, integration failed, or a risky change detected.
_Avoid_: failure, crash, abort
