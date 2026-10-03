# LLM_AGENT_RULES (STRICT PROTOCOL)
> This protocol is MANDATORY for all LLM agents. Failure to update memory is a failure of the task.

## 🧩 MANDATORY SKILLS
Every bundled skill is mandatory for every agent. Agents MUST follow ProjectMap, Reflections, ThinkBeforeCoding, Simplicity, SurgicalChanges, GoalDriven, FreshDeps, and ContextBudget together; none are optional preferences.
- **ProjectMap:** Read `llm-code-graph.md` before raw file inspection and use it to pick the smallest useful file set.
- **Reflections:** Read `llm-agent-project-learnings.md` before work and record reusable lessons after failures or non-obvious behavior.
- **ThinkBeforeCoding:** Surface assumptions, ambiguity, tradeoffs, and simpler options before non-trivial work.
- **Simplicity:** Write only what the task requires; no extra abstractions, features, or speculative handling.
- **SurgicalChanges:** Change only the explicitly required files and lines; no unrelated refactors or style churn. Clean up only artifacts introduced by your own change.
- **GoalDriven:** Define verifiable success criteria before implementation and report verification results or blockers.
- **FreshDeps:** Use latest stable compatible dependencies and current APIs. Avoid deprecated packages, methods, functions, flags, and patterns. If an agent repeats a stale or deprecated choice after correction, it MUST stop, re-read these rules, state that FreshDeps is mandatory, and replace the choice with the current stable approach.
- **ContextBudget:** Periodically condense working context into a compact rolling summary after each phase or every 10 tool calls.

## 🎯 GOAL-DRIVEN EXECUTION
For non-trivial tasks, agents should define the goal in verifiable terms, state the smallest plan, and report verification results or blockers before completion.

## 🧠 THE REFLECTION CYCLE
1. **PRE-TASK:** Before planning or making changes, read `llm-agent-project-learnings.md`.
2. **APPLY MEMORY:** Treat every relevant lesson as an active constraint. If a lesson matches the current file, tool, OS, dependency, or failure mode, state how it changes your approach.
3. **EXECUTION:** Monitor for failures, corrections, repeated mistakes, or non-obvious project behavior.
4. **POST-TASK:** Run `code-graph reflect <CAT> <LESSON>` for any new reusable lesson. Do not finish a bug fix, failed-command recovery, or environment workaround without either recording a reflection or explicitly stating that no new reusable lesson was learned.

## 📝 REFLECTION CATEGORIES
- `LOGIC`: Code bugs or complex regex pitfalls.
- `ENV`: OS compatibility or shell behaviors.
- `DEP`: Library bugs or version deprecations.
- `STYLE`: Project-specific architectural rules.

## 🔌 MCP TOOLS (if code-graph MCP server is configured)
Prefer these tools INSTEAD of CLI commands and raw file reads:
- `get_project_graph` — read full `llm-code-graph.md` (INSTEAD of Read)
- `search_symbols` — find symbols across all files (INSTEAD of Grep)
- `trace_dependencies` — map blast radius for a file (INSTEAD of manual EDGES parsing)
- `generate_graph` — refresh the graph (INSTEAD of `code-graph generate` CLI)
- `get_reflections` — read lessons (INSTEAD of Read on `llm-agent-project-learnings.md`)
- `add_reflection` — record a lesson (INSTEAD of `code-graph reflect` CLI)
- `get_file_symbols` — get symbols for one specific file
- `search_graph` — search file paths, symbol names, and descriptions together
