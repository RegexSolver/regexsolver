# Security Policy

## Supported versions

Fixes are published for the latest release of `regexsolver`.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's [security advisory form](https://github.com/RegexSolver/regexsolver/security/advisories/new) rather than in a public issue.

## Scope

`regexsolver` contains no `unsafe` code, enforced by `#![forbid(unsafe_code)]`, and performs no I/O. It is designed to run untrusted patterns, so the realistic concerns are:

- **An operation that escapes its budget.** Determinization is exponential in the worst case and automaton-to-regex conversion can grow super-polynomially, so the engine is meant to be bounded rather than fast on every input: an `ExecutionProfile` caps wall-clock time and state count, and `implicit_determinization(false)` refuses to determinize implicitly at all. An input that runs far past `execution_timeout`, allocates without bound below `max_number_of_states`, or does not terminate, is a bug worth reporting.
- **A panic, an unbounded recursion, or a stack overflow** reachable from a pattern, from a `FastAutomaton` built through the public API, or from a hand-built `RegularExpression` tree. Deeply nested trees are the known hazard: `RegularExpression::MAX_NESTING_DEPTH` bounds them and `to_automaton` returns `EngineError::RegexTooDeeplyNested` rather than overflowing the stack.
- **A wrong answer from a set operation.** These results are used to make authorization and validation decisions, so a `subset` that returns `true` for a language it does not contain, or a `to_pattern` whose output denotes a different set of strings than the term it came from, is a correctness bug with a security consequence.

`Term::matches` is the only entry point that runs a pattern against a string. It simulates the automaton over the whole set of reachable states rather than backtracking, so its cost is bounded by the input length times the size of the automaton, and catastrophic backtracking does not apply to it. The cost is in building the automaton, which is what `ExecutionProfile` bounds.
