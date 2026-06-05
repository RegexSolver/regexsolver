use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

use crate::error::EngineError;

/// Hold settings about limitations and constraints of operations execution within the engine.
///
/// # Examples:
///
/// ## Limiting the number of states
/// ```
/// use regexsolver::{Term, execution_profile::{ExecutionProfile, ExecutionProfileBuilder}, error::EngineError};
///
/// let term1 = Term::from_pattern(".*abcdef.*").unwrap();
/// let term2 = Term::from_pattern(".*defabc.*").unwrap();
///
/// let execution_profile = ExecutionProfileBuilder::new()
///     .max_number_of_states(5)
///     .build();
///
/// execution_profile.run(|| {
///     assert_eq!(EngineError::AutomatonHasTooManyStates, term1.intersection(&[term2]).unwrap_err());
/// });
/// ```
///
/// ## Limiting the execution time
/// ```
/// use regexsolver::{Term, execution_profile::{ExecutionProfile, ExecutionProfileBuilder}, error::EngineError};
/// use std::time::SystemTime;
///
/// let term = Term::from_pattern(".*abc.*cdef.*sqdsqf.*").unwrap();
///
/// let execution_profile = ExecutionProfileBuilder::new()
///     .execution_timeout(5) // 5ms
///     .build();
///
/// execution_profile.run(|| {
///     assert_eq!(EngineError::OperationTimeOutError, term.generate_strings(1000, 1_000_000).unwrap_err());
/// });
/// ```
///
/// ## Disabling implicit determinization
///
/// [`FastAutomaton`](crate::fast_automaton::FastAutomaton) operations that
/// require a deterministic automaton (`minimize`, `complement`,
/// `difference`, `equivalent`, `subset`, `get_cardinality`, ...)
/// determinize a non-deterministic input on their own by default. Since
/// subset construction can blow up exponentially, this can be disabled;
/// those operations then fail fast and determinization only happens through
/// an explicit `determinize()` call. [`Term`](crate::Term) methods are not
/// affected: that layer manages the underlying representation itself, so
/// its determinizations count as explicit.
///
/// ```
/// use regexsolver::CharRange;
/// use regexsolver::fast_automaton::FastAutomaton;
/// use regexsolver::execution_profile::ExecutionProfileBuilder;
/// use regexsolver::error::EngineError;
///
/// // Two overlapping transitions from the start state: non-deterministic.
/// let mut nfa = FastAutomaton::new_empty();
/// let s1 = nfa.new_state();
/// let s2 = nfa.new_state();
/// nfa.add_transition_from_range(0, s1, &CharRange::total()).unwrap();
/// nfa.add_transition_from_range(0, s2, &CharRange::total()).unwrap();
/// nfa.accept(s1);
///
/// let execution_profile = ExecutionProfileBuilder::new()
///     .implicit_determinization(false)
///     .build();
///
/// execution_profile.run(|| {
///     // `minimize` requires a DFA and refuses to determinize on its own.
///     assert_eq!(
///         EngineError::DeterministicAutomatonRequired,
///         nfa.clone().minimize().unwrap_err()
///     );
///
///     // Determinizing explicitly is always allowed.
///     let mut dfa = nfa.determinize().unwrap().into_owned();
///     assert!(dfa.minimize().is_ok());
/// });
/// ```
#[derive(Clone, Debug)]
pub struct ExecutionProfile {
    /// The maximum number of states that a non-determinitic finite automaton can hold, this is checked during the convertion of regular expression to automaton.
    max_number_of_states: Option<usize>,
    /// The longest time in milliseconds that an operation execution can last, there are no guaranties that the exact time will be respected.
    execution_timeout: Option<u64>,
    /// The time after when a [`EngineError::OperationTimeOutError`] should be thrown.
    execution_deadline: Option<Instant>,
    /// Whether [`FastAutomaton`](crate::fast_automaton::FastAutomaton)
    /// operations that require a deterministic automaton may determinize a
    /// non-deterministic input on their own (the default). When `false`,
    /// those operations return
    /// [`EngineError::DeterministicAutomatonRequired`] instead, so that the
    /// potentially exponential subset construction only ever happens through
    /// an explicit `determinize()` call. [`Term`](crate::Term) methods
    /// always work: that layer manages the representation itself.
    implicit_determinization: bool,
    /// Whether every determinization is followed by a minimization of the
    /// resulting automaton. Off by default: minimization costs an extra
    /// Hopcroft pass, but keeps downstream operations working on the
    /// smallest possible automata.
    minimize_after_determinization: bool,
}

impl PartialEq for ExecutionProfile {
    fn eq(&self, other: &ExecutionProfile) -> bool {
        self.max_number_of_states == other.max_number_of_states
            && self.execution_timeout == other.execution_timeout
            && self.implicit_determinization == other.implicit_determinization
            && self.minimize_after_determinization == other.minimize_after_determinization
    }
}

impl ExecutionProfile {
    /// Retrieve the current thread-local execution profile.
    pub fn get() -> ExecutionProfile {
        ThreadLocalParams::get_execution_profile()
    }

    /// Assert that `execution_timeout` is not exceeded.
    ///
    /// Return empty if `execution_timeout` is not exceeded.
    ///
    /// Return [`EngineError::OperationTimeOutError`] otherwise.
    pub(crate) fn assert_not_timed_out(&self) -> Result<(), EngineError> {
        if let Some(execution_deadline) = self.execution_deadline {
            if Instant::now() > execution_deadline {
                Err(EngineError::OperationTimeOutError)
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    /// Assert that `max_number_of_states` is not exceeded.
    ///
    /// Return empty if `max_number_of_states` is not exceeded.
    ///
    /// Return [`EngineError::AutomatonHasTooManyStates`] otherwise.
    pub(crate) fn assert_max_number_of_states(
        &self,
        number_of_states: usize,
    ) -> Result<(), EngineError> {
        if let Some(max_number_of_states) = self.max_number_of_states
            && number_of_states >= max_number_of_states
        {
            return Err(EngineError::AutomatonHasTooManyStates);
        }
        Ok(())
    }

    /// Assert that implicit determinization is allowed.
    ///
    /// Return empty if it is.
    ///
    /// Return [`EngineError::DeterministicAutomatonRequired`] otherwise.
    pub(crate) fn assert_implicit_determinization_allowed(&self) -> Result<(), EngineError> {
        if self.implicit_determinization {
            Ok(())
        } else {
            Err(EngineError::DeterministicAutomatonRequired)
        }
    }

    pub fn with_execution_timeout(mut self, execution_timeout_in_ms: u64) -> Self {
        self.execution_timeout = Some(execution_timeout_in_ms);
        self
    }

    pub fn with_max_number_of_states(mut self, max_number_of_states: usize) -> Self {
        self.max_number_of_states = Some(max_number_of_states);
        self
    }

    pub fn with_implicit_determinization(mut self, allowed: bool) -> Self {
        self.implicit_determinization = allowed;
        self
    }

    pub fn with_minimize_after_determinization(mut self, enabled: bool) -> Self {
        self.minimize_after_determinization = enabled;
        self
    }

    /// Whether every determinization should be followed by a minimization of
    /// the result.
    pub(crate) fn should_minimize_after_determinization(&self) -> bool {
        self.minimize_after_determinization
    }

    pub fn set(&self) -> &Self {
        self
    }

    /// Run the given closure with this profile at thread level, setting its start time to now.
    pub fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let initial_execution_profile = ThreadLocalParams::get_execution_profile();

        let mut execution_profile = self.clone();
        if let Some(execution_timeout) = execution_profile.execution_timeout {
            execution_profile.execution_deadline =
                Some(Instant::now() + Duration::from_millis(execution_timeout));
        }

        ThreadLocalParams::set_execution_profile(&execution_profile);
        let result = f();
        ThreadLocalParams::set_execution_profile(&initial_execution_profile);
        result
    }

    /// Like [`ExecutionProfile::run`], but does *not* reset its start time. Useful if you want to pass a profile state to a new thread.
    pub fn apply<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let initial_execution_profile = ThreadLocalParams::get_execution_profile();

        ThreadLocalParams::set_execution_profile(self);
        let result = f();
        ThreadLocalParams::set_execution_profile(&initial_execution_profile);
        result
    }
}

pub struct ExecutionProfileBuilder {
    /// The maximum number of states that a non-determinitic finite automaton can hold, this is checked during the convertion of regular expression to automaton.
    max_number_of_states: Option<usize>,
    /// The longest time in milliseconds that an operation execution can last, there are no guaranties that the exact time will be respected.
    execution_timeout: Option<u64>,
    /// Whether operations requiring a deterministic automaton may determinize
    /// a non-deterministic input on their own. Defaults to `true`.
    implicit_determinization: bool,
    /// Whether every determinization is followed by a minimization of the
    /// result. Defaults to `false`.
    minimize_after_determinization: bool,
}
impl Default for ExecutionProfileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionProfileBuilder {
    pub fn new() -> Self {
        Self {
            max_number_of_states: None,
            execution_timeout: None,
            implicit_determinization: true,
            minimize_after_determinization: false,
        }
    }

    pub fn execution_timeout(mut self, execution_timeout_in_ms: u64) -> Self {
        self.execution_timeout = Some(execution_timeout_in_ms);
        self
    }

    pub fn max_number_of_states(mut self, max_number_of_states: usize) -> Self {
        self.max_number_of_states = Some(max_number_of_states);
        self
    }

    /// Whether [`FastAutomaton`](crate::fast_automaton::FastAutomaton)
    /// operations that require a deterministic automaton may determinize a
    /// non-deterministic input on their own (the default). When set to
    /// `false`, those operations return
    /// [`EngineError::DeterministicAutomatonRequired`] instead; explicit
    /// `determinize()` calls — and [`Term`](crate::Term) methods, which
    /// manage the representation themselves — are always allowed.
    pub fn implicit_determinization(mut self, allowed: bool) -> Self {
        self.implicit_determinization = allowed;
        self
    }

    /// Whether every determinization is followed by a minimization of the
    /// resulting automaton. Off by default: minimization costs an extra
    /// Hopcroft pass, but keeps downstream operations working on the
    /// smallest possible automata. Inputs that are already deterministic are
    /// not touched.
    pub fn minimize_after_determinization(mut self, enabled: bool) -> Self {
        self.minimize_after_determinization = enabled;
        self
    }

    pub fn build(self) -> ExecutionProfile {
        ExecutionProfile {
            max_number_of_states: self.max_number_of_states,
            execution_timeout: self.execution_timeout,
            execution_deadline: None,
            implicit_determinization: self.implicit_determinization,
            minimize_after_determinization: self.minimize_after_determinization,
        }
    }
}

struct ThreadLocalParams;
impl ThreadLocalParams {
    thread_local! {
        static MAX_NUMBER_OF_STATES: RefCell<Option<usize>> = const { RefCell::new(None) };
        static EXECUTION_DEADLINE: RefCell<Option<Instant>> = const { RefCell::new(None) };
        static EXECUTION_TIMEOUT: RefCell<Option<u64>> = const { RefCell::new(None) };
        static IMPLICIT_DETERMINIZATION: RefCell<bool> = const { RefCell::new(true) };
        static MINIMIZE_AFTER_DETERMINIZATION: RefCell<bool> = const { RefCell::new(false) };
    }

    /// Store on the current thread [`ExecutionProfile`].
    fn set_execution_profile(profile: &ExecutionProfile) {
        ThreadLocalParams::MAX_NUMBER_OF_STATES.with(|cell| {
            *cell.borrow_mut() = profile.max_number_of_states;
        });

        ThreadLocalParams::EXECUTION_DEADLINE.with(|cell| {
            *cell.borrow_mut() = profile.execution_deadline;
        });

        ThreadLocalParams::EXECUTION_TIMEOUT.with(|cell| {
            *cell.borrow_mut() = profile.execution_timeout;
        });

        ThreadLocalParams::IMPLICIT_DETERMINIZATION.with(|cell| {
            *cell.borrow_mut() = profile.implicit_determinization;
        });

        ThreadLocalParams::MINIMIZE_AFTER_DETERMINIZATION.with(|cell| {
            *cell.borrow_mut() = profile.minimize_after_determinization;
        });
    }

    fn get_max_number_of_states() -> Option<usize> {
        ThreadLocalParams::MAX_NUMBER_OF_STATES.with(|cell| *cell.borrow())
    }

    fn get_execution_deadline() -> Option<Instant> {
        ThreadLocalParams::EXECUTION_DEADLINE.with(|cell| *cell.borrow())
    }

    fn get_execution_timeout() -> Option<u64> {
        ThreadLocalParams::EXECUTION_TIMEOUT.with(|cell| *cell.borrow())
    }

    fn get_implicit_determinization() -> bool {
        ThreadLocalParams::IMPLICIT_DETERMINIZATION.with(|cell| *cell.borrow())
    }

    fn get_minimize_after_determinization() -> bool {
        ThreadLocalParams::MINIMIZE_AFTER_DETERMINIZATION.with(|cell| *cell.borrow())
    }

    /// Return the [`ExecutionProfile`] stored on the current thread.
    fn get_execution_profile() -> ExecutionProfile {
        ExecutionProfile {
            max_number_of_states: Self::get_max_number_of_states(),
            execution_deadline: Self::get_execution_deadline(),
            execution_timeout: Self::get_execution_timeout(),
            implicit_determinization: Self::get_implicit_determinization(),
            minimize_after_determinization: Self::get_minimize_after_determinization(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Term, regex::RegularExpression};

    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_traits() -> Result<(), String> {
        assert_send::<ExecutionProfile>();
        assert_sync::<ExecutionProfile>();

        Ok(())
    }

    #[test]
    fn test_execution_get() -> Result<(), String> {
        let execution_profile = ExecutionProfileBuilder::new()
            .execution_timeout(1000)
            .max_number_of_states(8192)
            .build();

        execution_profile.run(|| {
            assert_eq!(execution_profile, ExecutionProfile::get());
        });

        Ok(())
    }

    #[test]
    fn test_execution() -> Result<(), String> {
        ExecutionProfileBuilder::new()
            .max_number_of_states(1)
            .build()
            .run(|| {
                let regex = RegularExpression::new("test").unwrap();

                assert!(regex.to_automaton().is_err());
                assert_eq!(
                    EngineError::AutomatonHasTooManyStates,
                    regex.to_automaton().unwrap_err()
                );
            });

        Ok(())
    }

    /// A two-way acyclic automaton with overlapping transitions: the
    /// smallest shape that is non-deterministic and reaches the
    /// determinization paths of every DFA-requiring operation.
    fn nondeterministic_automaton() -> crate::fast_automaton::FastAutomaton {
        use crate::fast_automaton::FastAutomaton;
        use crate::fast_automaton::condition::Condition;

        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let cond = Condition::total(a.get_spanning_set());
        a.add_transition(0, s1, &cond);
        a.add_transition(0, s2, &cond);
        a.accept(s1);
        a.accept(s2);
        assert!(!a.is_deterministic());
        a
    }

    #[test]
    fn test_implicit_determinization_disabled() {
        let nfa = nondeterministic_automaton();
        let dfa = nfa.determinize().unwrap().into_owned();

        ExecutionProfileBuilder::new()
            .implicit_determinization(false)
            .build()
            .run(|| {
                let err = EngineError::DeterministicAutomatonRequired;

                // Every DFA-requiring operation refuses to determinize a
                // non-deterministic input on its own...
                assert_eq!(nfa.clone().minimize().unwrap_err(), err);
                assert_eq!(nfa.clone().complement().unwrap_err(), err);
                assert_eq!(dfa.difference(&nfa).unwrap_err(), err);
                assert_eq!(nfa.equivalent(&dfa).unwrap_err(), err);
                assert_eq!(dfa.subset(&nfa).unwrap_err(), err);
                assert_eq!(nfa.get_cardinality().unwrap_err(), err);

                // ...but operations that work on NFAs directly are unaffected
                // (difference only determinizes the subtrahend)...
                assert!(nfa.difference(&dfa).is_ok());

                // ...deterministic inputs keep working...
                assert!(dfa.clone().minimize().is_ok());
                assert!(dfa.clone().complement().is_ok());
                assert!(dfa.get_cardinality().is_ok());
                assert!(dfa.equivalent(&dfa).is_ok());

                // ...and explicit determinization is always allowed.
                assert!(nfa.determinize().is_ok());
            });
    }

    #[test]
    fn test_minimize_after_determinization() {
        use crate::CharRange;
        use crate::fast_automaton::FastAutomaton;
        use crate::fast_automaton::condition::Condition;
        use crate::fast_automaton::spanning_set::SpanningSet;
        use regex_charclass::char::Char;

        // NFA over base 'a' + rest whose subset construction yields two
        // distinct but language-equivalent accept states ({f1, f3} on 'a',
        // {f2} on [^a]) — 3 determinized states, 2 after minimization.
        let range_a = CharRange::new_from_range(Char::new('a')..=Char::new('a'));
        let ss = SpanningSet::compute_spanning_set(std::slice::from_ref(&range_a));
        let mut nfa = FastAutomaton::new_empty();
        nfa.apply_new_spanning_set(&ss).unwrap();
        let f1 = nfa.new_state();
        let f2 = nfa.new_state();
        let f3 = nfa.new_state();
        let cond_a = Condition::from_range(&range_a, &ss).unwrap();
        let cond_rest = cond_a.complement();
        nfa.add_transition(0, f1, &cond_a);
        nfa.add_transition(0, f3, &cond_a); // overlaps with f1: non-deterministic
        nfa.add_transition(0, f2, &cond_rest);
        nfa.accept(f1);
        nfa.accept(f2);
        nfa.accept(f3);
        assert!(!nfa.is_deterministic());

        // Default: determinize alone does not minimize.
        let plain = nfa.determinize().unwrap().into_owned();
        assert!(plain.is_deterministic());
        assert!(!plain.is_minimal());
        assert_eq!(plain.get_number_of_states(), 3);

        ExecutionProfileBuilder::new()
            .minimize_after_determinization(true)
            .build()
            .run(|| {
                let minimized = nfa.determinize().unwrap().into_owned();
                assert!(minimized.is_deterministic());
                assert!(minimized.is_minimal());
                assert_eq!(minimized.get_number_of_states(), 2);
                assert!(minimized.equivalent(&plain).unwrap());

                // Already-deterministic inputs are returned untouched: the
                // flag only applies when a determinization actually happens.
                let same = plain.determinize().unwrap();
                assert!(!same.is_minimal());
            });
    }

    /// The `implicit_determinization` knob targets direct `FastAutomaton`
    /// usage; `Term` manages the underlying representation itself, so its
    /// whole public API must keep working when the knob is off.
    #[test]
    fn test_term_api_works_without_implicit_determinization() {
        let term = Term::from_automaton(nondeterministic_automaton());
        let other = Term::from_pattern("a*").unwrap();

        ExecutionProfileBuilder::new()
            .implicit_determinization(false)
            .build()
            .run(|| {
                // Methods that need a DFA internally determinize on Term's
                // behalf (an explicit choice of the Term layer)...
                assert!(term.difference(&other).is_ok());
                assert!(other.difference(&term).is_ok());
                assert!(term.complement().is_ok());
                assert!(term.equivalent(&other).is_ok());
                assert!(term.subset(&other).is_ok());
                assert!(other.subset(&term).is_ok());
                assert!(term.is_total().is_ok());
                assert!(term.get_cardinality().is_ok());
                assert!(term.minimize().is_ok());
                assert!(term.generate_strings(5, 0).is_ok());

                // ...and the rest of the API never needed one.
                assert!(term.concat(std::slice::from_ref(&other)).is_ok());
                assert!(term.union(std::slice::from_ref(&other)).is_ok());
                assert!(term.intersection(std::slice::from_ref(&other)).is_ok());
                assert!(term.repeat(0, Some(2)).is_ok());
                assert!(term.is_empty().is_ok());
                assert!(term.is_empty_string().is_ok());
                let _ = term.get_length();
                let _ = term.to_regex();
                let _ = term.to_pattern();
                assert!(term.to_automaton().is_ok());

                // The override is scoped: direct FastAutomaton usage stays
                // gated afterwards.
                assert_eq!(
                    nondeterministic_automaton().minimize().unwrap_err(),
                    EngineError::DeterministicAutomatonRequired
                );
            });
    }

    /// The two determinization knobs compose: implicit determinization
    /// stays gated, while an explicit `determinize()` both works and
    /// minimizes its result.
    #[test]
    fn test_minimize_after_determinization_with_implicit_disabled() {
        let nfa = nondeterministic_automaton();

        ExecutionProfileBuilder::new()
            .implicit_determinization(false)
            .minimize_after_determinization(true)
            .build()
            .run(|| {
                assert_eq!(
                    nfa.clone().minimize().unwrap_err(),
                    EngineError::DeterministicAutomatonRequired
                );

                let dfa = nfa.determinize().unwrap();
                assert!(dfa.is_deterministic());
                assert!(dfa.is_minimal());
                assert!(dfa.equivalent(&nfa.determinize().unwrap()).unwrap());
            });
    }

    #[test]
    fn test_implicit_determinization_default() {
        let nfa = nondeterministic_automaton();

        // Without the profile knob the historical behavior is unchanged.
        assert!(nfa.clone().minimize().is_ok());
        assert!(nfa.clone().complement().is_ok());
        assert!(nfa.get_cardinality().is_ok());
        assert!(nfa.equivalent(&nfa.clone()).is_ok());
    }

    #[test]
    fn test_execution_timeout_generate_strings() -> Result<(), String> {
        let term = Term::from_pattern(".*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz").unwrap();

        let execution_timeout_in_ms = 10;
        let start_time = Instant::now();
        ExecutionProfileBuilder::new()
            .execution_timeout(execution_timeout_in_ms)
            .build()
            .run(|| {
                assert_eq!(
                    EngineError::OperationTimeOutError,
                    term.generate_strings(100, 1_000_000).unwrap_err()
                );

                let run_duration = Instant::now().duration_since(start_time).as_millis();

                println!("{run_duration}");
                assert!(run_duration <= (execution_timeout_in_ms + 50) as u128);
            });

        Ok(())
    }

    #[test]
    fn test_execution_timeout_difference() -> Result<(), String> {
        let term1 = Term::from_pattern(".*abc.*def.*qdqd.*qsdsqdsqdz").unwrap();
        let term2 = Term::from_pattern(".*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz.*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz.*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz").unwrap();

        let execution_timeout_in_ms = 10;
        let start_time = Instant::now();
        ExecutionProfileBuilder::new()
            .execution_timeout(execution_timeout_in_ms)
            .build()
            .run(|| {
                assert_eq!(
                    EngineError::OperationTimeOutError,
                    term1.difference(&term2).unwrap_err()
                );

                let run_duration = Instant::now().duration_since(start_time).as_millis();

                println!("{run_duration}");
                assert!(run_duration <= (execution_timeout_in_ms + 25) as u128);
            });

        Ok(())
    }

    /*#[test]
    fn test_execution_timeout_intersection() -> Result<(), String> {
        let term1 = Term::from_pattern(".*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz").unwrap();
        let term2 = Term::from_pattern(".*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz.*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz.*abc.*def.*qdqd.*qsdsqdsqdz").unwrap();

        let execution_timeout_in_ms = 100;
        let start_time = SystemTime::now();
        ExecutionProfileBuilder::new()
            .execution_timeout(execution_timeout_in_ms)
            .build()
            .run(|| {
                assert_eq!(
                    EngineError::OperationTimeOutError,
                    term1.intersection(&[term2]).unwrap_err()
                );

                let run_duration = SystemTime::now()
                    .duration_since(start_time)
                    .expect("Time went backwards")
                    .as_millis();

                println!("{run_duration}");
                assert!(run_duration <= execution_timeout_in_ms + 100);
            });

        Ok(())
    }*/
}
