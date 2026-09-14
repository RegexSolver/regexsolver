use std::{cell::RefCell, time::Duration};
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use std::{sync::OnceLock, time::Instant};

use crate::error::EngineError;

/// A monotonic clock: the time elapsed since a fixed but otherwise arbitrary
/// origin, such as process start or page load.
///
/// The execution timeout is enforced by comparing readings of the profile's
/// clock. Every profile starts with a clock backed by [`std::time::Instant`],
/// except on `wasm32-unknown-unknown`, where the standard library cannot read
/// the time and the host has to supply one through
/// [`ExecutionProfileBuilder::clock`].
pub type Clock = fn() -> Duration;

/// [`std::time::Instant`], read as the time elapsed since the first reading.
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
fn std_clock() -> Duration {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed()
}

/// `wasm32-unknown-unknown` has no clock the standard library can read
/// (`Instant::now` panics there), so profiles start without one and a timeout
/// needs a host-provided [`Clock`].
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
const DEFAULT_CLOCK: Option<Clock> = Some(std_clock);
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
const DEFAULT_CLOCK: Option<Clock> = None;

/// Holds settings that constrain how operations execute within the engine.
///
/// # Examples
///
/// ## Limiting the number of states
/// ```
/// use regexsolver::{Term, execution_profile::ExecutionProfileBuilder, error::EngineError};
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
/// use regexsolver::{Term, execution_profile::ExecutionProfileBuilder, error::EngineError, fast_automaton::GenerationOptions};
///
/// let term = Term::from_pattern(".*abc.*cdef.*sqdsqf.*").unwrap();
///
/// let execution_profile = ExecutionProfileBuilder::new()
///     .execution_timeout(5) // 5ms
///     .build();
///
/// execution_profile.run(|| {
///     assert_eq!(EngineError::OperationTimeOutError, term.generate_strings(100_000_000, 0, GenerationOptions::new()).unwrap_err());
/// });
/// ```
///
/// ## Supplying the clock
///
/// The timeout compares readings of a monotonic [`Clock`]. The default reads
/// [`std::time::Instant`]; on `wasm32-unknown-unknown`, where the standard
/// library has no clock, pass one from the host (a binding to
/// `performance.now()`, which is monotonic; `Date.now()` is not). Any monotonic
/// `fn() -> Duration` works, which also makes a timeout testable without
/// waiting for it:
///
/// ```
/// use std::sync::atomic::{AtomicU64, Ordering};
/// use std::time::Duration;
/// use regexsolver::{Term, execution_profile::ExecutionProfileBuilder, error::EngineError, fast_automaton::GenerationOptions};
///
/// // Advances one millisecond per reading, so the deadline is hit after a
/// // fixed number of checks rather than after real time has passed.
/// static READINGS: AtomicU64 = AtomicU64::new(0);
/// fn ticking_clock() -> Duration {
///     Duration::from_millis(READINGS.fetch_add(1, Ordering::Relaxed))
/// }
///
/// let term = Term::from_pattern(".*abc.*cdef.*sqdsqf.*").unwrap();
///
/// let execution_profile = ExecutionProfileBuilder::new()
///     .execution_timeout(5)
///     .clock(ticking_clock)
///     .build();
///
/// execution_profile.run(|| {
///     assert_eq!(EngineError::OperationTimeOutError, term.generate_strings(100_000_000, 0, GenerationOptions::new()).unwrap_err());
/// });
/// ```
///
/// ## Disabling implicit determinization
///
/// [`FastAutomaton`](crate::fast_automaton::FastAutomaton) operations that
/// require a deterministic automaton (`minimize`, `complement`,
/// `difference`, `equivalent`, `subset`, `cardinality`, ...)
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
    /// The maximum number of states a non-deterministic finite automaton may
    /// hold, checked while converting a regular expression to an automaton.
    max_number_of_states: Option<usize>,
    /// The longest an operation may run, in milliseconds. It is checked
    /// between steps, so the exact time is not guaranteed.
    execution_timeout: Option<u64>,
    /// The [`clock`](ExecutionProfileBuilder::clock) reading past which
    /// [`EngineError::OperationTimeOutError`] is returned. Set by
    /// [`run`](Self::run), and only when a clock is available.
    execution_deadline: Option<Duration>,
    /// The clock the timeout is measured against; `None` on a target without
    /// one, until the host supplies it.
    clock: Option<Clock>,
    /// Whether [`FastAutomaton`](crate::fast_automaton::FastAutomaton)
    /// operations that require a deterministic automaton may determinize a
    /// non-deterministic input on their own (the default). When `false`,
    /// those operations return
    /// [`EngineError::DeterministicAutomatonRequired`] instead, so that the
    /// potentially exponential subset construction only ever happens through
    /// an explicit `determinize()` call. [`Term`](crate::Term) methods
    /// always work: that layer manages the representation itself.
    implicit_determinization: bool,
}

/// Equality compares the *configuration* (state limit, timeout, implicit
/// determinization) and deliberately ignores `execution_deadline` and the
/// clock: two profiles built alike compare equal whether or not one is
/// currently installed and running, and whichever clock they read.
impl PartialEq for ExecutionProfile {
    fn eq(&self, other: &ExecutionProfile) -> bool {
        self.max_number_of_states == other.max_number_of_states
            && self.execution_timeout == other.execution_timeout
            && self.implicit_determinization == other.implicit_determinization
    }
}

impl ExecutionProfile {
    /// Retrieves the current thread-local execution profile.
    pub fn get() -> ExecutionProfile {
        ThreadLocalParams::get_execution_profile()
    }

    /// Whether a execution deadline is configured. When it is not,
    /// [`assert_not_timed_out`](Self::assert_not_timed_out) can be
    /// skipped entirely instead of being computed for a check that
    /// cannot fail.
    #[inline]
    pub fn limits_execution_time(&self) -> bool {
        self.execution_deadline.is_some()
    }

    /// Assert that `execution_timeout` is not exceeded.
    ///
    /// Return empty if `execution_timeout` is not exceeded.
    ///
    /// Return [`EngineError::OperationTimeOutError`] otherwise.
    pub fn assert_not_timed_out(&self) -> Result<(), EngineError> {
        if let (Some(execution_deadline), Some(clock)) = (self.execution_deadline, self.clock)
            && clock() > execution_deadline
        {
            return Err(EngineError::OperationTimeOutError);
        }
        Ok(())
    }

    /// Whether a maximum number of states is configured. When it is not, the
    /// state-count heuristics feeding
    /// [`assert_max_number_of_states`](Self::assert_max_number_of_states) can
    /// be skipped entirely instead of being computed for a check that cannot
    /// fail.
    #[inline]
    pub fn limits_number_of_states(&self) -> bool {
        self.max_number_of_states.is_some()
    }

    /// Assert that `max_number_of_states` is not exceeded.
    ///
    /// `max_number_of_states` is the largest number of states an automaton may
    /// hold, so `number_of_states == max_number_of_states` is allowed and only
    /// strictly exceeding it returns [`EngineError::AutomatonHasTooManyStates`].
    pub fn assert_max_number_of_states(&self, number_of_states: usize) -> Result<(), EngineError> {
        if let Some(max_number_of_states) = self.max_number_of_states
            && number_of_states > max_number_of_states
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
    pub fn assert_implicit_determinization_allowed(&self) -> Result<(), EngineError> {
        if self.implicit_determinization {
            Ok(())
        } else {
            Err(EngineError::DeterministicAutomatonRequired)
        }
    }

    /// Returns a copy of this profile with the execution timeout set to
    /// `execution_timeout_in_ms` milliseconds. Use these `with_*` methods to
    /// derive a variant of an existing profile (e.g. one from
    /// [`get`](Self::get)); to build one from scratch, prefer
    /// [`ExecutionProfileBuilder`]. See
    /// [`ExecutionProfileBuilder::execution_timeout`].
    pub fn with_execution_timeout(mut self, execution_timeout_in_ms: u64) -> Self {
        self.execution_timeout = Some(execution_timeout_in_ms);
        self
    }

    /// Returns a copy of this profile with the maximum number of states set to
    /// `max_number_of_states`. See
    /// [`ExecutionProfileBuilder::max_number_of_states`].
    pub fn with_max_number_of_states(mut self, max_number_of_states: usize) -> Self {
        self.max_number_of_states = Some(max_number_of_states);
        self
    }

    /// Returns a copy of this profile with implicit determinization enabled or
    /// disabled. See [`ExecutionProfileBuilder::implicit_determinization`].
    pub fn with_implicit_determinization(mut self, allowed: bool) -> Self {
        self.implicit_determinization = allowed;
        self
    }

    /// Returns a copy of this profile reading the time from `clock`. See
    /// [`ExecutionProfileBuilder::clock`].
    ///
    /// A deadline already computed by [`run`](Self::run) is dropped, since it
    /// was measured against the previous clock and the two origins are
    /// unrelated. The next `run` computes one against `clock`.
    pub fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = Some(clock);
        self.execution_deadline = None;
        self
    }

    /// Runs the given closure with this profile installed for the current thread, setting its start time to now.
    ///
    /// # Panics
    ///
    /// If an execution timeout is set but the profile has no [`Clock`]. That
    /// only happens where the standard library cannot read the time and none
    /// is installed by default; supply one with [`ExecutionProfileBuilder::clock`]
    /// or [`with_clock`](Self::with_clock).
    pub fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _guard = ProfileRestoreGuard::install();

        let mut execution_profile = self.clone();
        if let Some(execution_timeout) = execution_profile.execution_timeout {
            let clock = execution_profile.clock.expect(
                "an execution timeout is set but the profile has no clock: this target cannot \
                 read the time through the standard library, so supply one with \
                 `ExecutionProfileBuilder::clock` (for instance a binding to `performance.now()`)",
            );
            // A timeout so large the deadline is unrepresentable is equivalent
            // to no deadline at all.
            execution_profile.execution_deadline =
                clock().checked_add(Duration::from_millis(execution_timeout));
        }

        ThreadLocalParams::set_execution_profile(&execution_profile);
        f()
    }

    /// Runs the closure like [`run`](Self::run), but does not reset the start time. Use this to propagate an already-started profile to worker threads without restarting the clock.
    pub fn apply<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _guard = ProfileRestoreGuard::install();

        ThreadLocalParams::set_execution_profile(self);
        f()
    }
}

/// Restores the thread-local execution profile captured at construction when
/// dropped, including on panic unwind. Keeps [`ExecutionProfile::run`] and
/// [`ExecutionProfile::apply`] panic-safe so a panicking closure cannot leak a
/// temporary profile onto a (possibly pooled) thread.
struct ProfileRestoreGuard {
    previous: ExecutionProfile,
}

impl ProfileRestoreGuard {
    fn install() -> Self {
        ProfileRestoreGuard {
            previous: ThreadLocalParams::get_execution_profile(),
        }
    }
}

impl Drop for ProfileRestoreGuard {
    fn drop(&mut self) {
        ThreadLocalParams::set_execution_profile(&self.previous);
    }
}

/// Builder for an [`ExecutionProfile`]. Start from [`new`](Self::new), set the
/// limits you want, and [`build`](Self::build) the immutable profile.
#[derive(Clone, Debug)]
pub struct ExecutionProfileBuilder {
    /// The maximum number of states a non-deterministic finite automaton may
    /// hold, checked while converting a regular expression to an automaton.
    max_number_of_states: Option<usize>,
    /// The longest an operation may run, in milliseconds. It is checked
    /// between steps, so the exact time is not guaranteed.
    execution_timeout: Option<u64>,
    /// Whether operations requiring a deterministic automaton may determinize
    /// a non-deterministic input on their own. Defaults to `true`.
    implicit_determinization: bool,
    /// The clock the timeout is measured against. Defaults to the standard
    /// library's, where it has one.
    clock: Option<Clock>,
}
impl Default for ExecutionProfileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionProfileBuilder {
    /// Creates a builder with no limits set and implicit determinization
    /// enabled (i.e. the defaults, equivalent to the ambient profile when none
    /// has been installed).
    pub fn new() -> Self {
        Self {
            max_number_of_states: None,
            execution_timeout: None,
            implicit_determinization: true,
            clock: DEFAULT_CLOCK,
        }
    }

    /// Sets the longest time, in milliseconds, that an operation may run before
    /// it aborts with [`EngineError::OperationTimeOutError`]. Enforcement is
    /// best-effort (checked between internal steps), so the exact deadline is
    /// not guaranteed. Unset by default (no timeout). The time is read from
    /// the profile's [`clock`](Self::clock).
    pub fn execution_timeout(mut self, execution_timeout_in_ms: u64) -> Self {
        self.execution_timeout = Some(execution_timeout_in_ms);
        self
    }

    /// Sets the [`Clock`] the execution timeout is measured against: any
    /// monotonic `fn() -> Duration`, compared by difference only.
    ///
    /// Defaults to [`std::time::Instant`], so this is only required where the
    /// standard library cannot read the time: bind `performance.now()` from the
    /// host and pass it here. A wall clock such as `Date.now()` does not
    /// qualify: it steps when the system time is adjusted, which either times
    /// an operation out early or never. Elsewhere it is a way to make the
    /// timeout deterministic, as in the
    /// [type-level example](ExecutionProfile#supplying-the-clock).
    pub fn clock(mut self, clock: Clock) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Caps the number of states an automaton may reach; operations that would
    /// exceed it abort with [`EngineError::AutomatonHasTooManyStates`]. This
    /// bounds the exponential blow-up of conversions such as determinization.
    /// Unset by default (no cap).
    pub fn max_number_of_states(mut self, max_number_of_states: usize) -> Self {
        self.max_number_of_states = Some(max_number_of_states);
        self
    }

    /// Whether [`FastAutomaton`](crate::fast_automaton::FastAutomaton)
    /// operations that require a deterministic automaton may determinize a
    /// non-deterministic input on their own (the default). When set to
    /// `false`, those operations return
    /// [`EngineError::DeterministicAutomatonRequired`] instead; explicit
    /// `determinize()` calls and [`Term`](crate::Term) methods (which
    /// manage the representation themselves) are always allowed.
    pub fn implicit_determinization(mut self, allowed: bool) -> Self {
        self.implicit_determinization = allowed;
        self
    }

    /// Builds the [`ExecutionProfile`]. Install it around a unit of work with
    /// [`ExecutionProfile::run`].
    pub fn build(self) -> ExecutionProfile {
        ExecutionProfile {
            max_number_of_states: self.max_number_of_states,
            execution_timeout: self.execution_timeout,
            execution_deadline: None,
            implicit_determinization: self.implicit_determinization,
            clock: self.clock,
        }
    }
}

struct ThreadLocalParams;
impl ThreadLocalParams {
    thread_local! {
        static MAX_NUMBER_OF_STATES: RefCell<Option<usize>> = const { RefCell::new(None) };
        static EXECUTION_DEADLINE: RefCell<Option<Duration>> = const { RefCell::new(None) };
        static EXECUTION_TIMEOUT: RefCell<Option<u64>> = const { RefCell::new(None) };
        static IMPLICIT_DETERMINIZATION: RefCell<bool> = const { RefCell::new(true) };
        static CLOCK: RefCell<Option<Clock>> = const { RefCell::new(DEFAULT_CLOCK) };
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

        ThreadLocalParams::CLOCK.with(|cell| {
            *cell.borrow_mut() = profile.clock;
        });
    }

    fn get_max_number_of_states() -> Option<usize> {
        ThreadLocalParams::MAX_NUMBER_OF_STATES.with(|cell| *cell.borrow())
    }

    fn get_execution_deadline() -> Option<Duration> {
        ThreadLocalParams::EXECUTION_DEADLINE.with(|cell| *cell.borrow())
    }

    fn get_clock() -> Option<Clock> {
        ThreadLocalParams::CLOCK.with(|cell| *cell.borrow())
    }

    fn get_execution_timeout() -> Option<u64> {
        ThreadLocalParams::EXECUTION_TIMEOUT.with(|cell| *cell.borrow())
    }

    fn get_implicit_determinization() -> bool {
        ThreadLocalParams::IMPLICIT_DETERMINIZATION.with(|cell| *cell.borrow())
    }

    /// Return the [`ExecutionProfile`] stored on the current thread.
    fn get_execution_profile() -> ExecutionProfile {
        ExecutionProfile {
            max_number_of_states: Self::get_max_number_of_states(),
            execution_deadline: Self::get_execution_deadline(),
            execution_timeout: Self::get_execution_timeout(),
            implicit_determinization: Self::get_implicit_determinization(),
            clock: Self::get_clock(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        sync::atomic::{AtomicU64, Ordering},
        time::Instant,
    };

    use crate::{Term, fast_automaton::GenerationOptions, regex::RegularExpression};

    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    thread_local! {
        static FAKE_NOW: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    }

    /// A clock the test moves by hand.
    fn fake_clock() -> Duration {
        FAKE_NOW.get()
    }

    /// A clock that advances one millisecond per reading.
    fn ticking_clock() -> Duration {
        static READINGS: AtomicU64 = AtomicU64::new(0);
        Duration::from_millis(READINGS.fetch_add(1, Ordering::Relaxed))
    }

    #[test]
    fn timeout_is_measured_against_the_profile_clock() {
        FAKE_NOW.set(Duration::from_millis(1_000));
        ExecutionProfileBuilder::new()
            .execution_timeout(10)
            .clock(fake_clock)
            .build()
            .run(|| {
                let profile = ExecutionProfile::get();
                assert!(profile.limits_execution_time());
                assert!(profile.assert_not_timed_out().is_ok());

                FAKE_NOW.set(Duration::from_millis(1_010));
                assert!(
                    profile.assert_not_timed_out().is_ok(),
                    "the deadline itself is allowed"
                );

                FAKE_NOW.set(Duration::from_millis(1_011));
                assert_eq!(
                    profile.assert_not_timed_out().unwrap_err(),
                    EngineError::OperationTimeOutError
                );
            });
    }

    #[test]
    fn operation_times_out_on_an_injected_clock() {
        let term = Term::from_pattern(".*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz").unwrap();
        ExecutionProfileBuilder::new()
            .execution_timeout(5)
            .clock(ticking_clock)
            .build()
            .run(|| {
                assert_eq!(
                    EngineError::OperationTimeOutError,
                    term.generate_strings(100_000_000, 0, GenerationOptions::new())
                        .unwrap_err()
                );
            });
    }

    #[test]
    fn apply_keeps_the_running_deadline() {
        FAKE_NOW.set(Duration::ZERO);
        ExecutionProfileBuilder::new()
            .execution_timeout(10)
            .clock(fake_clock)
            .build()
            .run(|| {
                let running = ExecutionProfile::get();
                FAKE_NOW.set(Duration::from_millis(20));
                running.apply(|| {
                    assert_eq!(
                        ExecutionProfile::get().assert_not_timed_out().unwrap_err(),
                        EngineError::OperationTimeOutError
                    );
                });
            });
    }

    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    #[test]
    fn ambient_profile_has_the_default_clock() {
        ExecutionProfile::get()
            .with_execution_timeout(60_000)
            .run(|| {
                let profile = ExecutionProfile::get();
                assert!(profile.limits_execution_time());
                assert!(profile.assert_not_timed_out().is_ok());
            });
    }

    #[test]
    #[cfg_attr(
        target_family = "wasm",
        ignore = "wasm is panic = abort: a panicking test aborts the whole binary"
    )]
    #[should_panic(expected = "no clock")]
    fn run_without_a_clock_panics_clearly() {
        let mut profile = ExecutionProfileBuilder::new().execution_timeout(10).build();
        profile.clock = None;
        profile.run(|| {});
    }

    #[test]
    fn run_without_a_timeout_needs_no_clock() {
        let mut profile = ExecutionProfileBuilder::new()
            .max_number_of_states(3)
            .build();
        profile.clock = None;
        profile.run(|| {
            let profile = ExecutionProfile::get();
            assert!(!profile.limits_execution_time());
            assert!(profile.assert_not_timed_out().is_ok());
        });
    }

    // `max_number_of_states(N)` allows exactly N states and only rejects N+1,
    // matching the documented "maximum an automaton may hold".
    #[test]
    fn max_number_of_states_allows_exactly_the_limit() {
        let profile = ExecutionProfileBuilder::new()
            .max_number_of_states(3)
            .build();
        assert!(profile.assert_max_number_of_states(2).is_ok());
        assert!(profile.assert_max_number_of_states(3).is_ok());
        assert_eq!(
            profile.assert_max_number_of_states(4).unwrap_err(),
            EngineError::AutomatonHasTooManyStates
        );
    }

    #[test]
    fn test_traits() -> Result<(), String> {
        assert_send::<ExecutionProfile>();
        assert_sync::<ExecutionProfile>();

        Ok(())
    }

    // `run`/`apply` must restore the previous thread profile even when the
    // closure panics: a leaked temporary profile would permanently poison
    // pooled (e.g. rayon) threads.
    #[test]
    #[cfg_attr(
        target_family = "wasm",
        ignore = "wasm is panic = abort: a panicking test aborts the whole binary"
    )]
    fn run_restores_previous_profile_on_panic() {
        let outer = ExecutionProfileBuilder::new()
            .max_number_of_states(123)
            .build();
        outer.run(|| {
            let inner = ExecutionProfileBuilder::new()
                .max_number_of_states(1)
                .build();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                inner.run(|| panic!("intentional test panic"));
            }));
            assert!(result.is_err());
            assert_eq!(outer, ExecutionProfile::get());
        });
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
        let cond = Condition::total(a.spanning_set());
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
                assert_eq!(nfa.cardinality().unwrap_err(), err);

                // ...but operations that work on NFAs directly are unaffected
                // (difference only determinizes the subtrahend)...
                assert!(nfa.difference(&dfa).is_ok());

                // ...deterministic inputs keep working...
                assert!(dfa.clone().minimize().is_ok());
                assert!(dfa.clone().complement().is_ok());
                assert!(dfa.cardinality().is_ok());
                assert!(dfa.equivalent(&dfa).is_ok());

                // ...and explicit determinization is always allowed.
                assert!(nfa.determinize().is_ok());
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
                assert!(term.cardinality().is_ok());
                assert!(term.minimize().is_ok());
                assert!(
                    term.generate_strings(5, 0, GenerationOptions::new())
                        .is_ok()
                );

                // ...and the rest of the API never needed one.
                assert!(term.concat(std::slice::from_ref(&other)).is_ok());
                assert!(term.union(std::slice::from_ref(&other)).is_ok());
                assert!(term.intersection(std::slice::from_ref(&other)).is_ok());
                assert!(term.repeat(0..=2).is_ok());
                assert!(term.is_empty().is_ok());
                assert!(term.is_empty_string().is_ok());
                let _ = term.length();
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

    #[test]
    fn test_implicit_determinization_default() {
        let nfa = nondeterministic_automaton();

        // Without the profile knob the historical behavior is unchanged.
        assert!(nfa.clone().minimize().is_ok());
        assert!(nfa.clone().complement().is_ok());
        assert!(nfa.cardinality().is_ok());
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
                    term.generate_strings(100_000_000, 1_000_000, GenerationOptions::new())
                        .unwrap_err()
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

        let execution_timeout_in_ms = 0;
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
                assert!(run_duration <= (execution_timeout_in_ms + 1000) as u128);
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
