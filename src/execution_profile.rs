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
///     assert_eq!(EngineError::OperationTimeOutError, term.generate_strings(1000).unwrap_err());
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
}

impl PartialEq for ExecutionProfile {
    fn eq(&self, other: &ExecutionProfile) -> bool {
        self.max_number_of_states == other.max_number_of_states
            && self.execution_timeout == other.execution_timeout
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
        if let Some(max_number_of_states) = self.max_number_of_states {
            if number_of_states >= max_number_of_states {
                return Err(EngineError::AutomatonHasTooManyStates);
            }
        }
        Ok(())
    }

    pub fn with_execution_timeout(mut self, execution_timeout_in_ms: u64) -> Self {
        self.execution_timeout = Some(execution_timeout_in_ms);
        self
    }

    pub fn with_max_number_of_states(mut self, max_number_of_states: usize) -> Self {
        self.max_number_of_states = Some(max_number_of_states);
        self
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
            execution_profile.execution_deadline = Some(Instant::now() + Duration::from_millis(execution_timeout));
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

    pub fn build(self) -> ExecutionProfile {
        ExecutionProfile {
            max_number_of_states: self.max_number_of_states,
            execution_timeout: self.execution_timeout,
            execution_deadline: None,
        }
    }
}

struct ThreadLocalParams;
impl ThreadLocalParams {
    thread_local! {
        static MAX_NUMBER_OF_STATES: RefCell<Option<usize>> = const { RefCell::new(None) };
        static EXECUTION_DEADLINE: RefCell<Option<Instant>> = const { RefCell::new(None) };
        static EXECUTION_TIMEOUT: RefCell<Option<u64>> = const { RefCell::new(None) };
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

    /// Return the [`ExecutionProfile`] stored on the current thread.
    fn get_execution_profile() -> ExecutionProfile {
        ExecutionProfile {
            max_number_of_states: Self::get_max_number_of_states(),
            execution_deadline: Self::get_execution_deadline(),
            execution_timeout: Self::get_execution_timeout(),
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
                    term.generate_strings(100).unwrap_err()
                );

                let run_duration = Instant::now()
                    .duration_since(start_time)
                    .as_millis();

                println!("{run_duration}");
                assert!(run_duration <= (execution_timeout_in_ms + 50) as u128);
            });

        Ok(())
    }

    #[test]
    fn test_execution_timeout_difference() -> Result<(), String> {
        let term1 = Term::from_pattern(".*abc.*def.*qdqd.*qsdsqdsqdz").unwrap();
        let term2 = Term::from_pattern(".*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz.*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz.*abc.*def.*qdsqd.*sqdsqd.*qsdsqdsqdz").unwrap();

        let execution_timeout_in_ms = 50;
        let start_time = Instant::now();
        ExecutionProfileBuilder::new()
            .execution_timeout(execution_timeout_in_ms)
            .build()
            .run(|| {
                assert_eq!(
                    EngineError::OperationTimeOutError,
                    term1.difference(&term2).unwrap_err()
                );

                let run_duration = Instant::now()
                    .duration_since(start_time)
                    .as_millis();

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
