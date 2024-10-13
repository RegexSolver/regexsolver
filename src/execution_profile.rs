use std::{cell::RefCell, time::SystemTime};

use crate::error::EngineError;

/// Hold settings about limitations and constraints of operations execution within the engine.
///
/// To apply the settings on the current thread you need to call the following function:
/// ```
/// use regexsolver::execution_profile::{ExecutionProfile, ThreadLocalParams};
///
/// let execution_profile = ExecutionProfile {
///     max_number_of_states: 1,
///     start_execution_time: None,
///     execution_timeout: 1000,
///     max_number_of_terms: 10,
/// };
/// 
/// // Store the settings on the current thread.
/// ThreadLocalParams::init_profile(&execution_profile);
/// ```
///
/// # Examples:
///
/// ## Limiting the number of states
/// ```
/// use regexsolver::{Term, execution_profile::{ExecutionProfile, ThreadLocalParams}, error::EngineError};
///
/// let term1 = Term::from_regex(".*abc.*").unwrap();
/// let term2 = Term::from_regex(".*def.*").unwrap();
///
/// let execution_profile = ExecutionProfile {
///     max_number_of_states: 1,
///     start_execution_time: None,
///     execution_timeout: 1000,
///     max_number_of_terms: 10,
/// };
/// ThreadLocalParams::init_profile(&execution_profile);
///
/// assert_eq!(EngineError::AutomatonHasTooManyStates, term1.intersection(&[term2]).unwrap_err());
/// ```
///
/// ## Limiting the number of terms
/// ```
/// use regexsolver::{Term, execution_profile::{ExecutionProfile, ThreadLocalParams}, error::EngineError};
///
/// let term1 = Term::from_regex(".*abc.*").unwrap();
/// let term2 = Term::from_regex(".*def.*").unwrap();
/// let term3 = Term::from_regex(".*hij.*").unwrap();
///
/// let execution_profile = ExecutionProfile {
///     max_number_of_states: 8192,
///     start_execution_time: None,
///     execution_timeout: 1000,
///     max_number_of_terms: 2,
/// };
/// ThreadLocalParams::init_profile(&execution_profile);
///
/// assert_eq!(EngineError::TooMuchTerms(2,3), term1.intersection(&[term2, term3]).unwrap_err());
/// ```
///
/// ## Limiting the execution time
/// ```
/// use regexsolver::{Term, execution_profile::{ExecutionProfile, ThreadLocalParams}, error::EngineError};
/// use std::time::SystemTime;
///
/// let term = Term::from_regex(".*abc.*cdef.*sqdsqf.*").unwrap();
///
/// let execution_profile = ExecutionProfile {
///     max_number_of_states: 8192,
///     start_execution_time: Some(SystemTime::now()),
///     execution_timeout: 1,
///     max_number_of_terms: 50,
/// };
/// ThreadLocalParams::init_profile(&execution_profile);
///
/// assert_eq!(EngineError::OperationTimeOutError, term.generate_strings(100).unwrap_err());
/// ```
pub struct ExecutionProfile {
    /// The maximum number of states that a non-determinitic finite automaton can hold, this is checked during the convertion of regular expression to automaton.
    pub max_number_of_states: usize,
    /// Timestamp of when the execution has started, if this value is not set the operations will never timeout.
    pub start_execution_time: Option<SystemTime>,
    /// The longest time in milliseconds that an operation execution can last, there are no guaranties that the exact time will be respected.
    pub execution_timeout: u128,
    /// The maximum number of terms that an operation can have.
    pub max_number_of_terms: usize,
}

impl ExecutionProfile {
    /// Assert that `execution_timeout` is not exceeded.
    ///
    /// Return empty if `execution_timeout` is not exceeded or if `start_execution_time` is not set.
    /// 
    /// Return [`EngineError::OperationTimeOutError`] otherwise.
    pub fn assert_not_timed_out(&self) -> Result<(), EngineError> {
        if let Some(start) = self.start_execution_time {
            let run_duration = SystemTime::now()
                .duration_since(start)
                .expect("Time went backwards")
                .as_millis();

            if run_duration > self.execution_timeout {
                Err(EngineError::OperationTimeOutError)
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}


/// Hold [`ExecutionProfile`] on the current thread.
/// 
/// The default [`ExecutionProfile`] is the following:
/// ```
/// use regexsolver::execution_profile::ExecutionProfile;
/// 
/// ExecutionProfile {
///     max_number_of_states: 8192,
///     start_execution_time: None,
///     execution_timeout: 1500,
///     max_number_of_terms: 50,
/// };
/// ```
pub struct ThreadLocalParams;
impl ThreadLocalParams {
    thread_local! {
        static MAX_NUMBER_OF_STATES: RefCell<usize> = const { RefCell::new(8192) };
        static START_EXECUTION_TIME: RefCell<Option<SystemTime>> = const { RefCell::new(None) };
        static EXECUTION_TIMEOUT: RefCell<u128> = const { RefCell::new(1500) };
        static MAX_NUMBER_OF_TERMS: RefCell<usize> = const { RefCell::new(50) };
    }

    /// Store on the current thread [`ExecutionProfile`].
    pub fn init_profile(profile: &ExecutionProfile) {
        ThreadLocalParams::MAX_NUMBER_OF_STATES.with(|cell| {
            *cell.borrow_mut() = profile.max_number_of_states;
        });

        ThreadLocalParams::START_EXECUTION_TIME.with(|cell| {
            *cell.borrow_mut() = profile.start_execution_time;
        });

        ThreadLocalParams::EXECUTION_TIMEOUT.with(|cell| {
            *cell.borrow_mut() = profile.execution_timeout;
        });

        ThreadLocalParams::MAX_NUMBER_OF_TERMS.with(|cell| {
            *cell.borrow_mut() = profile.max_number_of_terms;
        });
    }

    pub fn get_max_number_of_states() -> usize {
        ThreadLocalParams::MAX_NUMBER_OF_STATES.with(|cell| *cell.borrow())
    }

    pub fn get_start_execution_time() -> Option<SystemTime> {
        ThreadLocalParams::START_EXECUTION_TIME.with(|cell| *cell.borrow())
    }

    pub fn get_execution_timeout() -> u128 {
        ThreadLocalParams::EXECUTION_TIMEOUT.with(|cell| *cell.borrow())
    }

    pub fn get_max_number_of_terms() -> usize {
        ThreadLocalParams::MAX_NUMBER_OF_TERMS.with(|cell| *cell.borrow())
    }

    /// Return the [`ExecutionProfile`] stored on the current thread.
    pub fn get_execution_profile() -> ExecutionProfile {
        ExecutionProfile {
            max_number_of_states: Self::get_max_number_of_states(),
            start_execution_time: Self::get_start_execution_time(),
            execution_timeout: Self::get_execution_timeout(),
            max_number_of_terms: Self::get_max_number_of_terms(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    use super::*;

    #[test]
    fn test_execution() -> Result<(), String> {
        let execution_profile = ExecutionProfile {
            max_number_of_states: 1,
            start_execution_time: None,
            execution_timeout: 1000,
            max_number_of_terms: 10,
        };
        ThreadLocalParams::init_profile(&execution_profile);

        let regex = RegularExpression::new("test").unwrap();

        assert!(regex.to_automaton().is_err());
        assert_eq!(
            EngineError::AutomatonHasTooManyStates,
            regex.to_automaton().unwrap_err()
        );

        Ok(())
    }
}
