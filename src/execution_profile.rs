use std::{cell::RefCell, time::SystemTime};

use crate::error::EngineError;

/// Hold information about limitations and constraints of operations execution:
/// - max_number_of_states: the maximum number of states that a non-determinitic finite automaton can hold.
/// - start_execution_time: timestamp of when the execution has started.
/// - execution_timeout: the longest time in milliseconds that an operation execution can last.
/// - max_number_of_terms: the maximum number of terms that an operation can have.
pub struct ExecutionProfile {
    pub max_number_of_states: usize,
    pub start_execution_time: Option<SystemTime>,
    pub execution_timeout: u128,
    pub max_number_of_terms: usize,
}

impl ExecutionProfile {
    pub fn is_timed_out(&self) -> Result<(), EngineError> {
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

pub struct ThreadLocalParams;
impl ThreadLocalParams {
    thread_local! {
        static MAX_NUMBER_OF_STATES: RefCell<usize> = const { RefCell::new(8192) };
        static START_EXECUTION_TIME: RefCell<Option<SystemTime>> = const { RefCell::new(None) };
        static EXECUTION_TIMEOUT: RefCell<u128> = const { RefCell::new(1500) };
        static MAX_NUMBER_OF_TERMS: RefCell<usize> = const { RefCell::new(50) };
    }

    /// Initialize the thread local holding the ExecutionProfile.
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
