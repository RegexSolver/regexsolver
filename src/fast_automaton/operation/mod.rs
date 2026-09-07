use std::cmp;

use super::*;
use condition::converter::ConditionConverter;

/// Projects `condition` through `converter`, or borrows it unchanged when no
/// projection is needed (`None`: both operands already share the spanning
/// set, the dominant case in operation chains).
fn convert_condition<'c>(
    converter: Option<&ConditionConverter<'_, '_>>,
    condition: &'c Condition,
) -> Result<std::borrow::Cow<'c, Condition>, EngineError> {
    Ok(match converter {
        Some(converter) => std::borrow::Cow::Owned(converter.convert(condition)?),
        None => std::borrow::Cow::Borrowed(condition),
    })
}

mod concat;
mod determinize;
mod difference;
mod intersection;
mod minimize;
mod repeat;
mod union;

impl FastAutomaton {
    /// The shared preamble of the non-degenerate operation cores
    /// (`concat_mut_nondegenerate`, `union_mut_nondegenerate`): a cheap
    /// necessary condition for the caller-guaranteed invariant (the full
    /// degenerate checks are exactly what the cores exist to avoid re-running),
    /// the timeout check, and, only when a state limit is configured, the
    /// predicted-size check.
    fn assert_nondegenerate_operation_fits(
        &self,
        other: &FastAutomaton,
        predicted_states: impl FnOnce() -> usize,
    ) -> Result<(), crate::error::EngineError> {
        debug_assert!(!self.accept_states.is_empty() && !other.accept_states.is_empty());

        let execution_profile = crate::execution_profile::ExecutionProfile::get();
        execution_profile.assert_not_timed_out()?;
        if execution_profile.limits_number_of_states() {
            execution_profile.assert_max_number_of_states(predicted_states())?;
        }
        Ok(())
    }

    /// Removes "dead" states (those that cannot reach any accept state), since
    /// they never contribute to the language. If the language is empty the whole
    /// automaton collapses to the canonical empty automaton.
    pub fn remove_dead_states(&mut self) {
        if !self.is_empty() {
            let live_states = self.live_states();

            let mut dead_states = IntSet::default();
            for from_state in self.states() {
                if !live_states.contains(&from_state) {
                    dead_states.insert(from_state);
                }
            }
            self.remove_states(&dead_states);
        } else {
            self.make_empty();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_remove_dead_states() -> Result<(), String> {
        let automaton1 = RegularExpression::parse("(abc|ac|aaa)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse("(abcd|ac|aba)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();
        assert_eq!(3, intersection.number_of_states());
        assert_eq!(3, intersection.live_states().len());
        Ok(())
    }
}
