use super::*;

impl RegularExpression {
    /// Returns a simplified version by eliminating redundant constructs and applying canonical reductions.
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn simplify(&self) -> Self {
        match self {
            RegularExpression::Character(..) => self.clone(),
            RegularExpression::Repetition(regex, min, max_opt) => {
                // Delegate to `repeat`, which guards the nested-repetition
                // collapse with `can_simplify_nested_repetition`. Collapsing
                // `(r{a,b}){c,d}` to `r{a*c,b*d}` unconditionally is unsound
                // when the step lengths leave a gap (e.g. `(a{3,4}){1,2}`
                // would wrongly widen to `a{3,8}`).
                regex.simplify().repeat(*min, *max_opt)
            }
            RegularExpression::Concat(elements) => {
                let elements: VecDeque<_> =
                    elements.iter().map(|element| element.simplify()).collect();

                let mut regex = RegularExpression::new_empty_string();
                elements
                    .iter()
                    .for_each(|element| regex = regex.concat(element, true));
                regex
            }
            RegularExpression::Alternation(elements) => {
                let elements: Vec<_> = elements.iter().map(|element| element.simplify()).collect();

                RegularExpression::union_all(elements.iter())
            }
        }
    }
}
