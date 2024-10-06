use super::*;

impl RegularExpression {
    pub fn simplify(&self) -> Self {
        match self {
            RegularExpression::Character(_) => self.clone(),
            RegularExpression::Repetition(regex, min, max_opt) => {
                let regex = regex.simplify();
                match regex {
                    RegularExpression::Repetition(
                        simplified_regex,
                        simplified_min,
                        simplified_max_opt,
                    ) => {
                        let new_max = if let (Some(max), Some(simplified_max)) =
                            (max_opt, simplified_max_opt)
                        {
                            Some(max * simplified_max)
                        } else {
                            None
                        };
                        RegularExpression::Repetition(
                            simplified_regex,
                            min * simplified_min,
                            new_max,
                        )
                    }
                    _ => RegularExpression::Repetition(Box::new(regex), *min, *max_opt),
                }
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
                let elements: VecDeque<_> =
                    elements.iter().map(|element| element.simplify()).collect();

                let mut regex = RegularExpression::new_empty();
                elements
                    .iter()
                    .for_each(|element| regex = regex.union(element));
                regex
            }
        }
    }
}
