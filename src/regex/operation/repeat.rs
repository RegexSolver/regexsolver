use super::*;

impl RegularExpression {
    /// Returns the repetition of the `RegularExpression`, between `min` and `max_opt` times. If `max_opt` is `None`, the repetition is unbounded.
    pub fn repeat(&self, min: u32, max_opt: Option<u32>) -> RegularExpression {
        if self.is_total() {
            return RegularExpression::new_total();
        } else if self.is_empty() {
            return RegularExpression::new_empty();
        } else if self.is_empty_string() {
            return Self::new_empty_string();
        } else if let Some(max) = max_opt {
            if max < min || max == 0 {
                return RegularExpression::new_empty_string();
            } else if min == 1 && max == 1 {
                return self.clone();
            }
        }

        match self {
            RegularExpression::Repetition(regular_expression, o_min, o_max_opt) => {
                let new_max = if let (Some(max), Some(o_max)) = (max_opt, o_max_opt) {
                    Some(max * o_max)
                } else {
                    None
                };

                let o_min = *o_min;
                if let Some(o_max) = o_max_opt {
                    let o_max = *o_max;
                    if o_min <= 1 || max_opt.is_some() && max_opt.unwrap() == min {
                        RegularExpression::Repetition(
                            regular_expression.clone(),
                            min * o_min,
                            new_max,
                        )
                    } else if o_min == o_max && o_min > 1 {
                        RegularExpression::Repetition(Box::new(self.clone()), min, max_opt)
                    } else {
                        let r = ((o_max as f64) - 1f64) / ((o_max as f64) - (o_min as f64));
                        if r > cmp::max(2, min) as f64 {
                            return RegularExpression::Repetition(
                                Box::new(self.clone()),
                                min,
                                max_opt,
                            );
                        }

                        RegularExpression::Repetition(
                            regular_expression.clone(),
                            min * o_min,
                            new_max,
                        )
                    }
                } else if o_max_opt.is_none()
                    || max_opt.is_some() && (max_opt.unwrap() == min || max_opt.unwrap() == 1)
                    || o_max_opt.is_some() && o_max_opt.unwrap() == 1
                    || max_opt.is_none() && o_min == 0
                {
                    RegularExpression::Repetition(regular_expression.clone(), min * o_min, new_max)
                } else {
                    RegularExpression::Repetition(Box::new(self.clone()), min, max_opt)
                }
            }
            _ => RegularExpression::Repetition(Box::new(self.clone()), min, max_opt),
        }
    }
}