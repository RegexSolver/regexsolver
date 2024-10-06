use token::TokenError;

use crate::regex::RegularExpression;

use self::token::regex_operations_token::RegexOperationsToken;

use super::*;

impl Tokenizer<'_> {
    pub fn to_regex_operations_embedding(
        &self,
        regex_operations: &[(bool, RegularExpression)],
    ) -> Vec<RegexOperationsToken> {
        let mut vec = vec![];

        for (not, regex) in regex_operations {
            if !vec.is_empty() {
                vec.push(RegexOperationsToken::And);
            }
            if *not {
                vec.push(RegexOperationsToken::Not);
            }

            vec.extend(
                self.to_regex_embedding(regex)
                    .into_iter()
                    .map(RegexOperationsToken::RegexToken),
            );
        }

        vec
    }

    pub fn from_regex_operations_embedding(
        &self,
        vec: &[RegexOperationsToken],
    ) -> Result<Vec<(bool, RegularExpression)>, TokenError> {
        let mut operations = vec![];
        let mut current_regex_not = false;
        let mut current_regex_token = vec![];
        for token in vec {
            match token {
                RegexOperationsToken::RegexToken(regex_token) => {
                    current_regex_token.push(*regex_token)
                }
                RegexOperationsToken::And => {
                    let regex = self.from_regex_embedding(&current_regex_token)?;
                    operations.push((current_regex_not, regex));
                    current_regex_not = false;
                    current_regex_token.clear();
                }
                RegexOperationsToken::Not => current_regex_not = true,
                RegexOperationsToken::Error => return Err(TokenError::UnknownToken),
            };
        }

        if !current_regex_token.is_empty() {
            let regex = self.from_regex_embedding(&current_regex_token)?;
            operations.push((current_regex_not, regex));
        }

        Ok(operations)
    }
}

#[cfg(test)]
mod tests {
    use embed_regex_operations::token::Token;

    use crate::regex::RegularExpression;

    use super::*;

    #[test]
    fn test_tokenize() -> Result<(), String> {
        assert_embedding_convertion(&[(false, "(a|b)")]);
        assert_embedding_convertion(&[(false, "(|a)")]);
        assert_embedding_convertion(&[(false, ".*ab")]);
        assert_embedding_convertion(&[(true, "toto")]);
        assert_embedding_convertion(&[(false, ".{2,3}")]);
        assert_embedding_convertion(&[(false, "q(abc?|ca)x")]);
        assert_embedding_convertion(&[(false, ".*q(abc?|ca)x")]);
        assert_embedding_convertion(&[(false, "(abc){3,6}")]);
        assert_embedding_convertion(&[(true, "((|a)abd+){3}")]);

        assert_embedding_convertion(&[(false, ".*a.*"), (false, ".*b.*"), (true, ".*abc.*")]);
        Ok(())
    }

    fn assert_embedding_convertion(operations: &[(bool, &str)]) {
        let mut automaton = FastAutomaton::new_total();
        let operations: Vec<(bool, RegularExpression)> = operations
            .iter()
            .map(|(not, regex)| {
                let regex = RegularExpression::new(regex).unwrap();
                automaton = automaton.intersection(&regex.to_automaton().unwrap()).unwrap();
                (*not, regex)
            })
            .collect();

        let tokenizer = Tokenizer::new(&automaton);
        let embedding = tokenizer.to_regex_operations_embedding(&operations);

        // AI
        let embedding_u8: Vec<u8> = RegexOperationsToken::to_ai_tokens(&embedding).unwrap();
        assert_eq!(
            embedding,
            embedding_u8
                .iter()
                .map(|&t| RegexOperationsToken::from_ai_token(t))
                .collect::<Vec<_>>()
        );

        let unembedded_operations = tokenizer
            .from_regex_operations_embedding(&embedding)
            .unwrap();
        assert_eq!(operations, unembedded_operations);
    }
}
