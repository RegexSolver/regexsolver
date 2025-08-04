use super::*;
use crate::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde::{Deserializer, Serializer, de, ser};

use z85::{decode, encode};

use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use std::io::prelude::*;

use crate::tokenizer::token::{Token, automaton_token::AutomatonToken};

#[derive(Serialize, Deserialize, Debug)]
struct SerializedAutomaton(Vec<u16>, SpanningSet);

impl serde::Serialize for FastAutomaton {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let tokenizer = Tokenizer::new(self);
        match AutomatonToken::to_fair_tokens(&tokenizer.to_embedding()) {
            Ok(tokens) => {
                let serialized_automaton =
                    SerializedAutomaton(tokens, self.get_spanning_set().clone());

                let mut serialized = Vec::with_capacity(self.get_number_of_states() * 8);
                if let Err(err) = ciborium::into_writer(&serialized_automaton, &mut serialized) {
                    return Err(ser::Error::custom(err.to_string()));
                }

                serializer.serialize_str(&encode(compress_data(&serialized)))
            }
            Err(err) => Err(ser::Error::custom(err.to_string())),
        }
    }
}

impl<'de> serde::Deserialize<'de> for FastAutomaton {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match String::deserialize(deserializer) {
            Ok(decoded) => match decode(decoded) {
                Ok(compressed) => {
                    let payload = decompress_data(&compressed);

                    let automaton: Result<
                        SerializedAutomaton,
                        ciborium::de::Error<std::io::Error>,
                    > = ciborium::from_reader(&payload[..]);
                    match automaton {
                        Ok(automaton) => {
                            let mut temp_automaton = FastAutomaton::new_empty();
                            temp_automaton.spanning_set = automaton.1;
                            let tokenizer = Tokenizer::new(&temp_automaton);

                            match tokenizer.from_embedding(
                                &automaton
                                    .0
                                    .into_iter()
                                    .map(AutomatonToken::from_fair_token)
                                    .collect::<Vec<AutomatonToken>>(),
                            ) {
                                Ok(res) => Ok(res),
                                Err(err) => Err(de::Error::custom(err.to_string())),
                            }
                        }
                        Err(err) => Err(de::Error::custom(err.to_string())),
                    }
                }
                Err(err) => Err(de::Error::custom(err.to_string())),
            },
            Err(err) => Err(err),
        }
    }
}

fn compress_data(data: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).expect("Failed to write data");
    encoder.finish().expect("Failed to finish compression")
}

fn decompress_data(data: &[u8]) -> Vec<u8> {
    let mut decoder = ZlibDecoder::new(data);
    let mut decompressed_data = Vec::new();
    decoder
        .read_to_end(&mut decompressed_data)
        .expect("Failed to read data");
    decompressed_data
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    use super::*;

    #[test]
    fn test_serialization() -> Result<(), String> {
        assert_serialization("...");
        assert_serialization(".*abc");
        assert_serialization(".*");
        assert_serialization(".*abcdef.*dsqd");
        assert_serialization(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,2}",
        );
        assert_serialization(
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21-\\x5a\\x53-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])+)\\])",
        );

        Ok(())
    }

    fn assert_serialization(regex: &str) {
        let regex = RegularExpression::new(regex).unwrap();
        println!("{regex}");

        let automaton = regex.to_automaton().unwrap();

        let serialized = serde_json::to_string(&automaton).unwrap();
        println!("{serialized}");

        let unserialized: FastAutomaton = serde_json::from_str(&serialized).unwrap();

        let unserialized = unserialized.determinize().unwrap();
        let automaton = automaton.determinize().unwrap();

        assert!(automaton.difference(&unserialized).unwrap().is_empty());
        assert!(unserialized.difference(&automaton).unwrap().is_empty());
    }

    #[test]
    fn test_serialization_case_1() -> Result<(), String> {
        let automaton1 = RegularExpression::new(".*")
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::new("\\d+")
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = automaton2.determinize().unwrap();

        let difference = automaton1.difference(&automaton2).unwrap();

        let serialized = serde_json::to_string(&difference).unwrap();
        println!("{serialized}");

        let unserialized: FastAutomaton = serde_json::from_str(&serialized).unwrap();

        let unserialized = unserialized.determinize().unwrap();
        let automaton = difference.determinize().unwrap();

        assert!(automaton.difference(&unserialized).unwrap().is_empty());
        assert!(unserialized.difference(&automaton).unwrap().is_empty());

        Ok(())
    }
}
