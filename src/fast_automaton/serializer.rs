use super::*;
use lazy_static::lazy_static;
use rand::Rng;
use serde::{de, ser, Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::env;
use z85::{decode, encode};
use crate::tokenizer::Tokenizer;

use sha2::{Digest, Sha256};

use aes_gcm_siv::{
    aead::{Aead, KeyInit},
    Aes256GcmSiv, Nonce,
};
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::prelude::*;

use crate::tokenizer::token::{automaton_token::AutomatonToken, Token};

pub struct FastAutomatonReader {
    cipher: Aes256GcmSiv,
}

impl FastAutomatonReader {
    pub fn new() -> Self {
        let env_var = env::var("RS_FAIR_SECRET_KEY").unwrap_or("DEFAULT PASSKEY".to_string());
        let key = Sha256::digest(env_var.as_bytes());
        FastAutomatonReader {
            cipher: Aes256GcmSiv::new(&key),
        }
    }

    pub fn random_nonce() -> [u8; 12] {
        let mut nonce = [0u8; 12];
        rand::thread_rng().fill(&mut nonce);
        nonce
    }
}

lazy_static! {
    static ref SINGLETON_INSTANCE: FastAutomatonReader = FastAutomatonReader::new();
}

fn get_fast_automaton_reader() -> &'static FastAutomatonReader {
    &SINGLETON_INSTANCE
}

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

                serialized = compress_data(&serialized);

                let nonce = FastAutomatonReader::random_nonce();

                match get_fast_automaton_reader()
                    .cipher
                    .encrypt(Nonce::from_slice(&nonce), serialized.as_ref())
                {
                    Ok(ciphertext) => {
                        let mut encrypted = Vec::from_iter(nonce);
                        encrypted.extend(ciphertext);

                        serializer.serialize_str(&encode(&encrypted))
                    }
                    Err(err) => Err(ser::Error::custom(err.to_string())),
                }
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
                Ok(encrypted) => {
                    let nonce = &encrypted[0..12];
                    let payload = encrypted[12..].to_vec();
                    let cipher_result = get_fast_automaton_reader()
                        .cipher
                        .decrypt(Nonce::from_slice(nonce), payload.as_ref());

                    match cipher_result {
                        Ok(cipher_result) => {
                            let decrypted = decompress_data(&cipher_result);

                            let automaton: Result<
                                SerializedAutomaton,
                                ciborium::de::Error<std::io::Error>,
                            > = ciborium::from_reader(&decrypted[..]);
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
        assert_serialization("(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21-\\x5a\\x53-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])+)\\])");

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

        assert!(automaton.subtraction(&unserialized).unwrap().is_empty());
        assert!(unserialized.subtraction(&automaton).unwrap().is_empty());
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
            .unwrap()
            .determinize()
            .unwrap();

        let subtraction = automaton1.subtraction(&automaton2).unwrap();

        let serialized = serde_json::to_string(&subtraction).unwrap();
        println!("{serialized}");

        let unserialized: FastAutomaton = serde_json::from_str(&serialized).unwrap();

        let unserialized = unserialized.determinize().unwrap();
        let automaton = subtraction.determinize().unwrap();

        assert!(automaton.subtraction(&unserialized).unwrap().is_empty());
        assert!(unserialized.subtraction(&automaton).unwrap().is_empty());
        
        Ok(())
    }
}
