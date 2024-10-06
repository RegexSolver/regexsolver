use serde::{de, Deserializer, Serializer};

use super::*;

impl serde::Serialize for RegularExpression {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for RegularExpression {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let regex_string = match String::deserialize(deserializer) {
            Ok(str) => str,
            Err(err) => return Err(err),
        };
        match RegularExpression::new(&regex_string) {
            Ok(regex) => Ok(regex),
            Err(err) => Err(de::Error::custom(err.to_string())),
        }
    }
}
