use serde_derive::{Deserialize, Serialize};

/// Represent a number.
#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Cardinality<U> {
    /// An infinite number.
    Infinite,
    /// A finite number.
    Integer(U),
    /// A finite number too big to be represented.
    BigInteger,
}
