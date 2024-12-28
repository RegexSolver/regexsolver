#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represent a number.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(PartialEq, Eq, Debug, Clone)]
#[cfg_attr(feature = "serde", serde(tag = "type", content = "value"))]
pub enum Cardinality<U> {
    /// An infinite number.
    Infinite,
    /// A finite number.
    Integer(U),
    /// A finite number too big to be represented.
    BigInteger,
}
