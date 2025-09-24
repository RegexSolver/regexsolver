#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// Represent a number.
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
#[derive(PartialEq, Eq, Debug, Clone)]
#[cfg_attr(feature = "serializable", serde(tag = "type", content = "value", rename_all = "camelCase"))]
pub enum Cardinality<U> {
    /// An infinite number.
    Infinite,
    /// A finite number.
    Integer(U),
    /// A finite number too big to be represented.
    BigInteger,
}
