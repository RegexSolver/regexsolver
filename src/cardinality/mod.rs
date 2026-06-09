/// Represents a cardinality: either a specific integer, a number too large to represent exactly, or infinite.
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Cardinality<U> {
    /// An infinite number.
    Infinite,
    /// A finite number.
    Integer(U),
    /// A finite number too big to be represented.
    BigInteger,
}
