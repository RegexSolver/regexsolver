/// The number of strings of a language: a count, a count too large for its
/// integer type, or infinite.
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Cardinality<U> {
    /// Infinitely many.
    Infinite,
    /// A count that fits its integer type.
    Integer(U),
    /// A finite count too large for its integer type.
    BigInteger,
}
