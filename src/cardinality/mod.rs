use serde_derive::{Deserialize, Serialize};

pub trait IntegerTrait {}

impl IntegerTrait for u128 {}
impl IntegerTrait for u32 {}

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Cardinality<U: IntegerTrait> {
    Infinite,
    Integer(U),
    BigInteger,
}
