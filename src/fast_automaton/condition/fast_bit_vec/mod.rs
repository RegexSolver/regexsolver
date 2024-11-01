#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct FastBitVec {
    bits: Vec<u64>,
    n: usize,
}

impl std::fmt::Display for FastBitVec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.n {
            let bit = if self.get(i).unwrap() { 1 } else { 0 };
            write!(f, "{}", bit)?;
        }
        Ok(())
    }
}

impl FastBitVec {
    #[inline]
    pub fn from_elem(n: usize, bit: bool) -> Self {
        let nblocks = if n % 64 == 0 { n / 64 } else { n / 64 + 1 };
        let bits = vec![if bit { !0_u64 } else { 0_u64 }; nblocks];
        let mut bit_vec = FastBitVec { bits, n };
        bit_vec.fix_last_block();
        bit_vec
    }

    fn fix_last_block(&mut self) {
        if let Some((last_block, used_bits)) = self.last_block_mut_with_mask() {
            *last_block &= used_bits;
        }
    }

    #[inline]
    fn last_block_mut_with_mask(&mut self) -> Option<(&mut u64, u64)> {
        let extra_bits = self.len() % 64;
        if extra_bits > 0 {
            let mask = (1 << extra_bits) - 1;
            let storage_len = self.bits.len();
            Some((&mut self.bits[storage_len - 1], mask))
        } else {
            None
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn get(&self, i: usize) -> Option<bool> {
        if i >= self.n {
            return None;
        }
        let w = i / 64;
        let b = i % 64;
        self.bits.get(w).map(|&block| (block & (1 << b)) != 0)
    }

    #[inline]
    pub fn set(&mut self, i: usize, x: bool) {
        let w = i / 64;
        let b = i % 64;
        let flag = 1 << b;
        let val = if x {
            self.bits[w] | flag
        } else {
            self.bits[w] & !flag
        };
        self.bits[w] = val;
    }

    #[inline]
    pub fn complement(&mut self) {
        for w in &mut self.bits {
            *w = !*w;
        }
        self.fix_last_block();
    }

    #[inline]
    pub fn union(&mut self, other: &Self) {
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            let w = *a | b;
            *a = w;
        }
    }

    #[inline]
    pub fn intersection(&mut self, other: &Self) {
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            let w = *a & b;
            *a = w;
        }
    }

    #[inline]
    pub fn has_intersection(&self, other: &Self) -> bool {
        for (a, b) in self.bits.iter().zip(&other.bits) {
            if *a & b != 0 {
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn empty(&self) -> bool {
        self.bits.iter().all(|w| w == &0)
    }

    #[inline]
    pub fn total(&self) -> bool {
        let mut last_word = !0;
        self.bits.iter().all(|elem| {
            let tmp = last_word;
            last_word = *elem;
            tmp == !0
        }) && (last_word == Self::mask_for_bits(self.n))
    }

    fn mask_for_bits(bits: usize) -> u64 {
        (!0) >> ((64 - bits % 64) % 64)
    }

    pub fn get_hot_bits(&self) -> Vec<bool> {
        let mut hot_bits = Vec::with_capacity(self.n);
        for i in 0..self.n {
            hot_bits.push(self.get(i).unwrap());
        }
        hot_bits
    }
}
