pub trait MethodParameters<'a, T: 'a> {
    /// the iterator that yields `&'a T`
    type Iter: Iterator<Item = &'a T>;
    fn parameters(self) -> Self::Iter;
}

impl<'a, T> MethodParameters<'a, T> for &'a T {
    type Iter = std::iter::Once<&'a T>;
    fn parameters(self) -> Self::Iter {
        std::iter::once(self)
    }
}

impl<'a, T> MethodParameters<'a, T> for &'a [&'a T] {
    type Iter = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn parameters(self) -> Self::Iter {
        self.iter().copied()
    }
}

impl<'a, T> MethodParameters<'a, T> for &'a [T] {
    type Iter = std::slice::Iter<'a, T>;
    fn parameters(self) -> Self::Iter {
        self.iter()
    }
}

impl<'a, T> MethodParameters<'a, T> for &'a Vec<T> {
    type Iter = std::slice::Iter<'a, T>;
    fn parameters(self) -> Self::Iter {
        self.iter()
    }
}

impl<'a, T> MethodParameters<'a, T> for &'a Vec<&'a T> {
    type Iter = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn parameters(self) -> Self::Iter {
        self.iter().copied()
    }
}

impl<'a, T, const N: usize> MethodParameters<'a, T> for &'a [T; N] {
    type Iter = std::slice::Iter<'a, T>;
    fn parameters(self) -> Self::Iter {
        self.iter()
    }
}

impl<'a, T, const N: usize> MethodParameters<'a, T> for &'a [&'a T; N] {
    type Iter = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn parameters(self) -> Self::Iter {
        self.iter().copied()
    }
}
