pub const PRE_COMPUTE_START: usize = 10;
pub const PRE_COMPUTE_END: usize = 18;

pub const fn get_pre_compute_size(start: usize, end: usize) -> usize {
    let nums = (1 << (end - start + 1)) - 1;
    let base = 1 << start;
    return base * nums;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_pre_compute_size() {
        assert_eq!(get_pre_compute_size(10, 10), 1024);
        assert_eq!(get_pre_compute_size(10, 11), 3072);
        assert_eq!(get_pre_compute_size(14, 14), 16384);
        assert_eq!(get_pre_compute_size(10, 16), 130048);
    }
}
