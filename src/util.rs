/// Extract u64 from byte data.
pub fn getu64(data: &[u8], off: usize) -> u64 {
    let data = &data[off..off + 8];
    u64::from_le_bytes(data.try_into().unwrap())
}

/// Store u64 to byte data.
pub fn setu64(data: &mut [u8], val: u64) {
    data[0..8].copy_from_slice(&val.to_le_bytes());
}

/// Extract unsigned value of n bytes from data.
pub fn get(data: &[u8], off: usize, n: usize) -> u64 {
    let mut buf = [0_u8; 8];
    buf[0..n].copy_from_slice(&data[off..off + n]);
    u64::from_le_bytes(buf)
}

/// Store unsigned value of n bytes to data.
pub fn set(data: &mut [u8], off: usize, val: u64, n: usize) {
    let bytes = val.to_le_bytes();
    data[off..off + n].copy_from_slice(&bytes[0..n]);
}
