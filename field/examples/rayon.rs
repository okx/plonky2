use rayon::prelude::*;

fn main() {
    let packed_n = 1 << 4;
    let packed_m = 4;
    let half_packed_m = packed_m >> 1;

    let mut arr: Vec<usize> = (0..(packed_n as usize)).into_iter().map(|x| x).collect();
    arr.par_chunks_mut(packed_m)
        .enumerate()
        .for_each(|(chunk_idx, slice)| {
            let (lo, hi) = slice.split_at_mut(half_packed_m);

            lo.par_iter_mut()
                .zip(hi)
                .enumerate()
                .for_each(|(j, (lo_val, hi_val))| {
                    let k = packed_m * chunk_idx;
                    println!("{:?}, {:?}, lo_val: {:?}, hi_val: {:?}", k + j, k + half_packed_m + j, *lo_val, *hi_val);
                })
        });

}
