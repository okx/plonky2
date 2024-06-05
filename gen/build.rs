#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::fs::File;
use std::io::{Error, Write};
use std::path::Path;
use std::{env, fs};
extern crate alloc;
use alloc::alloc::{alloc, Layout};

use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::ops::Square;
use plonky2_field::types::Field;
use plonky2_util::pre_compute::{get_pre_compute_size, PRE_COMPUTE_END, PRE_COMPUTE_START};

pub struct RouContext<const START: usize, const END: usize>
where
    [(); get_pre_compute_size(START, END)]:,
{
    pub pre_rou: [u64; get_pre_compute_size(START, END)],
}

impl<const START: usize, const END: usize> RouContext<START, END>
where
    [(); get_pre_compute_size(START, END)]:,
{
    /// Create a new `ECMultContext` from raw values.
    ///
    /// # Safety
    /// The function is unsafe because incorrect value of `pre_g` can lead to
    /// crypto logic failure. You most likely do not want to use this function,
    /// but `ECMultContext::new_boxed`.
    pub const unsafe fn new_from_raw(pre_rou: [u64; get_pre_compute_size(START, END)]) -> Self {
        Self { pre_rou }
    }

    /// Inspect raw values of `RouContext<N>`.
    pub fn inspect_raw(&self) -> &[u64; get_pre_compute_size(START, END)] {
        &self.pre_rou
    }

    /// Generate a new `RouContext<N>` on the heap.
    pub fn new_boxed() -> Box<Self> {
        // This unsafe block allocates RouContext<N> and then fills in the value. This is to avoid allocating it on stack
        // because the data is big.
        let this = unsafe {
            let ptr = alloc(Layout::new::<RouContext<START, END>>()) as *mut RouContext<START, END>;
            let mut this = Box::from_raw(ptr);

            // let lg_n: usize = START;
            let mut j = 0;
            for lg_n in START..END + 1 {
                let mut bases = Vec::with_capacity(lg_n);
                let mut base = GoldilocksField::primitive_root_of_unity(lg_n);
                bases.push(base);
                for _ in 1..lg_n {
                    base = base.square(); // base = g^2^_
                    bases.push(base);
                }

                // let mut root_table = Vec::with_capacity(lg_n);

                for lg_m in 1..=lg_n {
                    let half_m = 1 << (lg_m - 1);
                    let base = bases[lg_n - lg_m];
                    let root_row: Vec<GoldilocksField> =
                        base.powers().take(half_m.max(2)).collect();
                    // root_table.push(root_row);
                    for val in root_row {
                        this.pre_rou[j] = val.0;
                        j += 1;
                    }
                }
            }

            this
        };

        this
    }
}

pub fn generate_to(file: &mut File) -> Result<(), Error> {
    println!(
        "generate for start {:?}, end {:?}",
        PRE_COMPUTE_START, PRE_COMPUTE_END
    );
    let context = RouContext::<PRE_COMPUTE_START, PRE_COMPUTE_END>::new_boxed();
    let pre_g = context.inspect_raw().as_ref();

    _ = file.write_fmt(format_args!("["));
    for pg in pre_g {
        _ = file.write_fmt(format_args!("    {},", pg));
    }
    _ = file.write_fmt(format_args!("]"));

    Ok(())
}

fn main() {
    println!("cargo:rerun-if-changed=src");
    let directory = env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("directory: {:?}", directory);
    let path = Path::new(&directory)
        .parent()
        .unwrap()
        .join("field/generated"); //
    if !path.exists() {
        fs::create_dir_all(&path).expect("create file error");
    }
    println!("path: {:?}", path);
    let mut file = File::create(&path.join("goldilock_root_of_unity.rs"))
        .expect("Create const.rs file failed");
    generate_to(&mut file).expect("Write const_gen.rs file failed");

    file.flush()
        .expect("Flush goldilock_root_of_unity.rs file failed");
}
