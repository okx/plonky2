extern crate bindgen;

use std::env;
use std::path::{PathBuf, Path};

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let poseidon_dir  = Path::new(&dir).join("poseidon_bn128");
    println!("cargo:rustc-link-search=native={}", poseidon_dir.display());

    println!("cargo:rustc-link-lib=static=poseidon-permute-c");

    let bindings = bindgen::Builder::default()
        .header(poseidon_dir.join("wrapper.h").display().to_string())
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("{}", out_path.to_str().unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
