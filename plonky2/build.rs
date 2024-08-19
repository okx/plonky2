extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn merkle_avx512() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let mdir = Path::new(&dir).join("merkle_avx512");
    let lib_file = mdir.join("merkle_avx512.a");

    if !lib_file.exists() {
        assert!(env::set_current_dir(&mdir).is_ok());
        Command::new("make")
            .output()
            .expect("failed to execute make");
        assert!(env::set_current_dir(&dir).is_ok());
    }

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}", mdir.display());
    // println!("cargo:rustc-link-search=/home/ubuntu/git/plonky2-okx/plonky2/merkle_avx512");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    // println!("cargo:rustc-link-lib=cmerkle-poseidon-rust");
    println!("cargo:rustc-link-lib=static=merkle_avx512");
    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=stdc++");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed={}/merkle.h", mdir.display());

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("merkle_avx512/merkle.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from("merkle_avx512");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn poseidon_bn128() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let poseidon_dir = Path::new(&dir).join("poseidon_bn128");

    if cfg!(target_os = "macos") {
        println!("target os is macos");
        let output = std::process::Command::new("go")
            .arg("version")
            .output()
            .expect("Failed to run command");

        // Check the exit status
        if output.status.success() {
            // Go is installed
            println!(
                "Go is installed: {:?}",
                String::from_utf8_lossy(&output.stdout)
            );
        } else {
            // Go is not installed
            panic!("Go is not installed");
        }
        let poseidon_c_dir = Path::new(&dir).join("go-iden3-crypto");
        println!("poseidon_c_dir: {:?}", poseidon_c_dir);
        if poseidon_c_dir.exists() {
            std::process::Command::new("sh")
                .arg("-c")
                .arg("rm")
                .arg("-rf")
                .arg(poseidon_c_dir.clone())
                .output()
                .expect("rm go-iden3-crypto failure");
        }
        println!("start clone go iden3");
        std::process::Command::new("git")
            .arg("clone")
            .arg("https://github.com/polymerdao/go-iden3-crypto.git")
            .arg(poseidon_c_dir.clone())
            .output()
            .expect("clone go iden3 crypto failure");
        println!("end clone go iden3");

        let ret = std::process::Command::new("sh")
            .arg("-c")
            .arg("./compile.sh")
            .current_dir(poseidon_c_dir.clone().join("poseidon-permute-c"))
            .output()
            .expect("compile poseidon permute c failure");
        println!("compile lib ret: {:?}", ret);

        std::process::Command::new("mv")
            .arg("libposeidon-permute-c.a")
            .arg(poseidon_dir.join("libposeidon-permute-c-mac.a"))
            .current_dir(poseidon_c_dir.clone().join("poseidon-permute-c"))
            .output()
            .expect("mv failure");
    }
    println!("cargo:rustc-link-search=native={}", poseidon_dir.display());

    if cfg!(target_os = "macos") {
        println!("link to mac lib");
        println!("cargo:rustc-link-lib=static=poseidon-permute-c-mac");
    } else {
        println!("link to linux lib");
        println!("cargo:rustc-link-lib=static=poseidon-permute-c");
    }

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

fn main() {
    poseidon_bn128();
    merkle_avx512();
}
