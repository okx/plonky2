use std::env;
use std::fs::write;
use std::path::{Path};
#[cfg(feature = "precompile")]
use std::Path::{PathBuf};
use std::process::Command;
#[cfg(feature = "precompile")]
use std::str::FromStr;

use anyhow::Context;
#[cfg(feature = "precompile")]
use plonky2_util::pre_compute::{get_pre_compute_size, PRE_COMPUTE_END, PRE_COMPUTE_START};
use proc_macro2::TokenStream;
#[cfg(feature = "precompile")]
use quote::quote;
#[cfg(feature = "precompile")]
use syn::Lit;

#[cfg(feature = "precompile")]
fn build_precompile() {
    println!("cargo:rerun-if-changed=generated");
    let cargo_manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let path: PathBuf = PathBuf::from_str(&cargo_manifest_dir)
        .unwrap()
        .join("generated/goldilock_root_of_unity.rs");

    let token_stream = build_token_stream(&path).expect("build token stream error");
    _ = write_generated_file(token_stream, "goldilock_root_of_unity.rs");
}
fn main() {

    #[cfg(feature = "precompile")]
    build_precompile();
}

#[cfg(feature = "precompile")]
fn build_token_stream(path: &PathBuf) -> anyhow::Result<TokenStream> {
    let size = get_pre_compute_size(PRE_COMPUTE_START, PRE_COMPUTE_END);
    let token = syn::parse_str::<Lit>(&format!("{}", size)).unwrap();
    if path.exists() {
        let stream: proc_macro2::TokenStream =
            format!("\"{}\"", path.to_str().unwrap()).parse().unwrap();

        Ok(quote! {
            pub static PRE_COMPILED: [u64; #token] =

            unsafe {
               include!(
                #stream
                )
            };

        })
    } else {
        Ok(quote! {
            pub static PRE_COMPILED: [u64; #token] = [0; #token];

        })
    }
}

pub fn write_generated_file(content: TokenStream, out_file: &str) -> anyhow::Result<()> {
    let out_dir = env::var_os("OUT_DIR").context("failed to get OUT_DIR env var")?;
    let path = Path::new(&out_dir).join(out_file);
    let code = content.to_string();

    _ = write(&path, code);

    // Try to format the output for debugging purposes.
    // Doesn't matter if rustfmt is unavailable.
    let _ = Command::new("rustfmt").arg(path).output();

    Ok(())
}
