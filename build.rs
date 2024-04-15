use std::process::Command;

fn main() {
    // Run `git submodule update --init --recursive`
    Command::new("git")
        .args(&["submodule", "update", "--init", "--recursive"])
        .status()
        .expect("Failed to update submodules");
}