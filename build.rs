fn main() {
    #[cfg(feature = "blas")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
