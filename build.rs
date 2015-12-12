extern crate gcc;

fn main() {
  gcc::Config::new()
    .compiler("/usr/local/cuda/bin/nvcc")
    .opt_level(3)
    // FIXME(20151207): for working w/ K80.
    //.flag("-arch=sm_37")
    .flag("-arch=sm_50")
    .flag("-Xcompiler")
    .flag("\'-fPIC\'")
    .include("src/cu")
    .include("/usr/local/cuda/include")
    .file("src/cu/map_kernels.cu")
    .compile("libarray_cuda_kernels.a");

  println!("cargo:rustc-flags=-L /usr/local/cuda/lib64");
}
