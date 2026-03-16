fn main() {
    #[cfg(feature = "grpc")]
    {
        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        tonic_build::configure()
            .build_server(true)
            .build_client(false)
            .file_descriptor_set_path(out_dir.join("helix_descriptor.bin"))
            .compile_protos(&["proto/helix.proto"], &["proto"])
            .expect("Failed to compile protobuf");
    }
}
