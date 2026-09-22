{
  lib,
  rustPlatform,
  pkg-config,
  openssl,
  onnxruntime,
  makeWrapper,
  stdenv,
}:
rustPlatform.buildRustPackage {
  pname = "memex";
  version = (lib.importTOML ../Cargo.toml).package.version;

  src = lib.cleanSource ../.;

  cargoLock = {
    lockFile = ../Cargo.lock;
  };

  buildNoDefaultFeatures = true;
  buildFeatures = [ "ort-load-dynamic" ];

  nativeBuildInputs = [
    pkg-config
    makeWrapper
  ];

  buildInputs = [
    openssl
  ];

  # ONNX Runtime is dlopen'd at runtime (fastembed `ort-load-dynamic`), so the
  # binary needs the dylib's path on ORT_DYLIB_PATH.
  postInstall = ''
    wrapProgram $out/bin/memex \
      --set ORT_DYLIB_PATH ${onnxruntime}/lib/libonnxruntime${stdenv.hostPlatform.extensions.sharedLibrary}
  '';

  # Tests require network access to download embedding models
  doCheck = false;

  meta = {
    description = "Fast local history search for local agent logs";
    homepage = "https://github.com/nicosuave/memex";
    license = lib.licenses.mit;
    mainProgram = "memex";
    maintainers = [];
  };
}
