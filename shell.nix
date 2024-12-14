{ pkgs ? import <nixpkgs> { } }:
(pkgs.buildFHSEnv {
  name = "python-env";
  targetPkgs = pkgs: (with pkgs; [
    python312
    libGL
    xorg.libX11
    zlib
  ]);
  runScript = pkgs.writeScript "init.sh" ''
    python -m venv venv
    source venv/bin/activate
    exec bash
  '';
}).env
