{ pkgs, lib, config, inputs, ... }:

let
  scli = pkgs.stdenv.mkDerivation {
    pname = "scli";
    version = "0.2.4";

    src = pkgs.fetchurl {
      url = "https://github.com/scallop-lang/scallop/releases/download/0.2.4/scli-0.2.4-linux-x86_64";
      sha256 = "jF7Ib82w29VWmO/3VwrHOW0LCHjmASB/ho1h+dZIK5o=";
    };

    phases = [ "installPhase" ];

    installPhase = ''
      install -m755 -D $src $out/bin/scli
    '';
  };

  sclrepl = pkgs.stdenv.mkDerivation {
    pname = "sclrepl";
    version = "0.2.4";

    src = pkgs.fetchurl {
      url = "https://github.com/scallop-lang/scallop/releases/download/0.2.4/sclrepl-0.2.4-linux-x86_64";
      sha256 = "AROIyz7Gh73UM/mJkEPBg5K1qtZ4lGhwwDsJql5Ntus=";
    };

    phases = [ "installPhase" ];

    installPhase = ''
      install -m755 -D $src $out/bin/sclrepl
    '';
  };

in
{
  languages.python = {
    enable = true;
    version = "3.10"; # scallopy requires python 3.10 due to cp310
    venv.enable = true;
    uv.enable = true;
    uv.sync.enable = true;
  };

  env = {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
      pkgs.zlib
      pkgs.libGL
      pkgs.libGLU
      pkgs.wlroots
      pkgs.ncurses5
    ] + ":/run/opengl-driver/lib";
    NIX_LD = lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
    
    CUDA_PATH = "${pkgs.cudatoolkit}";
    EXTRA_LDFLAGS = "-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib";
    EXTRA_CCFLAGS = "-I/usr/include";
  };

  packages = with pkgs; [
    scli
    sclrepl
    zlib
    libGL
    libGLU
    wlroots
    ncurses5
  ];
}
