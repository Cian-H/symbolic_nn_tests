{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };

  inputs.nixpkgs.url = "github:NixOS/nixpkgs";

  outputs = { self, nixpkgs, utils }: utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in
    {
      devShell = pkgs.mkShell
        {
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            pkgs.zlib
            pkgs.libGL
            pkgs.libGLU
            pkgs.wlroots
            pkgs.ncurses5
            pkgs.linuxKernel.packages.linux_latest_libre.nvidia_x11
          ];
          NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";

          buildInputs = with pkgs; [
            ruff
            ruff-lsp
            rye
            uv
          ];

          shellHook = ''
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=$(nix eval --raw nixpkgs#addOpenGLRunpath.driverLink)/lib
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"
          '';
        };
    }
  );
}

