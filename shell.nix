let

  # Pin to a specific nixpkgs commit for reproducibility.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/4684fd6b0c01e4b7d99027a34c93c2e09ecafee2.tar.gz")
  {
    config.allowUnfree = true; 
  };

in pkgs.mkShell {

  nativeBuildInputs = [
    pkgs.python312
    pkgs.uv
  ];

  shellHook = ''
    echo "NIX Environment with CUDA availability"
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/:$LD_LIBRARY_PATH
    echo $CUDA_PATH
    echo $LD_LIBRARY_PATH
  ''; 
}
