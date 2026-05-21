{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
  };

  outputs = {self, nixpkgs, fenix, crane}: let 
    pkgs = nixpkgs.legacyPackages."x86_64-linux";
    fenixLib = fenix.packages."x86_64-linux";
    rustToolchain = fenixLib.fromToolchainFile {
      file = ./rust-toolchain.toml;
      sha256 = "sha256-gh/xTkxKHL4eiRXzWv8KP7vfjSk61Iq48x47BEDFgfk=";
    };
    neededPackages = with pkgs; [
      
    ];
    packages_for_target = {
      target, toolchain
    }: ((crane.mkLib nixpkgs.legacyPackages.${pkgs.stdenv.hostPlatform.system}).overrideToolchain toolchain).buildPackage ({
      name =  "sir_animal_package";
      CARGO_BUILD_TARGET = target;
      src = ./.; # ./.
      buildInputs = neededPackages; # Runtime dependencies (Naming is funky) - also available at compile time
      nativeBuildInputs = with pkgs; [gnum4]; # Stuff we need at compile time only
      cargoHash = "";

      doCheck = true; # Do the unit tests
    });
  in  
  {
    packages."x86_64-linux".default = packages_for_target {target= "x86_64-unknown-linux-gnu"; toolchain = rustToolchain;};
  };
}