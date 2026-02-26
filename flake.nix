{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    jail-nix.url = "sourcehut:~alexdavid/jail.nix";
    jailed-agents.url = "github:btmxh/jailed-agents";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      systems,
      jail-nix,
      jailed-agents,
      git-hooks,
      ...
    }:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          inherit (self.checks.${system}.pre-commit-check) shellHook enabledPackages config;
          inherit (config) package configFile;
          jail = jail-nix.lib.init pkgs;
        in
        {
          default = pkgs.mkShell {
            buildInputs = enabledPackages;
            shellHook = shellHook + ''
              export PATH="$PWD/bin:$PATH"
            '';
            packages =
              with pkgs;
              [
                ruff
                ty

                rustc
                cargo
                rustfmt
                clippy

                nixd
                nixfmt-rfc-style
              ]
              ++ (builtins.attrValues (
                jailed-agents.lib.${system}.makeJailedAgents {
                  extraPkgs = [
                    ruff
                    ty
                    nixfmt-rfc-style
                    package
                  ]
                  ++ enabledPackages;

                  extraJailOptions = with jail.combinators; [
                    (readonly configFile)
                    (readonly (lib.getExe package))
                  ];
                }
              ));
          };
        }
      );

      # Run the hooks with `nix fmt`.
      formatter = forEachSystem (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          inherit (self.checks.${system}.pre-commit-check) config;
          inherit (config) package configFile;
          script = ''
            "${pkgs.lib.getExe package}" run --all-files --config ${configFile}
          '';
        in
        pkgs.writeShellScriptBin "pre-commit-run" script
      );

      # Run the hooks in a sandbox with `nix flake check`.
      # Read-only filesystem and no internet access.
      checks = forEachSystem (system: {
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            nixfmt.enable = true;
            statix.enable = true;
            check-yaml.enable = true;
            end-of-file-fixer.enable = true;
            trim-trailing-whitespace.enable = true;
            ruff.enable = true;
            ruff-format.enable = true;
            rustfmt.enable = true;
            check-toml.enable = true;
            taplo.enable = true;
          };
        };
      });
    };
}
