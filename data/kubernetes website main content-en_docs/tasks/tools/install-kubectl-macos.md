---
reviewers
- mikedanese
title Install and Set Up kubectl on macOS
content_type task
weight 10
---

# #  heading prerequisites

You must use a kubectl version that is within one minor version difference of
your cluster. For example, a v client can communicate
with v, v,
and v control planes.
Using the latest compatible version of kubectl helps avoid unforeseen issues.

# # Install kubectl on macOS

The following methods exist for installing kubectl on macOS

- [Install kubectl on macOS](#install-kubectl-on-macos)
  - [Install kubectl binary with curl on macOS](#install-kubectl-binary-with-curl-on-macos)
  - [Install with Homebrew on macOS](#install-with-homebrew-on-macos)
  - [Install with Macports on macOS](#install-with-macports-on-macos)
- [Verify kubectl configuration](#verify-kubectl-configuration)
- [Optional kubectl configurations and plugins](#optional-kubectl-configurations-and-plugins)
  - [Enable shell autocompletion](#enable-shell-autocompletion)
  - [Install `kubectl convert` plugin](#install-kubectl-convert-plugin)

# # # Install kubectl binary with curl on macOS

1. Download the latest release

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)bindarwinamd64kubectl

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)bindarwinarm64kubectl

   To download a specific version, replace the `(curl -L -s httpsdl.k8s.ioreleasestable.txt)`
   portion of the command with the specific version.

   For example, to download version  on Intel macOS, type

   ```bash
   curl -LO httpsdl.k8s.ioreleasevbindarwinamd64kubectl
   ```

   And for macOS on Apple Silicon, type

   ```bash
   curl -LO httpsdl.k8s.ioreleasevbindarwinarm64kubectl
   ```

1. Validate the binary (optional)

   Download the kubectl checksum file

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)bindarwinamd64kubectl.sha256

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)bindarwinarm64kubectl.sha256

   Validate the kubectl binary against the checksum file

   ```bash
   echo (cat kubectl.sha256)  kubectl  shasum -a 256 --check
   ```

   If valid, the output is

   ```console
   kubectl OK
   ```

   If the check fails, `shasum` exits with nonzero status and prints output similar to

   ```console
   kubectl FAILED
   shasum WARNING 1 computed checksum did NOT match
   ```

   Download the same version of the binary and checksum.

1. Make the kubectl binary executable.

   ```bash
   chmod x .kubectl
   ```

1. Move the kubectl binary to a file location on your system `PATH`.

   ```bash
   sudo mv .kubectl usrlocalbinkubectl
   sudo chown root usrlocalbinkubectl
   ```

   Make sure `usrlocalbin` is in your PATH environment variable.

1. Test to ensure the version you installed is up-to-date

   ```bash
   kubectl version --client
   ```

   Or use this for detailed view of version

   ```cmd
   kubectl version --client --outputyaml
   ```

1. After installing and validating kubectl, delete the checksum file

   ```bash
   rm kubectl.sha256
   ```

# # # Install with Homebrew on macOS

If you are on macOS and using [Homebrew](httpsbrew.sh) package manager,
you can install kubectl with Homebrew.

1. Run the installation command

   ```bash
   brew install kubectl
   ```

   or

   ```bash
   brew install kubernetes-cli
   ```

1. Test to ensure the version you installed is up-to-date

   ```bash
   kubectl version --client
   ```

# # # Install with Macports on macOS

If you are on macOS and using [Macports](httpsmacports.org) package manager,
you can install kubectl with Macports.

1. Run the installation command

   ```bash
   sudo port selfupdate
   sudo port install kubectl
   ```

1. Test to ensure the version you installed is up-to-date

   ```bash
   kubectl version --client
   ```

# # Verify kubectl configuration

# # Optional kubectl configurations and plugins

# # # Enable shell autocompletion

kubectl provides autocompletion support for Bash, Zsh, Fish, and PowerShell
which can save you a lot of typing.

Below are the procedures to set up autocompletion for Bash, Fish, and Zsh.

# # # Configure kuberc

See [kuberc](docsreferencekubectlkuberc) for more information.

# # # Install `kubectl convert` plugin

1. Download the latest release with the command

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)bindarwinamd64kubectl-convert

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)bindarwinarm64kubectl-convert

1. Validate the binary (optional)

   Download the kubectl-convert checksum file

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)bindarwinamd64kubectl-convert.sha256

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)bindarwinarm64kubectl-convert.sha256

   Validate the kubectl-convert binary against the checksum file

   ```bash
   echo (cat kubectl-convert.sha256)  kubectl-convert  shasum -a 256 --check
   ```

   If valid, the output is

   ```console
   kubectl-convert OK
   ```

   If the check fails, `shasum` exits with nonzero status and prints output similar to

   ```console
   kubectl-convert FAILED
   shasum WARNING 1 computed checksum did NOT match
   ```

   Download the same version of the binary and checksum.

1. Make kubectl-convert binary executable

   ```bash
   chmod x .kubectl-convert
   ```

1. Move the kubectl-convert binary to a file location on your system `PATH`.

   ```bash
   sudo mv .kubectl-convert usrlocalbinkubectl-convert
   sudo chown root usrlocalbinkubectl-convert
   ```

   Make sure `usrlocalbin` is in your PATH environment variable.

1. Verify plugin is successfully installed

   ```shell
   kubectl convert --help
   ```

   If you do not see an error, it means the plugin is successfully installed.

1. After installing the plugin, clean up the installation files

   ```bash
   rm kubectl-convert kubectl-convert.sha256
   ```

# # # Uninstall kubectl on macOS

Depending on how you installed `kubectl`, use one of the following methods.

# # # Uninstall kubectl using the command-line

1.  Locate the `kubectl` binary on your system

    ```bash
    which kubectl
    ```

1.  Remove the `kubectl` binary

    ```bash
    sudo rm
    ```
    Replace `` with the path to the `kubectl` binary from the previous step. For example, `sudo rm usrlocalbinkubectl`.

# # # Uninstall kubectl using homebrew

If you installed `kubectl` using Homebrew, run the following command

```bash
brew remove kubectl
```

# #  heading whatsnext
