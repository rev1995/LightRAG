---
reviewers
- mikedanese
title Install and Set Up kubectl on Linux
content_type task
weight 10
---

# #  heading prerequisites

You must use a kubectl version that is within one minor version difference of
your cluster. For example, a v client can communicate
with v, v,
and v control planes.
Using the latest compatible version of kubectl helps avoid unforeseen issues.

# # Install kubectl on Linux

The following methods exist for installing kubectl on Linux

- [Install kubectl binary with curl on Linux](#install-kubectl-binary-with-curl-on-linux)
- [Install using native package management](#install-using-native-package-management)
- [Install using other package management](#install-using-other-package-management)

# # # Install kubectl binary with curl on Linux

1. Download the latest release with the command

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxamd64kubectl

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxarm64kubectl

   To download a specific version, replace the `(curl -L -s httpsdl.k8s.ioreleasestable.txt)`
   portion of the command with the specific version.

   For example, to download version  on Linux x86-64, type

   ```bash
   curl -LO httpsdl.k8s.ioreleasevbinlinuxamd64kubectl
   ```

   And for Linux ARM64, type

   ```bash
   curl -LO httpsdl.k8s.ioreleasevbinlinuxarm64kubectl
   ```

1. Validate the binary (optional)

   Download the kubectl checksum file

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxamd64kubectl.sha256

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxarm64kubectl.sha256

   Validate the kubectl binary against the checksum file

   ```bash
   echo (cat kubectl.sha256)  kubectl  sha256sum --check
   ```

   If valid, the output is

   ```console
   kubectl OK
   ```

   If the check fails, `sha256` exits with nonzero status and prints output similar to

   ```console
   kubectl FAILED
   sha256sum WARNING 1 computed checksum did NOT match
   ```

   Download the same version of the binary and checksum.

1. Install kubectl

   ```bash
   sudo install -o root -g root -m 0755 kubectl usrlocalbinkubectl
   ```

   If you do not have root access on the target system, you can still install
   kubectl to the `.localbin` directory

   ```bash
   chmod x kubectl
   mkdir -p .localbin
   mv .kubectl .localbinkubectl
   # and then append (or prepend) .localbin to PATH
   ```

1. Test to ensure the version you installed is up-to-date

   ```bash
   kubectl version --client
   ```

   Or use this for detailed view of version

   ```cmd
   kubectl version --client --outputyaml
   ```

# # # Install using native package management

 tab nameDebian-based distributions

1. Update the `apt` package index and install packages needed to use the Kubernetes `apt` repository

   ```shell
   sudo apt-get update
   # apt-transport-https may be a dummy package if so, you can skip that package
   sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
   ```

2. Download the public signing key for the Kubernetes package repositories. The same signing key is used for all repositories so you can disregard the version in the URL

   ```shell
   # If the folder `etcaptkeyrings` does not exist, it should be created before the curl command, read the note below.
   # sudo mkdir -p -m 755 etcaptkeyrings
   curl -fsSL httpspkgs.k8s.iocorestabledebRelease.key  sudo gpg --dearmor -o etcaptkeyringskubernetes-apt-keyring.gpg
   sudo chmod 644 etcaptkeyringskubernetes-apt-keyring.gpg # allow unprivileged APT programs to read this keyring
   ```

In releases older than Debian 12 and Ubuntu 22.04, folder `etcaptkeyrings` does not exist by default, and it should be created before the curl command.

3. Add the appropriate Kubernetes `apt` repository. If you want to use Kubernetes version different than ,
   replace  with the desired minor version in the command below

   ```shell
   # This overwrites any existing configuration in etcaptsources.list.dkubernetes.list
   echo deb [signed-byetcaptkeyringskubernetes-apt-keyring.gpg] httpspkgs.k8s.iocorestabledeb   sudo tee etcaptsources.list.dkubernetes.list
   sudo chmod 644 etcaptsources.list.dkubernetes.list   # helps tools such as command-not-found to work correctly
   ```

To upgrade kubectl to another minor release, youll need to bump the version in `etcaptsources.list.dkubernetes.list` before running `apt-get update` and `apt-get upgrade`. This procedure is described in more detail in [Changing The Kubernetes Package Repository](docstasksadminister-clusterkubeadmchange-package-repository).

4. Update `apt` package index, then install kubectl

   ```shell
   sudo apt-get update
   sudo apt-get install -y kubectl
   ```

 tab

 tab nameRed Hat-based distributions

1. Add the Kubernetes `yum` repository. If you want to use Kubernetes version
   different than , replace  with
   the desired minor version in the command below.

   ```bash
   # This overwrites any existing configuration in etcyum.repos.dkubernetes.repo
   cat rpm
   enabled1
   gpgcheck1
   gpgkeyhttpspkgs.k8s.iocorestablerpmrepodatarepomd.xml.key
   EOF
   ```

To upgrade kubectl to another minor release, youll need to bump the version in `etcyum.repos.dkubernetes.repo` before running `yum update`. This procedure is described in more detail in [Changing The Kubernetes Package Repository](docstasksadminister-clusterkubeadmchange-package-repository).

2. Install kubectl using `yum`

   ```bash
   sudo yum install -y kubectl
   ```

 tab

 tab nameSUSE-based distributions

1. Add the Kubernetes `zypper` repository. If you want to use Kubernetes version
   different than , replace  with
   the desired minor version in the command below.

   ```bash
   # This overwrites any existing configuration in etczypprepos.dkubernetes.repo
   cat rpm
   enabled1
   gpgcheck1
   gpgkeyhttpspkgs.k8s.iocorestablerpmrepodatarepomd.xml.key
   EOF
   ```

To upgrade kubectl to another minor release, youll need to bump the version in `etczypprepos.dkubernetes.repo`
before running `zypper update`. This procedure is described in more detail in
[Changing The Kubernetes Package Repository](docstasksadminister-clusterkubeadmchange-package-repository).

2. Update `zypper` and confirm the new repo addition

   ```bash
   sudo zypper update
   ```

   When this message appears, press t or a

   ```
   New repository or package signing key received

   Repository       Kubernetes
   Key Fingerprint  1111 2222 3333 4444 5555 6666 7777 8888 9999 AAAA
   Key Name         isvkubernetes OBS Project
   Key Algorithm    RSA 2048
   Key Created      Thu 25 Aug 2022 012111 PM -03
   Key Expires      Sat 02 Nov 2024 012111 PM -03 (expires in 85 days)
   Rpm Name         gpg-pubkey-9a296436-6307a177

   Note Signing data enables the recipient to verify that no modifications occurred after the data
   were signed. Accepting data with no, wrong or unknown signature can lead to a corrupted system
   and in extreme cases even to a system compromise.

   Note A GPG pubkey is clearly identified by its fingerprint. Do not rely on the keys name. If
   you are not sure whether the presented key is authentic, ask the repository provider or check
   their web site. Many providers maintain a web page showing the fingerprints of the GPG keys they
   are using.

   Do you want to reject the key, trust temporarily, or trust always [rta] (r) a
   ```

3. Install kubectl using `zypper`

   ```bash
   sudo zypper install -y kubectl
   ```

 tab

# # # Install using other package management

 tab nameSnap
If you are on Ubuntu or another Linux distribution that supports the
[snap](httpssnapcraft.iodocscoreinstall) package manager, kubectl
is available as a [snap](httpssnapcraft.io) application.

```shell
snap install kubectl --classic
kubectl version --client
```

 tab

 tab nameHomebrew
If you are on Linux and using [Homebrew](httpsdocs.brew.shHomebrew-on-Linux)
package manager, kubectl is available for [installation](httpsdocs.brew.shHomebrew-on-Linux#install).

```shell
brew install kubectl
kubectl version --client
```

 tab

# # Verify kubectl configuration

# # Optional kubectl configurations and plugins

# # # Enable shell autocompletion

kubectl provides autocompletion support for Bash, Zsh, Fish, and PowerShell,
which can save you a lot of typing.

Below are the procedures to set up autocompletion for Bash, Fish, and Zsh.

# # # Configure kuberc

See [kuberc](docsreferencekubectlkuberc) for more information.

# # # Install `kubectl convert` plugin

1. Download the latest release with the command

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxamd64kubectl-convert

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxarm64kubectl-convert

1. Validate the binary (optional)

   Download the kubectl-convert checksum file

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxamd64kubectl-convert.sha256

   curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxarm64kubectl-convert.sha256

   Validate the kubectl-convert binary against the checksum file

   ```bash
   echo (cat kubectl-convert.sha256) kubectl-convert  sha256sum --check
   ```

   If valid, the output is

   ```console
   kubectl-convert OK
   ```

   If the check fails, `sha256` exits with nonzero status and prints output similar to

   ```console
   kubectl-convert FAILED
   sha256sum WARNING 1 computed checksum did NOT match
   ```

   Download the same version of the binary and checksum.

1. Install kubectl-convert

   ```bash
   sudo install -o root -g root -m 0755 kubectl-convert usrlocalbinkubectl-convert
   ```

1. Verify plugin is successfully installed

   ```shell
   kubectl convert --help
   ```

   If you do not see an error, it means the plugin is successfully installed.

1. After installing the plugin, clean up the installation files

   ```bash
   rm kubectl-convert kubectl-convert.sha256
   ```

# #  heading whatsnext
