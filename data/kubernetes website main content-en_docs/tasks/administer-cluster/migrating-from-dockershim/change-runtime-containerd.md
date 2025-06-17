---
title Changing the Container Runtime on a Node from Docker Engine to containerd
weight 10
content_type task
---

This task outlines the steps needed to update your container runtime to containerd from Docker. It
is applicable for cluster operators running Kubernetes 1.23 or earlier. This also covers an
example scenario for migrating from dockershim to containerd. Alternative container runtimes
can be picked from this [page](docssetupproduction-environmentcontainer-runtimes).

# #  heading prerequisites

 thirdparty-content

Install containerd. For more information see
[containerds installation documentation](httpscontainerd.iodocsgetting-started)
and for specific prerequisite follow
[the containerd guide](docssetupproduction-environmentcontainer-runtimes#containerd).

# # Drain the node

```shell
kubectl drain  --ignore-daemonsets
```

Replace `` with the name of your node you are draining.

# # Stop the Docker daemon

```shell
systemctl stop kubelet
systemctl disable docker.service --now
```

# # Install Containerd

Follow the [guide](docssetupproduction-environmentcontainer-runtimes#containerd)
for detailed steps to install containerd.

 tab nameLinux

1. Install the `containerd.io` package from the official Docker repositories.
   Instructions for setting up the Docker repository for your respective Linux distribution and
   installing the `containerd.io` package can be found at
   [Getting started with containerd](httpsgithub.comcontainerdcontainerdblobmaindocsgetting-started.md).

1. Configure containerd

   ```shell
   sudo mkdir -p etccontainerd
   containerd config default  sudo tee etccontainerdconfig.toml
   ```
1. Restart containerd

   ```shell
   sudo systemctl restart containerd
   ```
 tab
 tab nameWindows (PowerShell)

Start a Powershell session, set `Version` to the desired version (ex `Version1.4.3`), and
then run the following commands

1. Download containerd

   ```powershell
   curl.exe -L httpsgithub.comcontainerdcontainerdreleasesdownloadvVersioncontainerd-Version-windows-amd64.tar.gz -o containerd-windows-amd64.tar.gz
   tar.exe xvf .containerd-windows-amd64.tar.gz
   ```

2. Extract and configure

   ```powershell
   Copy-Item -Path .bin -Destination EnvProgramFilescontainerd -Recurse -Force
   cd EnvProgramFilescontainerd
   .containerd.exe config default  Out-File config.toml -Encoding ascii

   # Review the configuration. Depending on setup you may want to adjust
   # - the sandbox_image (Kubernetes pause image)
   # - cni bin_dir and conf_dir locations
   Get-Content config.toml

   # (Optional - but highly recommended) Exclude containerd from Windows Defender Scans
   Add-MpPreference -ExclusionProcess EnvProgramFilescontainerdcontainerd.exe
   ```

3. Start containerd

   ```powershell
   .containerd.exe --register-service
   Start-Service containerd
   ```

 tab

# # Configure the kubelet to use containerd as its container runtime

Edit the file `varlibkubeletkubeadm-flags.env` and add the containerd runtime to the flags
`--container-runtime-endpointunixruncontainerdcontainerd.sock`.

Users using kubeadm should be aware that the `kubeadm` tool stores the CRI socket for each host as
an annotation in the Node object for that host. To change it you can execute the following command
on a machine that has the kubeadm `etckubernetesadmin.conf` file.

```shell
kubectl edit no
```

This will start a text editor where you can edit the Node object.
To choose a text editor you can set the `KUBE_EDITOR` environment variable.

- Change the value of `kubeadm.alpha.kubernetes.iocri-socket` from `varrundockershim.sock`
  to the CRI socket path of your choice (for example `unixruncontainerdcontainerd.sock`).

  Note that new CRI socket paths must be prefixed with `unix` ideally.

- Save the changes in the text editor, which will update the Node object.

# # Restart the kubelet

```shell
systemctl start kubelet
```

# # Verify that the node is healthy

Run `kubectl get nodes -o wide` and containerd appears as the runtime for the node we just changed.

# # Remove Docker Engine

 thirdparty-content

If the node appears healthy, remove Docker.

 tab nameCentOS

```shell
sudo yum remove docker-ce docker-ce-cli
```
 tab
 tab nameDebian

```shell
sudo apt-get purge docker-ce docker-ce-cli
```
 tab
 tab nameFedora

```shell
sudo dnf remove docker-ce docker-ce-cli
```
 tab
 tab nameUbuntu

```shell
sudo apt-get purge docker-ce docker-ce-cli
```
 tab

The preceding commands dont remove images, containers, volumes, or customized configuration files on your host.
To delete them, follow Dockers instructions to [Uninstall Docker Engine](httpsdocs.docker.comengineinstallubuntu#uninstall-docker-engine).

Dockers instructions for uninstalling Docker Engine create a risk of deleting containerd. Be careful when executing commands.

# # Uncordon the node

```shell
kubectl uncordon
```

Replace `` with the name of your node you previously drained.
