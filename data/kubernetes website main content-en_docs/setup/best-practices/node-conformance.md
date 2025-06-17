---
reviewers
- Random-Liu
title Validate node setup
weight 30
---

# # Node Conformance Test

*Node conformance test* is a containerized test framework that provides a system
verification and functionality test for a node. The test validates whether the
node meets the minimum requirements for Kubernetes a node that passes the test
is qualified to join a Kubernetes cluster.

# # Node Prerequisite

To run node conformance test, a node must satisfy the same prerequisites as a
standard Kubernetes node. At a minimum, the node should have the following
daemons installed

* CRI-compatible container runtimes such as Docker, containerd and CRI-O
* kubelet

# # Running Node Conformance Test

To run the node conformance test, perform the following steps

1. Work out the value of the `--kubeconfig` option for the kubelet for example
   `--kubeconfigvarlibkubeletconfig.yaml`.
    Because the test framework starts a local control plane to test the kubelet,
    use `httplocalhost8080` as the URL of the API server.
    There are some other kubelet command line parameters you may want to use

   * `--cloud-provider` If you are using `--cloud-providergce`, you should
     remove the flag to run the test.

1. Run the node conformance test with command

   ```shell
   # CONFIG_DIR is the pod manifest path of your kubelet.
   # LOG_DIR is the test output path.
   sudo docker run -it --rm --privileged --nethost
     -v rootfs -v CONFIG_DIRCONFIG_DIR -v LOG_DIRvarresult
     registry.k8s.ionode-test0.2
   ```

# # Running Node Conformance Test for Other Architectures

Kubernetes also provides node conformance test docker images for other
architectures

  Arch         Image
-------------------------
  amd64   node-test-amd64
  arm      node-test-arm
 arm64    node-test-arm64

# # Running Selected Test

To run specific tests, overwrite the environment variable `FOCUS` with the
regular expression of tests you want to run.

```shell
sudo docker run -it --rm --privileged --nethost
  -v rootfsro -v CONFIG_DIRCONFIG_DIR -v LOG_DIRvarresult
  -e FOCUSMirrorPod  # Only run MirrorPod test
  registry.k8s.ionode-test0.2
```

To skip specific tests, overwrite the environment variable `SKIP` with the
regular expression of tests you want to skip.

```shell
sudo docker run -it --rm --privileged --nethost
  -v rootfsro -v CONFIG_DIRCONFIG_DIR -v LOG_DIRvarresult
  -e SKIPMirrorPod  # Run all conformance tests but skip MirrorPod test
  registry.k8s.ionode-test0.2
```

Node conformance test is a containerized version of
[node e2e test](httpsgithub.comkubernetescommunityblobmastercontributorsdevelsig-nodee2e-node-tests.md).
By default, it runs all conformance tests.

Theoretically, you can run any node e2e test if you configure the container and
mount required volumes properly. But **it is strongly recommended to only run conformance
test**, because it requires much more complex configuration to run non-conformance test.

# # Caveats

* The test leaves some docker images on the node, including the node conformance
  test image and images of containers used in the functionality
  test.
* The test leaves dead containers on the node. These containers are created
  during the functionality test.
