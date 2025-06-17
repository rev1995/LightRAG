---
title Fine Parallel Processing Using a Work Queue
content_type task
weight 30
---

In this example, you will run a Kubernetes Job that runs multiple parallel
tasks as worker processes, each running as a separate Pod.

In this example, as each pod is created, it picks up one unit of work
from a task queue, processes it, and repeats until the end of the queue is reached.

Here is an overview of the steps in this example

1. **Start a storage service to hold the work queue.**  In this example, you will use Redis to store
   work items.  In the [previous example](docstasksjobcoarse-parallel-processing-work-queue),
   you used RabbitMQ.  In this example, you will use Redis and a custom work-queue client library
   this is because AMQP does not provide a good way for clients to
   detect when a finite-length work queue is empty.  In practice you would set up a store such
   as Redis once and reuse it for the work queues of many jobs, and other things.
1. **Create a queue, and fill it with messages.**  Each message represents one task to be done.  In
   this example, a message is an integer that we will do a lengthy computation on.
1. **Start a Job that works on tasks from the queue**.  The Job starts several pods.  Each pod takes
   one task from the message queue, processes it, and repeats until the end of the queue is reached.

# #  heading prerequisites

You will need a container image registry where you can upload images to run in your cluster.
The example uses [Docker Hub](httpshub.docker.com), but you could adapt it to a different
container image registry.

This task example also assumes that you have Docker installed locally. You use Docker to
build container images.

Be familiar with the basic,
non-parallel, use of [Job](docsconceptsworkloadscontrollersjob).

# # Starting Redis

For this example, for simplicity, you will start a single instance of Redis.
See the [Redis Example](httpsgithub.comkubernetesexamplestreemasterguestbook) for an example
of deploying Redis scalably and redundantly.

You could also download the following files directly

- [`redis-pod.yaml`](examplesapplicationjobredisredis-pod.yaml)
- [`redis-service.yaml`](examplesapplicationjobredisredis-service.yaml)
- [`Dockerfile`](examplesapplicationjobredisDockerfile)
- [`job.yaml`](examplesapplicationjobredisjob.yaml)
- [`rediswq.py`](examplesapplicationjobredisrediswq.py)
- [`worker.py`](examplesapplicationjobredisworker.py)

To start a single instance of Redis, you need to create the redis pod and redis service

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationjobredisredis-pod.yaml
kubectl apply -f httpsk8s.ioexamplesapplicationjobredisredis-service.yaml
```

# # Filling the queue with tasks

Now lets fill the queue with some tasks.  In this example, the tasks are strings to be
printed.

Start a temporary interactive pod for running the Redis CLI.

```shell
kubectl run -i --tty temp --image redis --command binsh
```
```
Waiting for pod defaultredis2-c7h78 to be running, status is Pending, pod ready false
Hit enter for command prompt
```

Now hit enter, start the Redis CLI, and create a list with some work items in it.

```shell
redis-cli -h redis
```
```console
redis6379 rpush job2 apple
(integer) 1
redis6379 rpush job2 banana
(integer) 2
redis6379 rpush job2 cherry
(integer) 3
redis6379 rpush job2 date
(integer) 4
redis6379 rpush job2 fig
(integer) 5
redis6379 rpush job2 grape
(integer) 6
redis6379 rpush job2 lemon
(integer) 7
redis6379 rpush job2 melon
(integer) 8
redis6379 rpush job2 orange
(integer) 9
redis6379 lrange job2 0 -1
1) apple
2) banana
3) cherry
4) date
5) fig
6) grape
7) lemon
8) melon
9) orange
```

So, the list with key `job2` will be the work queue.

Note if you do not have Kube DNS setup correctly, you may need to change
the first step of the above block to `redis-cli -h REDIS_SERVICE_HOST`.

# # Create a container image #create-an-image

Now you are ready to create an image that will process the work in that queue.

Youre going to use a Python worker program with a Redis client to read
the messages from the message queue.

A simple Redis work queue client library is provided,
called `rediswq.py` ([Download](examplesapplicationjobredisrediswq.py)).

The worker program in each Pod of the Job uses the work queue
client library to get work.  Here it is

 code_sample languagepython fileapplicationjobredisworker.py

You could also download [`worker.py`](examplesapplicationjobredisworker.py),
[`rediswq.py`](examplesapplicationjobredisrediswq.py), and
[`Dockerfile`](examplesapplicationjobredisDockerfile) files, then build
the container image. Heres an example using Docker to do the image build

```shell
docker build -t job-wq-2 .
```

# # # Push the image

For the [Docker Hub](httpshub.docker.com), tag your app image with
your username and push to the Hub with the below commands. Replace
`` with your Hub username.

```shell
docker tag job-wq-2 job-wq-2
docker push job-wq-2
```

You need to push to a public repository or [configure your cluster to be able to access
your private repository](docsconceptscontainersimages).

# # Defining a Job

Here is a manifest for the Job you will create

 code_sample fileapplicationjobredisjob.yaml

Be sure to edit the manifest to
change `gcr.iomyproject` to your own path.

In this example, each pod works on several items from the queue and then exits when there are no more items.
Since the workers themselves detect when the workqueue is empty, and the Job controller does not
know about the workqueue, it relies on the workers to signal when they are done working.
The workers signal that the queue is empty by exiting with success.  So, as soon as **any** worker
exits with success, the controller knows the work is done, and that the Pods will exit soon.
So, you need to leave the completion count of the Job unset. The job controller will wait for
the other pods to complete too.

# # Running the Job

So, now run the Job

```shell
# this assumes you downloaded and then edited the manifest already
kubectl apply -f .job.yaml
```

Now wait a bit, then check on the Job

```shell
kubectl describe jobsjob-wq-2
```
```
Name             job-wq-2
Namespace        default
Selector         controller-uidb1c7e4e3-92e1-11e7-b85e-fa163ee3c11f
Labels           controller-uidb1c7e4e3-92e1-11e7-b85e-fa163ee3c11f
                  job-namejob-wq-2
Annotations
Parallelism      2
Completions
Start Time       Mon, 11 Jan 2022 170759 0000
Pods Statuses    1 Running  0 Succeeded  0 Failed
Pod Template
  Labels       controller-uidb1c7e4e3-92e1-11e7-b85e-fa163ee3c11f
                job-namejob-wq-2
  Containers
   c
    Image              container-registry.exampleexampleprojectjob-wq-2
    Port
    Environment
    Mounts
  Volumes
Events
  FirstSeen    LastSeen    Count    From            SubobjectPath    Type        Reason            Message
  ---------    --------    -----    ----            -------------    --------    ------            -------
  33s          33s         1        job-controller                 Normal      SuccessfulCreate  Created pod job-wq-2-lglf8
```

You can wait for the Job to succeed, with a timeout
```shell
# The check for condition name is case insensitive
kubectl wait --forconditioncomplete --timeout300s jobjob-wq-2
```

```shell
kubectl logs podsjob-wq-2-7r7b2
```
```
Worker with sessionID bbd72d0a-9e5c-4dd6-abf6-416cc267991f
Initial queue state emptyFalse
Working on banana
Working on date
Working on lemon
```

As you can see, one of the pods for this Job worked on several work units.

# # Alternatives

If running a queue service or modifying your containers to use a work queue is inconvenient, you may
want to consider one of the other
[job patterns](docsconceptsworkloadscontrollersjob#job-patterns).

If you have a continuous stream of background processing work to run, then
consider running your background workers with a ReplicaSet instead,
and consider running a background processing library such as
[httpsgithub.comresqueresque](httpsgithub.comresqueresque).
