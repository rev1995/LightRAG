---
title Monitoring, Logging, and Debugging
description Set up monitoring and logging to troubleshoot a cluster, or debug a containerized application.
weight 40
reviewers
- brendandburns
- davidopp
content_type concept
no_list true
card
  name tasks
  weight 999
  title Getting help
---

Sometimes things go wrong. This guide is aimed at making them right. It has
two sections

* [Debugging your application](docstasksdebugdebug-application) - Useful
  for users who are deploying code into Kubernetes and wondering why it is not working.
* [Debugging your cluster](docstasksdebugdebug-cluster) - Useful
  for cluster administrators and people whose Kubernetes cluster is unhappy.

You should also check the known issues for the [release](httpsgithub.comkuberneteskubernetesreleases)
youre using.

# # Getting help

If your problem isnt answered by any of the guides above, there are variety of
ways for you to get help from the Kubernetes community.

# # # Questions

The documentation on this site has been structured to provide answers to a wide
range of questions. [Concepts](docsconcepts) explain the Kubernetes
architecture and how each component works, while [Setup](docssetup) provides
practical instructions for getting started. [Tasks](docstasks) show how to
accomplish commonly used tasks, and [Tutorials](docstutorials) are more
comprehensive walkthroughs of real-world, industry-specific, or end-to-end
development scenarios. The [Reference](docsreference) section provides
detailed documentation on the [Kubernetes API](docsreferencegeneratedkubernetes-api)
and command-line interfaces (CLIs), such as [`kubectl`](docsreferencekubectl).

# # Help! My question isnt covered!  I need help now!

# # # Stack Exchange, Stack Overflow, or Server Fault #stack-exchange

If you have questions related to *software development* for your containerized app,
you can ask those on [Stack Overflow](httpsstackoverflow.comquestionstaggedkubernetes).

If you have Kubernetes questions related to *cluster management* or *configuration*,
you can ask those on
[Server Fault](httpsserverfault.comquestionstaggedkubernetes).

There are also several more specific Stack Exchange network sites which might
be the right place to ask Kubernetes questions in areas such as
[DevOps](httpsdevops.stackexchange.comquestionstaggedkubernetes),
[Software Engineering](httpssoftwareengineering.stackexchange.comquestionstaggedkubernetes),
or [InfoSec](httpssecurity.stackexchange.comquestionstaggedkubernetes).

Someone else from the community may have already asked a similar question or
may be able to help with your problem.

The Kubernetes team will also monitor
[posts tagged Kubernetes](httpsstackoverflow.comquestionstaggedkubernetes).
If there arent any existing questions that help, **please ensure that your question
is [on-topic on Stack Overflow](httpsstackoverflow.comhelpon-topic),
[Server Fault](httpsserverfault.comhelpon-topic), or the Stack Exchange
Network site youre asking on**, and read through the guidance on
[how to ask a new question](httpsstackoverflow.comhelphow-to-ask),
before asking a new one!

# # # Slack

Many people from the Kubernetes community hang out on Kubernetes Slack in the `#kubernetes-users` channel.
Slack requires registration you can [request an invitation](httpsslack.kubernetes.io),
and registration is open to everyone). Feel free to come and ask any and all questions.
Once registered, access the [Kubernetes organisation in Slack](httpskubernetes.slack.com)
via your web browser or via Slacks own dedicated app.

Once you are registered, browse the growing list of channels for various subjects of
interest. For example, people new to Kubernetes may also want to join the
[`#kubernetes-novice`](httpskubernetes.slack.commessageskubernetes-novice) channel. As another example, developers should join the
[`#kubernetes-contributors`](httpskubernetes.slack.commessageskubernetes-contributors) channel.

There are also many country specific  local language channels. Feel free to join
these channels for localized support and info

Country  Channels
---------------------
China  [`#cn-users`](httpskubernetes.slack.commessagescn-users), [`#cn-events`](httpskubernetes.slack.commessagescn-events)
Finland  [`#fi-users`](httpskubernetes.slack.commessagesfi-users)
France  [`#fr-users`](httpskubernetes.slack.commessagesfr-users), [`#fr-events`](httpskubernetes.slack.commessagesfr-events)
Germany  [`#de-users`](httpskubernetes.slack.commessagesde-users), [`#de-events`](httpskubernetes.slack.commessagesde-events)
India  [`#in-users`](httpskubernetes.slack.commessagesin-users), [`#in-events`](httpskubernetes.slack.commessagesin-events)
Italy  [`#it-users`](httpskubernetes.slack.commessagesit-users), [`#it-events`](httpskubernetes.slack.commessagesit-events)
Japan  [`#jp-users`](httpskubernetes.slack.commessagesjp-users), [`#jp-events`](httpskubernetes.slack.commessagesjp-events)
Korea  [`#kr-users`](httpskubernetes.slack.commessageskr-users)
Netherlands  [`#nl-users`](httpskubernetes.slack.commessagesnl-users)
Norway  [`#norw-users`](httpskubernetes.slack.commessagesnorw-users)
Poland  [`#pl-users`](httpskubernetes.slack.commessagespl-users)
Russia  [`#ru-users`](httpskubernetes.slack.commessagesru-users)
Spain  [`#es-users`](httpskubernetes.slack.commessageses-users)
Sweden  [`#se-users`](httpskubernetes.slack.commessagesse-users)
Turkey  [`#tr-users`](httpskubernetes.slack.commessagestr-users), [`#tr-events`](httpskubernetes.slack.commessagestr-events)

# # # Forum

Youre welcome to join the official Kubernetes Forum [discuss.kubernetes.io](httpsdiscuss.kubernetes.io).

# # # Bugs and feature requests

If you have what looks like a bug, or you would like to make a feature request,
please use the [GitHub issue tracking system](httpsgithub.comkuberneteskubernetesissues).

Before you file an issue, please search existing issues to see if your issue is
already covered.

If filing a bug, please include detailed information about how to reproduce the
problem, such as

* Kubernetes version `kubectl version`
* Cloud provider, OS distro, network configuration, and container runtime version
* Steps to reproduce the problem
