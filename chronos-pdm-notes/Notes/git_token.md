---
id: git_token
aliases: []
tags: []
---

# Git Access Token

Git Access token that I created to clone the `chronos_pdm` repository on `acquario3`:

```plaintext
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
```

Command to clone the forked `chronos-rul` repository using the correct access token:

```bash
git clone https://FrizzoDavide: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@github.com/FrizzoDavide/chronos-rul.git
```

Command to make `chronos-rul` a submodule, also here I have to put the access token I guess:

```bash
git submodule add https://FrizzoDavide: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@github.com/FrizzoDavide/chronos-rul.git
```

Command to insert the changes I made on the cloned `chronos-forecasting` repository into the forked `chronos-rul` repository:

```bash
cd path/to/old/cloned/repo  # Important:  Navigate into the submodule!
patch -p1 < path/to/my-changes.patch
```

# Git Access Token for `SSM_PDM` project

I have to generate an access token also for the `SSM_PDM` repository, since it asked me (probably because I tried to push something after a long time):

```txt
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
```

This token will expire on May 26 2025.

# Git Access Token for Overleaf `git` integration

This is the access token to manage the access on the `git` repository connected to my Overleaf projects:

```txt
olp_6ULq8CfdiIxx5lujzEI5nxOwhugtXk2YPW77
```

It expires on April 24th 2026.
