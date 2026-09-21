# DVC.org account (Studio)

The DVC command I installed with pip does not need an account. It versions files on this machine.

The website [dvc.org](https://dvc.org/) also has Studio. That part needs a login if I want the model and data to show up online, not only in the `storage/` folder.

## Create the account

1. Open [https://dvc.org/](https://dvc.org/) and choose Get started with DVC (the Git extension, not lakeFS).
2. Sign up with GitHub, GitLab, or Bitbucket. Studio does not ask for a separate password. Docs: [studio login](https://doc.dvc.org/command-reference/studio/login).
3. If the site sends me to Studio, that is the same account. Current Studio host is [https://studio.datachain.ai](https://studio.datachain.ai). Older links may say studio.iterative.ai.

## Connect this folder after the account exists

From `Task[03]`:

```bash
dvc studio login --no-open
```

The terminal prints a code. I paste that code in the browser page it names. The token stays in DVC config on this machine. I do not put the token in `config.yaml` or in git.

I already versioned the model with a local remote named `localstore` (`storage/`). Studio login does not replace that. It only lets me share later.

## What is already versioned

```bash
dvc status
```

Tracked files have a `.dvc` pointer next to them. The big bytes live in `storage/files`. If I delete `models/final_model.joblib` I can get it back with:

```bash
dvc pull
```
