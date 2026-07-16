# Connecting to the AWS control plane

Once the service is set up on AWS ([Setting up the live service on AWS](aws.md)) and running, this
is how a laptop reaches it — both the Dagster UI and a shell on the always-on control-plane box —
over Tailscale.

Everything routes through the OCF tailnet. The control-plane box has **no publicly-reachable
ports** (its security group allows no inbound traffic except Postgres from the Fargate workers),
and the Dagster webserver has **no login of its own**, so joining the tailnet is what grants — and
gates — access. Anyone on the OCF tailnet can reach the UI and, via Tailscale SSH, a shell as
`ubuntu`; that is intended for this box. See
[the access-phasing plan](../roadmap/live-service.md#access-phasing) for why the security model is
"tailnet membership is the authentication" at this stage.

This page is the client-laptop counterpart to [Step 12](aws.md#step-12-join-the-tailnet) of the
AWS runbook: that step joins the *box* to the tailnet, once; this page gets *your* laptop onto the
same tailnet so it can see the box.

## Prerequisites

- **The AWS stack is set up and running** — [Setting up the live service on AWS](aws.md) has been
  completed. In particular the control-plane box `nged-forecast-ctrl` has joined the OCF tailnet
  with Tailscale SSH enabled ([Step 12](aws.md#step-12-join-the-tailnet)), and the Dagster stack is
  up ([Step 15](aws.md#step-15-start-the-stack-and-connect-over-tailscale)).
- **An OCF Google Workspace account** (`…@openclimatefix.org`). The tailnet is org-scoped, so a
  personal Google account cannot join it — and only devices on the OCF tailnet can see
  `nged-forecast-ctrl`.

## Step 1 — Install Tailscale on your laptop

- **macOS / Windows**: install the GUI client from
  [tailscale.com/download](https://tailscale.com/download) (on macOS, `brew install --cask
  tailscale` works too).
- **Linux**: use the same install script the box uses in
  [Step 12](aws.md#step-12-join-the-tailnet) — it adds Tailscale's own APT repository so the client
  keeps getting the current stable release through `apt upgrade`:

    ```bash
    curl -fsSL https://tailscale.com/install.sh | sh
    ```

## Step 2 — Join the OCF tailnet

Sign in **with your OCF Google Workspace account** (`…@openclimatefix.org`) so your laptop joins
the shared OCF org tailnet rather than a personal one:

- **GUI client**: launch Tailscale, choose **Log in**, then **Sign in with Google**, and pick your
  `…@openclimatefix.org` account.
- **CLI (Linux)**: run `sudo tailscale up`, open the URL it prints, and sign in with the OCF Google
  account.

A device is on **one tailnet at a time**. If your laptop is already signed into a personal
Tailscale account, switch it to the OCF account first — the account menu in the GUI client, or
`sudo tailscale switch` / `sudo tailscale login` on the CLI — otherwise `nged-forecast-ctrl` will
not appear.

## Step 3 — Confirm you can reach the box

```bash
tailscale status                     # nged-forecast-ctrl should appear in the list
tailscale ping nged-forecast-ctrl
```

`nged-forecast-ctrl` is the box's stable MagicDNS name (set by `--hostname` in
[Step 12](aws.md#step-12-join-the-tailnet)). If the name ever fails to resolve — MagicDNS is off on
your client, say — read the box's raw Tailscale IP (a `100.x` address) from the `tailscale status`
output and use that instead.

## Step 4 — Open the Dagster UI

In a browser, open **`http://nged-forecast-ctrl:3000`** (plain `http`, MagicDNS name; the raw
`100.x` Tailscale IP works too). There is no login prompt — that is by design (see the security
note at the top of this page).

From here, driving the running service — promoting a champion, materialising or backfilling a slot,
inspecting a forecast — is [Operating the live service](operations.md).

## Step 5 — SSH into the box

```bash
ssh ubuntu@nged-forecast-ctrl
```

No SSH key and no key management: the box runs **Tailscale SSH** (the `--ssh` flag in
[Step 12](aws.md#step-12-join-the-tailnet)), so access is governed by the tailnet's ACLs rather
than a key file, and the login user is `ubuntu`. Use this shell for the box-side `docker compose`
operations — checking service health, tailing logs, restarting the stack — that live in
[Step 15](aws.md#step-15-start-the-stack-and-connect-over-tailscale), for example:

```bash
cd ~/nged-forecast
docker compose ps                    # all four services should be Up
docker compose logs -f daemon
```

## See also

- [Setting up the live service on AWS](aws.md) — the one-time bring-up this page assumes is done,
  including [Step 12](aws.md#step-12-join-the-tailnet) (joining the box to the tailnet) and
  [Step 15](aws.md#step-15-start-the-stack-and-connect-over-tailscale) (starting the stack).
- [Operating the live service](operations.md) — what to do once you are connected: promotion, the
  6-hourly schedule, inspecting forecasts, backfilling missed slots.
