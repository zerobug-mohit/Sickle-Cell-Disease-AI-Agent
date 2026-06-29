# SCD WhatsApp Bot — GCP VM Deployment

Deploy the bot on a Google Cloud VM with a free DuckDNS hostname and Caddy
(automatic Let's Encrypt HTTPS). The FAISS index is copied prebuilt, so the VM
does **not** need the PDFs, Tesseract, or any ingestion step.

Placeholders used below:
- `VM_IP`            = the VM's static external IP
- `SUBDOMAIN`        = your DuckDNS hostname, e.g. `scd-bot.duckdns.org`

---

## Phase 1 — Create the GCP VM

In the Google Cloud Console → **Compute Engine → VM instances → Create instance**:

- **Name:** `scd-bot`
- **Region/Zone:** closest to your users (e.g. `asia-south1` for India)
- **Machine type:** `e2-micro` (2 vCPU shared, 1 GB) for lowest cost — we add a 2 GB swap
  file (Phase 4) to cover the tight RAM. Step up to `e2-small` (2 GB) if you see memory pressure.
- **Boot disk:** Ubuntu **24.04 LTS**, 20 GB standard
- **Firewall:** check **Allow HTTP traffic** and **Allow HTTPS traffic**
- Create.

Then reserve a **static IP** so it never changes:
- **VPC network → IP addresses → Reserve external static address**, attach it to the `scd-bot` VM.
- Note this `VM_IP`.

(Firewall ports 80 + 443 are opened by the HTTP/HTTPS checkboxes; Caddy needs both.)

---

## Phase 2 — Free HTTPS hostname (DuckDNS)

1. Go to **duckdns.org**, sign in (Google/GitHub).
2. Create a subdomain, e.g. `scd-bot` → gives `scd-bot.duckdns.org` (= `SUBDOMAIN`).
3. Set its **current IP** to your `VM_IP` and **update**.
4. Verify from your laptop: `nslookup SUBDOMAIN` should resolve to `VM_IP`.

---

## Phase 3 — Get the code onto the VM (git clone)

SSH into the VM, then clone the public repo into `/opt/scd-bot`. The repo already
includes the prebuilt FAISS index, so there is nothing else to upload.

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip

# 1 GB VM (e2-micro): add 2 GB swap as an OOM safety net
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
free -h   # confirm swap is active

# Dedicated service user + app dir
sudo useradd --system --create-home --home-dir /opt/scd-bot scdbot || true
sudo git clone https://github.com/zerobug-mohit/Sickle-Cell-Disease-AI-Agent.git /opt/scd-bot
sudo chown -R scdbot:scdbot /opt/scd-bot
```

> Later updates are just: `cd /opt/scd-bot && sudo -u scdbot git pull && sudo systemctl restart scd-bot`

---

## Phase 4 — Create `.env` and install dependencies

The `.env` is **not** in the repo (it holds secrets). Create it on the VM with your
real values — copy them from your laptop's `.env`:

```bash
sudo -u scdbot tee /opt/scd-bot/.env >/dev/null <<'EOF'
META_ACCESS_TOKEN=YOUR_PERMANENT_META_TOKEN
META_PHONE_NUMBER_ID=YOUR_PHONE_NUMBER_ID
META_VERIFY_TOKEN=YOUR_VERIFY_TOKEN
META_APP_SECRET=YOUR_APP_SECRET
WHATSAPP_API_VERSION=v22.0
OPENAI_API_KEY=YOUR_OPENAI_KEY
PORT=8000
SESSION_BACKEND=memory
REDIS_URL=redis://localhost:6379
FAISS_INDEX_PATH=./data/faiss_index
TRAINING_MATERIALS_DIR=./data/training_materials
GOOGLE_CREDENTIALS_PATH=
INTERACTION_SHEET_ID=
EOF
sudo chmod 600 /opt/scd-bot/.env
```

(Alternatively `scp` your laptop `.env` to the VM and move it into `/opt/scd-bot/`.)

Then the venv + dependencies (no Tesseract needed — we are not ingesting on the VM):

```bash
sudo -u scdbot bash -c '
  cd /opt/scd-bot
  python3 -m venv .venv
  .venv/bin/pip install --no-cache-dir --upgrade pip
  .venv/bin/pip install --no-cache-dir -r requirements.txt
'

# Sanity check: index loads
sudo -u scdbot bash -c 'cd /opt/scd-bot && .venv/bin/python -c "
from app.rag.vector_store import VectorStore
vs = VectorStore(); vs.load(\"data/faiss_index\")
print(\"index vectors:\", vs._index.ntotal)"'
```

---

## Phase 5 — Run as a service (systemd)

```bash
sudo cp /opt/scd-bot/deploy/scd-bot.service /etc/systemd/system/scd-bot.service
sudo systemctl daemon-reload
sudo systemctl enable --now scd-bot
sudo systemctl status scd-bot --no-pager        # should be active (running)
curl -s localhost:8000/webhook && echo          # local reachability (403 = up)
```

Logs: `journalctl -u scd-bot -f`

---

## Phase 6 — HTTPS reverse proxy (Caddy)

```bash
# Install Caddy
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy

# Configure: point Caddy at our app, with auto-HTTPS for the DuckDNS host
sudo sed -i 's/YOUR_SUBDOMAIN.duckdns.org/SUBDOMAIN/' /opt/scd-bot/deploy/Caddyfile
sudo cp /opt/scd-bot/deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl restart caddy
sudo systemctl status caddy --no-pager
```

Caddy will fetch a Let's Encrypt cert within a few seconds. Verify from your laptop:
```bash
curl -s "https://SUBDOMAIN/webhook?hub.mode=subscribe&hub.verify_token=YOUR_VERIFY_TOKEN&hub.challenge=OK123"
# -> should print: OK123
```

---

## Phase 7 — Point Meta at the new URL

The webhook callback currently points at the ngrok URL. Update it to the VM URL.
(Ask the assistant to run this, or do it via the Meta dashboard:
WhatsApp → Configuration → Webhook → Edit → Callback `https://SUBDOMAIN/webhook`,
verify token `YOUR_VERIFY_TOKEN`.)

API method (app access token = `APP_ID|APP_SECRET`):
```bash
curl -X POST "https://graph.facebook.com/v22.0/YOUR_APP_ID/subscriptions" \
  --data-urlencode "object=whatsapp_business_account" \
  --data-urlencode "callback_url=https://SUBDOMAIN/webhook" \
  --data-urlencode "verify_token=YOUR_VERIFY_TOKEN" \
  --data-urlencode "fields=messages" \
  --data-urlencode "access_token=YOUR_APP_ID|YOUR_APP_SECRET"
```

The WABA→app subscription persists, so no need to redo it.

---

## Phase 8 — Test

Send a WhatsApp message to the test number. Watch the VM:
```bash
journalctl -u scd-bot -f
```
You should see `msg from=...` then a `graph.facebook.com/.../messages 200 OK`.

---

## Notes
- **Memory (e2-micro / 1 GB):** the 2 GB swap (Phase 4) covers spikes. Check usage with
  `free -h` and `systemctl status scd-bot`. If the service is OOM-killed (look for `Killed`
  in `journalctl -u scd-bot`), resize the VM to `e2-small`:
  `gcloud compute instances set-machine-type scd-bot --machine-type=e2-small --zone=ZONE` (VM stopped).
- After this, you can stop the local laptop server + ngrok — the VM is the live host.
- The DuckDNS IP must stay pointed at the VM's static IP (it won't change since it's reserved).
- To update code later: rebuild the bundle, re-upload, `sudo tar -xzf ... -C /opt/scd-bot`, `sudo systemctl restart scd-bot`.
- The VM only runs the server; re-ingestion (new PDFs) is still done on your laptop, then copy `data/faiss_index/` up and restart.
