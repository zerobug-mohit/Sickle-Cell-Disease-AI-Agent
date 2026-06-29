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
- **Machine type:** `e2-small` (2 vCPU, 2 GB) is enough; `e2-medium` (4 GB) for headroom
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

## Phase 3 — Upload the app bundle

From your **laptop** (the bundle `scd-bot-deploy.tar.gz` was generated in the project root;
it contains `app/`, `requirements.txt`, `data/faiss_index/`, `deploy/`, and `.env`):

Using GCP in-browser SSH: click **Upload file** (gear menu) and upload `scd-bot-deploy.tar.gz`.

Or with gcloud CLI:
```bash
gcloud compute scp scd-bot-deploy.tar.gz scd-bot:~ --zone=YOUR_ZONE
```

---

## Phase 4 — Set up the app on the VM

SSH into the VM, then:

```bash
# System deps (no Tesseract needed — we are not ingesting on the VM)
sudo apt update
sudo apt install -y python3-venv python3-pip

# Dedicated service user + app dir
sudo useradd --system --create-home --home-dir /opt/scd-bot scdbot || true
sudo mkdir -p /opt/scd-bot

# Unpack the bundle into /opt/scd-bot
sudo tar -xzf ~/scd-bot-deploy.tar.gz -C /opt/scd-bot
sudo chown -R scdbot:scdbot /opt/scd-bot

# Python venv + dependencies
sudo -u scdbot bash -c '
  cd /opt/scd-bot
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
'

# Sanity check: index loads
sudo -u scdbot /opt/scd-bot/.venv/bin/python -c "
from app.rag.vector_store import VectorStore
vs = VectorStore(); vs.load('/opt/scd-bot/data/faiss_index')
print('index vectors:', vs._index.ntotal)
"
```

The `.env` is already inside the bundle (set `SESSION_BACKEND=memory`; the permanent
Meta token and OpenAI key are already filled in).

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
curl -s "https://SUBDOMAIN/webhook?hub.mode=subscribe&hub.verify_token=scd_bot_wh_7Kq2mZ&hub.challenge=OK123"
# -> should print: OK123
```

---

## Phase 7 — Point Meta at the new URL

The webhook callback currently points at the ngrok URL. Update it to the VM URL.
(Ask the assistant to run this, or do it via the Meta dashboard:
WhatsApp → Configuration → Webhook → Edit → Callback `https://SUBDOMAIN/webhook`,
verify token `scd_bot_wh_7Kq2mZ`.)

API method (app access token = `APP_ID|APP_SECRET`):
```bash
curl -X POST "https://graph.facebook.com/v22.0/989312020608269/subscriptions" \
  --data-urlencode "object=whatsapp_business_account" \
  --data-urlencode "callback_url=https://SUBDOMAIN/webhook" \
  --data-urlencode "verify_token=scd_bot_wh_7Kq2mZ" \
  --data-urlencode "fields=messages" \
  --data-urlencode "access_token=989312020608269|YOUR_APP_SECRET"
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
- After this, you can stop the local laptop server + ngrok — the VM is the live host.
- The DuckDNS IP must stay pointed at the VM's static IP (it won't change since it's reserved).
- To update code later: rebuild the bundle, re-upload, `sudo tar -xzf ... -C /opt/scd-bot`, `sudo systemctl restart scd-bot`.
- The VM only runs the server; re-ingestion (new PDFs) is still done on your laptop, then copy `data/faiss_index/` up and restart.
