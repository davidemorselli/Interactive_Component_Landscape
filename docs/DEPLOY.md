# Deploying the app on the internet

This walkthrough explains how to
- obtain (rent) a server
- deploy the application
- set up the Cloudflare reverse proxy to add authentication and protect the server.

To keep this document short, I don't go into detail on every step. For more guidance on any of them, I recommend asking an LLM.

## What to rent

The app needs about 2.6 GB of RAM. 


Unfortunately, because of the current RAM and SSD shortages, rental prices are very high right now.

| Provider | Plan | Specs | Price incl. 8.1% CH VAT |
| --- | --- | --- | --- |
| **[Hetzner](https://www.hetzner.com/cloud/cost-optimized/)**, Cost-Optimized | CX23 | 2 vCPU, 4 GB, 40 GB | €7 not orderable |
| [Hetzner](https://www.hetzner.com/cloud/regular-performance/), Regular Performance | CPX22 | 2 vCPU, 4 GB, 80 GB | €21.61 |
| [netcup](https://www.netcup.com/en/server/vps-lite) | VPS Lite 1 G12s | 2 vCPU, 4 GB, 80 GB | €4.43 |



Hetzner and Netcup offer the best value for money among European hosting providers.

My first choice would be Hetzner's CX33, since ordering one is easy. Unfortunately it is very often out of stock.. The Regular Performance option is always available but very expensive. 

The alternative is Netcup. I have rented servers from them before and they are very good, but the process is a bit more involved: they verify the identity of new customers and approve orders by hand, which can take 1-2 days, and its plans have a 6 month minimum term.

Once you have picked a plan, you need to choose the server's OS. I recommend Ubuntu Server, the standard option. You then register an SSH key on the server so you can connect to it from your own computer.

## Setup

Everything below runs as root on the new server, once.

**1. Install Docker**

```sh
curl -fsSL https://get.docker.com | sh
```

**2. Get the code and the data**

I recommend using a git repository (e.g. GitHub) to manage the code and push changes to the server. To do so, download the code saved on the drive and put it in a GitHub repository.

It is also best to keep the repository private and add a deploy SSH key on the server so it can pull the code. Once that is done, clone the code onto the server:

```sh
git clone -b website https://github.com/<user>/Interactive_Component_Landscape.git ica
cd ica
```
The data files are usually not carried in the github repository because they are too large and Github has a limit of 100Mb. You need to download them to your computer from the drive and then send them to the server using the SSH protocol. You can use the `scp` or `rclone` commands to do so.

**3. Create the Cloudflare tunnel**

Cloudflare acts as a reverse proxy for the application. In short, every request to the application goes through Cloudflare first. This lets you add an authentication layer (you choose who can access the app) and means the server has no ports open to the internet, which protects it from attacks.

To configure this, you need:

- A Cloudflare account (free)
- A domain name (paid)

You can buy the domain directly on [domains.cloudflare.com](https://domains.cloudflare.com): search for an available name, buy it, and it appears in your Cloudflare account with nothing else to configure. Cloudflare resells at cost price, with no margin and no increase at renewal (a `.com` costs about $10/year, the same price every year).

In a browser, on a Cloudflare account with a domain you control:

1. [one.dash.cloudflare.com](https://one.dash.cloudflare.com) → Networks → Tunnels → **Create a tunnel** → Cloudflared.
2. Name it, then **copy the long token** out of the install command it shows you (just the token, not the command). Ignore the rest of that page.
3. Public Hostname → **Add** → choose your subdomain (e.g. `landscape.example.org`), Service type **HTTP**, URL `web:8050` → Save.

**DNS and certificates: there is nothing to do.** Step 3 creates the DNS entry for the subdomain by itself, and Cloudflare issues and renews the HTTPS certificate automatically.

**4. Fill in the secrets**

On the server, store the secrets in a file named `.env` (see `.env.example` for a sample).

Create the file:

```sh
touch .env
nano .env
```

and fill it with the following content:
```
OPENROUTER_API_KEY=sk-or-...
TUNNEL_TOKEN=<the token from step 3>
COMPOSE_PROFILES=tunnel
```

**5. Start it**

Then launch the application with:

```sh
docker compose --profile tunnel up -d --build
```

The **first start takes 15–20 minutes** and looks frozen — it is computing the ICA and UMAP and caching them. Watch it with `docker compose logs -f web` and wait for the line `Ready.`. Every later start takes ~25 seconds.

The site is now live on your subdomain — and open to the whole internet. Do step 6 the same day.

**6. Adding access control**

The following steps add the protection layer.

In the same Cloudflare dashboard: Access → Applications → **Add an application** → Self-hosted.

1. Application domain: the subdomain from step 3.
2. Add a policy: Action **Allow**, and one rule — either *Emails* with the addresses you want to let in, or *Emails ending in* `@unil.ch` for the whole university.
3. Save.

Visitors now get a Cloudflare page asking for their email, receive a six-digit code, and only then reach the app. There is nothing to install on their side and no password to share. Free for up to 50 people.

## Afterwards

Update to the latest code:

```sh
cd ica && git pull && docker compose --profile tunnel up -d --build
```


## The server can host other applications

The app runs in a Docker container. You can run other containers with other applications alongside it, so that the server's resources are shared.