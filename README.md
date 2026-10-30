# ShipScan - AI Food Scanner (START HERE)

You only do TWO things yourself: add your photos, then run a few commands.
Then scan items with your phone using Expo Go.

## What's inside
```
SmartPantry_Complete/
├── dataset/              <- PUT YOUR PHOTOS HERE (one folder per food)
│   ├── apple/  banana/  bread/  egg/
│   └── milk_carton/  tomato/  packaged_snack/  canned_food/
├── split_dataset.py      step 1: splits photos into train/val/test
├── train_model.py        step 2: trains YOUR model
├── app.py + templates/   step 3: server (also a web page) using your model
├── expo_app/             step 4: phone app for Expo Go
├── Train_ShipScan_Colab.ipynb  (optional) train for free on Google Colab
├── extract_frames.py     (optional) turns a video into photos
└── requirements.txt
```

The app combines your trained classes with a general image/text recognizer for
food and common household items. It tries to reject images it identifies as
unrelated, including animals, but image recognition is not perfect. The web app
supports photo uploads and webcam capture; its scan log is stored in that
browser and can be cleared from the page. For best grocery results, add photos
to `dataset/red_onion/` and `dataset/carrot/`, then rerun the split and training
steps. The general model's first run downloads its pretrained weights.

## Step 0 - Install (once)
Python 3.10 or 3.11, then in a terminal in this folder:
```
pip install -r requirements.txt
```

## Step 1 - Add your photos
Drag photos into the matching folders in `dataset/`.
- 100+ per folder is ideal (30 is enough for a test); vary angle, light, background
- You never paste images into code - the scripts read the folders
- Different foods? Rename/add/delete folders. Folder names become labels.
  Keep at least 2 folders with photos and delete any empty ones.
- Shortcut: film a 30 sec video, then `python extract_frames.py video.mp4 apple`

## Step 2 - Train
```
python split_dataset.py
python train_model.py
```
Accuracy numbers for your report: `model/results_summary.txt`, `training_history.png`.

## Step 3 - Start the server
```
python app.py
```
It prints the address to use on your phone. (You can also open
http://127.0.0.1:5000 on the computer. Green banner = your own model.)

## Step 4 - Use your phone with Expo Go
Follow `expo_app/README.md`.

## Publish a public page and use the prediction server

GitHub Pages hosts the public web interface in `docs/`, but it cannot run the
Python model. For a no-cloud-compute-cost demo, the prediction server can run on
your computer and a temporary Cloudflare Quick Tunnel can connect the page to
it. Your computer and server must stay on; anyone with the tunnel URL can send
images to the server.

1. Create a **public** GitHub repository and upload the project files. The
   `.gitignore` excludes training photos, split datasets, and local environments.
2. In GitHub, open **Settings → Pages**, choose **Deploy from a branch**, select
   `main` and the `/docs` folder, and save. The public page URL will be
   `https://YOUR-USERNAME.github.io/REPOSITORY/`.
3. Start the prediction server in this project folder. If port 5000 is already
   occupied, in PowerShell run:
   ```
   $env:PORT = "5001"
   python app.py
   ```
4. Install `cloudflared` using Cloudflare's official instructions, then start a
   second terminal and run:
   ```
   cloudflared tunnel --url http://127.0.0.1:5001
   ```
   Copy the temporary `https://...trycloudflare.com` URL it prints.
5. Open the GitHub Pages link. It connects automatically, supports webcam or
   phone-camera capture, and shows a per-browser scan log with a **Clear log**
   button. If you restart the temporary tunnel, update its address in
   `docs/index.html` and publish the change.

Quick Tunnels are for testing, have no uptime guarantee, and stop working when
`cloudflared` exits. They do not make the prediction service permanently hosted.
Do not submit private photos through a public tunnel.

For permanent hosting, use a paid prediction server. The prepared `Dockerfile`
can be used with a container host, but check its current pricing and memory
requirements first.

## Troubleshooting
- "No images found yet": put photos in the `dataset/` folders first.
- Training errors about empty classes: delete empty folders in `dataset/`.
- Training slow: use Google Colab or fewer images first.
- After retraining, restart `app.py`.
