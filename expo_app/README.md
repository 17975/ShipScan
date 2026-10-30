# ShipScan - Expo Go phone app

Uses your phone camera and sends each photo over Wi-Fi to the ShipScan
server (app.py) running on your computer. It classifies food and reports when
an image appears unrelated, including household objects and animals; recognition
can make mistakes.

## Requirements
- Node.js (https://nodejs.org, LTS version) on your computer
- The "Expo Go" app on your phone (App Store / Google Play)
- Phone and computer on the SAME Wi-Fi network

## One-time setup (run in a terminal, in the folder that CONTAINS SmartPantry_Complete)
```
npx create-expo-app@latest shipscan-expo --template blank
cd shipscan-expo
npx expo install expo-camera @react-native-async-storage/async-storage
```
Then replace the generated `App.js` with the `App.js` from this folder
(SmartPantry_Complete/expo_app/App.js).

## Every time you want to use it
1. Terminal 1 - start the server (in SmartPantry_Complete, after training):
   ```
   python app.py
   ```
   It prints a line like `PHONE (Expo Go) server address: http://192.168.1.23:5000`

2. Terminal 2 - start Expo (in shipscan-expo):
   ```
   npx expo start
   ```
   Scan the QR code with your phone (Camera app on iPhone, Expo Go app on Android).

3. In the app, type the server address from step 1, tap "Test connection".
   You should see that food recognition is ready.

4. Point the camera at food:
   - "Scan food & add to log" takes one photo and logs food predictions
   - Non-food photos are reported as "This is not a food item."
   - "Auto-scan" keeps classifying the camera image every ~2.5 seconds
   - "Clear" removes the scan log stored in the app

## Troubleshooting
- "Cannot reach server": same Wi-Fi? Address typed exactly as printed? If
  Windows asks to allow Python through the firewall, click Allow (private networks).
- University/guest Wi-Fi often blocks device-to-device traffic. Use your phone's
  hotspot for the laptop, or a home network.
- General food recognition downloads its model weights the first
  time the server starts, so the computer needs internet access on that first run.
- After retraining, restart app.py.
