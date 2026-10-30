import React, { useRef, useState, useEffect } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  ScrollView, ActivityIndicator, Switch, SafeAreaView,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import AsyncStorage from '@react-native-async-storage/async-storage';

const GREEN = '#2F6F4F';
const DEFAULT_URL = 'http://192.168.1.10:5000'; // replace in the app with the address your server prints
const AUTO_INTERVAL_MS = 2500;
const MIN_CONFIDENCE = 50; // below this we say "not sure"

export default function App() {
  const cameraRef = useRef(null);
  const busyRef = useRef(false);
  const [permission, requestPermission] = useCameraPermissions();
  const [serverUrl, setServerUrl] = useState(DEFAULT_URL);
  const [status, setStatus] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [auto, setAuto] = useState(false);
  const [log, setLog] = useState([]);

  useEffect(() => {
    AsyncStorage.getItem('serverUrl').then((v) => { if (v) setServerUrl(v); }).catch(() => {});
  }, []);

  const baseUrl = () => serverUrl.trim().replace(/\/$/, '');

  const saveUrl = (v) => {
    setServerUrl(v);
    AsyncStorage.setItem('serverUrl', v).catch(() => {});
  };

  const withTimeout = async (url, options, ms = 15000) => {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), ms);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(t);
    }
  };

  const testConnection = async () => {
    setStatus('Testing...');
    try {
      const res = await withTimeout(`${baseUrl()}/health`, {}, 120000);
      const data = await res.json();
      setStatus(
        data.using_custom_model && data.using_general_food_model
          ? '✅ Connected - using your trained model plus general food recognition'
          : data.using_general_food_model
            ? '✅ Connected - using general food recognition'
            : '⚠️ Connected, but server has no food recognition model'
      );
    } catch (e) {
      setStatus('❌ Cannot reach server. Check the address, Wi-Fi, and that app.py is running.');
    }
  };

  const scan = async (addToLog) => {
    if (busyRef.current || !cameraRef.current) return;
    busyRef.current = true;
    setLoading(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.5, skipProcessing: true });
      const form = new FormData();
      form.append('image', { uri: photo.uri, name: 'scan.jpg', type: 'image/jpeg' });
      form.append('log', addToLog ? '1' : '0');

      const res = await withTimeout(`${baseUrl()}/predict`, { method: 'POST', body: form }, 120000);
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      setResult(data);
      setError(null);
      if (addToLog && data.confidence >= MIN_CONFIDENCE) {
        setLog((prev) => [
          { item: data.label, confidence: data.confidence, time: new Date().toLocaleTimeString() },
          ...prev,
        ]);
      }
    } catch (e) {
      setError('Scan failed: ' + (e.message || 'unknown error') + '\nCheck the server address and Wi-Fi.');
    } finally {
      busyRef.current = false;
      setLoading(false);
    }
  };

  // Auto-scan: take a picture every few seconds while switched on
  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => scan(false), AUTO_INTERVAL_MS);
    return () => clearInterval(id);
  }, [auto, serverUrl]);

  if (!permission) return <View style={styles.center}><ActivityIndicator /></View>;
  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.text}>ShipScan needs camera access to scan items.</Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Allow camera</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const confident = result && result.confidence >= MIN_CONFIDENCE;
  const scoreLabel = result && result.recognition_source === 'general' ? 'match score' : 'confidence';

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>🥫 ShipScan</Text>

        <View style={styles.card}>
          <Text style={styles.label}>Server address (printed when you run app.py)</Text>
          <TextInput
            style={styles.input}
            value={serverUrl}
            onChangeText={saveUrl}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
            placeholder="http://192.168.x.x:5000"
          />
          <TouchableOpacity style={styles.btnLight} onPress={testConnection}>
            <Text style={styles.btnLightText}>Test connection</Text>
          </TouchableOpacity>
          {!!status && <Text style={styles.status}>{status}</Text>}
        </View>

        <View style={styles.cameraWrap}>
          <CameraView ref={cameraRef} style={styles.camera} facing="back" />
          {loading && (
            <View style={styles.badge}><ActivityIndicator color="#fff" /></View>
          )}
        </View>

        <View style={styles.row}>
          <Text style={styles.text}>Auto-scan (every ~2.5s)</Text>
          <Switch value={auto} onValueChange={setAuto} trackColor={{ true: GREEN }} />
        </View>

        <TouchableOpacity style={styles.btn} onPress={() => scan(true)}>
          <Text style={styles.btnText}>Scan & add to pantry</Text>
        </TouchableOpacity>

        {!!error && <Text style={styles.error}>{error}</Text>}

        {result && (
          <View style={styles.card}>
            {confident ? (
              <>
                <Text style={styles.resultMain}>{result.label}</Text>
                <Text style={styles.sub}>{result.confidence.toFixed(1)}% {scoreLabel}</Text>
              </>
            ) : (
              <>
                <Text style={styles.resultMain}>Not sure</Text>
                <Text style={styles.sub}>
                  Best guess: {result.label} ({result.confidence.toFixed(1)}% {scoreLabel}). Try a closer, better-lit shot.
                </Text>
              </>
            )}
            <Text style={styles.sub}>
              Recognized by {result.recognition_source === 'general' ? 'general food model' : 'your trained model'}
            </Text>
            {Object.entries(result.all_confidences || {}).map(([name, pct]) => (
              <View key={name} style={{ marginTop: 6 }}>
                <Text style={styles.small}>{name}: {Number(pct).toFixed(1)}%</Text>
                <View style={styles.barBg}>
                  <View style={[styles.barFill, { width: `${Math.min(100, pct)}%` }]} />
                </View>
              </View>
            ))}
            {!!(result.additional_food_matches || []).length && (
              <View style={{ marginTop: 12 }}>
                <Text style={styles.cardTitle}>Other possible foods</Text>
                {result.additional_food_matches.map((match) => (
                  <Text key={match.label} style={styles.small}>
                      {match.label}: {Number(match.confidence).toFixed(1)}% match score
                  </Text>
                ))}
              </View>
            )}
          </View>
        )}

        <View style={styles.card}>
          <View style={styles.row}>
            <Text style={styles.cardTitle}>Pantry log</Text>
            <TouchableOpacity onPress={() => setLog([])}><Text style={{ color: GREEN }}>Clear</Text></TouchableOpacity>
          </View>
          {log.length === 0 && <Text style={styles.small}>No items added yet.</Text>}
          {log.map((e, i) => (
            <Text key={i} style={styles.logRow}>
              ✔ {e.item} - {e.confidence.toFixed(1)}% - {e.time}
            </Text>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#f7f7f5' },
  container: { padding: 16, paddingTop: 40 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  title: { fontSize: 26, fontWeight: 'bold', color: GREEN, marginBottom: 12 },
  card: { backgroundColor: '#fff', borderRadius: 10, padding: 14, marginBottom: 14, elevation: 1 },
  cardTitle: { fontSize: 16, fontWeight: 'bold' },
  label: { fontSize: 12, color: '#555', marginBottom: 6 },
  input: { borderWidth: 1, borderColor: '#ccc', borderRadius: 6, padding: 8, fontSize: 14 },
  status: { marginTop: 8, fontSize: 13 },
  cameraWrap: { borderRadius: 12, overflow: 'hidden', marginBottom: 12 },
  camera: { height: 320 },
  badge: { position: 'absolute', top: 10, right: 10, backgroundColor: 'rgba(0,0,0,0.5)', borderRadius: 20, padding: 8 },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  text: { fontSize: 15 },
  btn: { backgroundColor: GREEN, padding: 14, borderRadius: 8, alignItems: 'center', marginBottom: 12 },
  btnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  btnLight: { backgroundColor: '#e8f1ec', padding: 10, borderRadius: 8, alignItems: 'center', marginTop: 8 },
  btnLightText: { color: GREEN, fontWeight: '600' },
  error: { color: '#b00020', marginBottom: 12 },
  resultMain: { fontSize: 24, fontWeight: 'bold', textTransform: 'capitalize' },
  sub: { color: '#555', marginBottom: 6 },
  small: { fontSize: 12 },
  barBg: { height: 6, backgroundColor: '#eee', borderRadius: 3, overflow: 'hidden' },
  barFill: { height: 6, backgroundColor: GREEN },
  logRow: { fontSize: 14, paddingVertical: 3 },
});
