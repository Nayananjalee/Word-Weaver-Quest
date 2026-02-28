/**
 * GestureRecognizerService - Singleton Preloader
 * 
 * Downloads and initializes MediaPipe GestureRecognizer at app startup
 * so it's instantly ready when the child starts a story.
 * 
 * WHAT THIS SOLVES:
 * - Previously: WASM (~5MB) + Model (~10MB) downloaded AFTER story loads = long wait
 * - Now: Downloads begin immediately when app loads (while child picks topic)
 * - By story time, gesture recognizer is already cached and ready
 * 
 * USAGE:
 *   import gestureService from './GestureRecognizerService';
 *   gestureService.preload();                    // Call on app mount
 *   const recognizer = await gestureService.get(); // Instant if preloaded
 */

import { GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision';

// Pin to specific version for better CDN caching (avoid @latest re-fetching)
const MEDIAPIPE_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';

class GestureRecognizerService {
  constructor() {
    this._recognizer = null;
    this._promise = null;
    this._error = null;
    this._loadStartTime = null;
    this._listeners = new Set();
  }

  /**
   * Start preloading immediately. Safe to call multiple times.
   * Returns the initialization promise.
   */
  preload() {
    if (this._promise) return this._promise;

    this._loadStartTime = Date.now();
    console.log('GestureService: 🚀 Preloading MediaPipe (WASM + model)...');

    this._promise = this._initialize();
    return this._promise;
  }

  /**
   * Get the recognizer instance. Awaits preload if not finished.
   * Returns null if initialization failed.
   */
  async get() {
    if (this._recognizer) return this._recognizer;
    if (!this._promise) this.preload();

    try {
      await this._promise;
      return this._recognizer;
    } catch {
      return null;
    }
  }

  /**
   * Get recognizer synchronously (returns null if not ready yet).
   */
  getSync() {
    return this._recognizer;
  }

  /**
   * Check if recognizer is ready.
   */
  isReady() {
    return this._recognizer !== null;
  }

  /**
   * Get any error from initialization.
   */
  getError() {
    return this._error;
  }

  /**
   * Subscribe to ready state changes.
   */
  onReady(callback) {
    if (this._recognizer) {
      callback(this._recognizer);
      return () => {};
    }
    this._listeners.add(callback);
    return () => this._listeners.delete(callback);
  }

  /**
   * Internal initialization - downloads WASM + model
   */
  async _initialize() {
    try {
      // Step 1: Download WASM runtime (~5MB) 
      console.log('GestureService: ⬇️ Downloading WASM runtime...');
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

      const wasmTime = Date.now() - this._loadStartTime;
      console.log(`GestureService: ✅ WASM ready (${(wasmTime / 1000).toFixed(1)}s)`);

      // Step 2: Download model + create recognizer (~10MB)
      console.log('GestureService: ⬇️ Downloading gesture model...');
      const recognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      const totalTime = Date.now() - this._loadStartTime;
      console.log(`GestureService: ✅ Gesture recognizer ready! (${(totalTime / 1000).toFixed(1)}s total)`);

      this._recognizer = recognizer;
      this._error = null;

      // Notify all listeners
      this._listeners.forEach(cb => cb(recognizer));
      this._listeners.clear();

      return recognizer;
    } catch (err) {
      const totalTime = Date.now() - this._loadStartTime;
      console.warn(`GestureService: ❌ Failed after ${(totalTime / 1000).toFixed(1)}s:`, err.message);
      this._error = err;
      this._recognizer = null;
      throw err;
    }
  }
}

// Export singleton instance
const gestureService = new GestureRecognizerService();
export default gestureService;
