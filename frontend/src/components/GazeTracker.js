import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useSharedCamera } from './SharedCameraProvider';
import API_BASE_URL from '../config';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-converter';
import './GazeTracker.css';

/**
 * GazeTracker Component
 * 
 * Real-time eye gaze tracking using TensorFlow.js Face Landmarks Detection.
 * Uses shared camera stream to avoid conflicts with other components.
 * 
 * Features:
 * - Eye landmark detection (468 facial landmarks)
 * - Pupil position calculation
 * - Gaze-to-screen coordinate mapping
 * - 9-point calibration system
 * - Real-time gaze recording (10 Hz)
 */

const GazeTracker = ({ userId, onGazePoint, showVisualization = false, isActive = true }) => {
  const { stream, error: cameraError, isInitialized: cameraInitialized, subscribe } = useSharedCamera();
  
  const canvasRef = useRef(null);
  const gazeOverlayRef = useRef(null);
  const detectorRef = useRef(null);
  const animationFrameRef = useRef(null);
  const gazeIntervalRef = useRef(null);
  const videoElementRef = useRef(null);

  const [isInitialized, setIsInitialized] = useState(false);
  const [isCalibrated, setIsCalibrated] = useState(true); // Auto-skip calibration for children
  const [calibrationStep, setCalibrationStep] = useState(0);
  const [, setError] = useState(null);
  const [currentGaze, setCurrentGaze] = useState({ x: 0.5, y: 0.5 });
  const [gazeConfidence, setGazeConfidence] = useState(0);

  // Calibration data storage
  const calibrationData = useRef([]);
  const calibrationPoints = [
    { x: 0.1, y: 0.1 }, // Top-left
    { x: 0.5, y: 0.1 }, // Top-center
    { x: 0.9, y: 0.1 }, // Top-right
    { x: 0.1, y: 0.5 }, // Middle-left
    { x: 0.5, y: 0.5 }, // Center
    { x: 0.9, y: 0.5 }, // Middle-right
    { x: 0.1, y: 0.9 }, // Bottom-left
    { x: 0.5, y: 0.9 }, // Bottom-center
    { x: 0.9, y: 0.9 }, // Bottom-right
  ];

  // Initialize TensorFlow.js Face Landmarks Detection with shared camera
  useEffect(() => {
    if (!cameraInitialized || !stream) {
      if (cameraError) {
        setError(cameraError);
      }
      return;
    }

    const unsubscribe = subscribe('GazeTracker');
    
    const initializeDetector = async () => {
      try {
        console.log('GazeTracker: Initializing with shared camera...');
        console.log('GazeTracker: Waiting for TensorFlow.js backend...');
        
        // Wait for TensorFlow.js backend to be ready
        await tf.ready();
        console.log('GazeTracker: TensorFlow.js backend ready:', tf.getBackend());
        
        console.log('GazeTracker: Initializing Face Landmarks Detection...');

        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        const detectorConfig = {
          runtime: 'tfjs',
          refineLandmarks: true,
          maxFaces: 1
        };

        const detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
        detectorRef.current = detector;

        // Create video element for processing
        const video = document.createElement('video');
        video.srcObject = stream;
        video.autoPlay = true;
        video.playsInline = true;
        videoElementRef.current = video;

        await video.play();
        console.log('GazeTracker: TensorFlow.js detector initialized successfully!');
        setIsInitialized(true);
        startProcessing();

      } catch (err) {
        console.error('GazeTracker: Failed to initialize detector:', err);
        // Only log the error - don't display it to users as it's not critical for gameplay
        // The gaze tracking is an optional enhancement feature
        console.warn('Gaze tracking disabled due to initialization error:', err.message || err.name);
        setIsInitialized(true);
        setIsCalibrated(true);
        // Don't set error state to avoid showing error banner
      }
    };

    initializeDetector();

    return () => {
      cleanup();
      unsubscribe();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream, cameraInitialized, cameraError]);

  // Cleanup resources
  const cleanup = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (gazeIntervalRef.current) {
      clearInterval(gazeIntervalRef.current);
    }
    // Don't stop the stream - it's managed by SharedCameraProvider
  };

  // Start processing video frames with TensorFlow.js
  const startProcessing = () => {
    const processFrame = async () => {
      if (videoElementRef.current && detectorRef.current && videoElementRef.current.readyState === 4) {
        const faces = await detectorRef.current.estimateFaces(videoElementRef.current, {
          flipHorizontal: false
        });
        
        if (faces && faces.length > 0) {
          const keypoints = faces[0].keypoints;
          onFaceLandmarksDetected(keypoints);
        } else {
          setGazeConfidence(0);
        }
      }
      animationFrameRef.current = requestAnimationFrame(processFrame);
    };

    processFrame();
  };

  // Process detected face landmarks
  const onFaceLandmarksDetected = useCallback((keypoints) => {
    if (!keypoints || keypoints.length === 0) {
      setGazeConfidence(0);
      return;
    }

    // Convert keypoints array to landmarks object
    const landmarks = keypoints.map(kp => ({ x: kp.x, y: kp.y, z: kp.z || 0 }));

    // Calculate gaze from eye landmarks
    const gaze = calculateGazeFromLandmarks(landmarks);

    if (gaze) {
      setCurrentGaze(gaze.position);
      setGazeConfidence(gaze.confidence);

      // Draw visualization if enabled
      if (showVisualization && canvasRef.current) {
        drawFaceMeshVisualization(landmarks, gaze);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showVisualization]);

  // Calculate gaze position from facial landmarks
  const calculateGazeFromLandmarks = (landmarks) => {
    // Eye landmarks indices (MediaPipe Face Mesh)
    const leftEyeIndices = {
      innerCorner: 133,
      outerCorner: 33,
      topLid: 159,
      bottomLid: 145,
      pupil: 468 // Iris landmark
    };

    const rightEyeIndices = {
      innerCorner: 362,
      outerCorner: 263,
      topLid: 386,
      bottomLid: 374,
      pupil: 473 // Iris landmark
    };

    // Get left eye pupil position
    const leftPupil = landmarks[leftEyeIndices.pupil];
    const leftInner = landmarks[leftEyeIndices.innerCorner];
    const leftOuter = landmarks[leftEyeIndices.outerCorner];
    const leftTop = landmarks[leftEyeIndices.topLid];
    const leftBottom = landmarks[leftEyeIndices.bottomLid];

    // Get right eye pupil position
    const rightPupil = landmarks[rightEyeIndices.pupil];
    const rightInner = landmarks[rightEyeIndices.innerCorner];
    const rightOuter = landmarks[rightEyeIndices.outerCorner];

    if (!leftPupil || !rightPupil) {
      return null;
    }

    // Calculate horizontal gaze ratio (0 = left, 1 = right)
    const leftHorizontalRatio = (leftPupil.x - leftOuter.x) / (leftInner.x - leftOuter.x);
    const rightHorizontalRatio = (rightPupil.x - rightOuter.x) / (rightInner.x - rightOuter.x);
    const horizontalRatio = (leftHorizontalRatio + rightHorizontalRatio) / 2;

    // Calculate vertical gaze ratio (0 = up, 1 = down)
    const verticalRatio = (leftPupil.y - leftTop.y) / (leftBottom.y - leftTop.y);

    // Apply calibration if available
    let gazeX = horizontalRatio;
    let gazeY = verticalRatio;

    if (isCalibrated && calibrationData.current.length === 9) {
      const calibrated = applyCalibratedMapping(gazeX, gazeY);
      gazeX = calibrated.x;
      gazeY = calibrated.y;
    }

    // Clamp to [0, 1]
    gazeX = Math.max(0, Math.min(1, gazeX));
    gazeY = Math.max(0, Math.min(1, gazeY));

    // Calculate confidence based on eye openness
    const eyeOpenness = Math.abs(leftTop.y - leftBottom.y);
    const confidence = Math.min(1, eyeOpenness * 20);

    return {
      position: { x: gazeX, y: gazeY },
      confidence: confidence
    };
  };

  // Apply calibration mapping
  const applyCalibratedMapping = (rawX, rawY) => {
    // Simple linear interpolation using calibration points
    // In production, use polynomial regression for better accuracy

    if (calibrationData.current.length < 9) {
      return { x: rawX, y: rawY };
    }

    // Find nearest calibration points and interpolate
    let totalWeight = 0;
    let weightedX = 0;
    let weightedY = 0;

    calibrationData.current.forEach((cal, idx) => {
      const dx = rawX - cal.rawX;
      const dy = rawY - cal.rawY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const weight = 1 / (distance + 0.001); // Avoid division by zero

      totalWeight += weight;
      weightedX += weight * cal.screenX;
      weightedY += weight * cal.screenY;
    });

    return {
      x: weightedX / totalWeight,
      y: weightedY / totalWeight
    };
  };

  // Draw visualization
  const drawFaceMeshVisualization = (landmarks, gaze) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = videoElementRef.current?.videoWidth || 640;
    canvas.height = videoElementRef.current?.videoHeight || 480;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw facial landmarks (optional, for debugging)
    ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
    landmarks.forEach(landmark => {
      ctx.beginPath();
      ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 1, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw gaze indicator
    ctx.fillStyle = `rgba(255, 0, 0, ${gaze.confidence})`;
    ctx.beginPath();
    ctx.arc(gaze.position.x * canvas.width, gaze.position.y * canvas.height, 10, 0, 2 * Math.PI);
    ctx.fill();
  };

  // Record gaze point to backend
  const recordGazePoint = useCallback(async () => {
    if (!isActive || !isCalibrated || gazeConfidence < 0.3) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/track-gaze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          x: currentGaze.x,
          y: currentGaze.y,
          confidence: gazeConfidence
        })
      });

      const data = await response.json();

      if (onGazePoint) {
        onGazePoint(data);
      }

      // Update gaze overlay
      if (gazeOverlayRef.current) {
        gazeOverlayRef.current.style.left = `${currentGaze.x * 100}%`;
        gazeOverlayRef.current.style.top = `${currentGaze.y * 100}%`;
        gazeOverlayRef.current.style.opacity = gazeConfidence;
      }
    } catch (err) {
      console.error('Failed to record gaze:', err);
    }
  }, [userId, currentGaze, gazeConfidence, isActive, isCalibrated, onGazePoint]);

  // Start gaze recording interval (10 Hz = 100ms)
  useEffect(() => {
    if (isCalibrated && isActive) {
      gazeIntervalRef.current = setInterval(recordGazePoint, 100);
    }

    return () => {
      if (gazeIntervalRef.current) {
        clearInterval(gazeIntervalRef.current);
      }
    };
  }, [isCalibrated, isActive, recordGazePoint]);

  // Handle calibration point click
  const handleCalibrationClick = () => {
    if (calibrationStep >= calibrationPoints.length) {
      return;
    }

    const currentPoint = calibrationPoints[calibrationStep];

    // Record calibration data
    calibrationData.current.push({
      screenX: currentPoint.x,
      screenY: currentPoint.y,
      rawX: currentGaze.x,
      rawY: currentGaze.y
    });

    if (calibrationStep === calibrationPoints.length - 1) {
      setIsCalibrated(true);
      setCalibrationStep(0);
    } else {
      setCalibrationStep(calibrationStep + 1);
    }
  };

  // Skip calibration (use raw gaze)
  const skipCalibration = () => {
    setIsCalibrated(true);
    setCalibrationStep(0);
  };

  // Render calibration screen
  if (isInitialized && !isCalibrated) {
    const currentPoint = calibrationPoints[calibrationStep];

    return (
      <div className="gaze-calibration-screen">
        <div className="calibration-overlay">
          <h2>Eye Tracking Calibration</h2>
          <p>Look at the red circle and click when ready</p>
          <p className="calibration-progress">
            Point {calibrationStep + 1} of {calibrationPoints.length}
          </p>

          <div
            className="calibration-target"
            style={{
              left: `${currentPoint.x * 100}%`,
              top: `${currentPoint.y * 100}%`
            }}
            onClick={handleCalibrationClick}
          >
            <div className="calibration-target-inner"></div>
          </div>

          <button onClick={skipCalibration} className="skip-calibration-btn">
            Skip Calibration
          </button>
        </div>
      </div>
    );
  }

  // Render gaze overlay (when tracking is active)
  return (
    <div className="gaze-tracker-container">
      {showVisualization && (
        <canvas ref={canvasRef} className="gaze-canvas-overlay" />
      )}

      {/* Gaze cursor indicator */}
      <div
        ref={gazeOverlayRef}
        className="gaze-cursor"
        style={{
          left: `${currentGaze.x * 100}%`,
          top: `${currentGaze.y * 100}%`,
          opacity: gazeConfidence
        }}
      />

      {/* Error message - Hidden since gaze tracking is optional */}
      {/* Errors are logged to console instead of displaying to users */}
    </div>
  );
};

export default GazeTracker;
