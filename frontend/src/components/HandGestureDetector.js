import React, { useEffect, useRef, useState } from 'react';
import { useSharedCamera } from './SharedCameraProvider';
import gestureService from './GestureRecognizerService';

const HandGestureDetector = ({ onGestureDetected, isActive }) => {
  const { stream, error: cameraError, isInitialized: cameraInitialized, subscribe, retryCamera } = useSharedCamera();
  const videoElementRef = useRef(null);
  const canvasRef = useRef(null);
  const [gestureRecognizer, setGestureRecognizer] = useState(null);
  const [fingerCount, setFingerCount] = useState(0);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState(null);
  const animationFrameRef = useRef(null);
  const isActiveRef = useRef(isActive); // Track isActive in ref for timer access
  
  // Keep isActiveRef in sync with isActive prop
  useEffect(() => {
    isActiveRef.current = isActive;
  }, [isActive]);

  // Get preloaded GestureRecognizer from singleton service (instant if preloaded)
  useEffect(() => {
    let cancelled = false;

    const getRecognizer = async () => {
      // Check if already ready (instant path)
      const cached = gestureService.getSync();
      if (cached) {
        console.log('HandGestureDetector: ⚡ Using pre-cached gesture recognizer (instant!)');
        setGestureRecognizer(cached);
        setIsReady(true);
        return;
      }

      // Otherwise wait for preload to finish
      console.log('HandGestureDetector: ⏳ Waiting for gesture recognizer preload...');
      try {
        const recognizer = await gestureService.get();
        if (cancelled) return;
        if (recognizer) {
          console.log('HandGestureDetector: ✅ Gesture recognizer ready');
          setGestureRecognizer(recognizer);
          setIsReady(true);
        } else {
          console.warn('HandGestureDetector: Gesture recognizer unavailable, gesture control disabled');
          setIsReady(false);
        }
      } catch (err) {
        if (cancelled) return;
        console.warn('HandGestureDetector: Could not get gesture recognizer:', err.message);
        setIsReady(false);
      }
    };

    getRecognizer();

    return () => {
      cancelled = true;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Use shared camera - initialize video as soon as stream is ready
  // Keep camera alive even when not actively detecting gestures
  useEffect(() => {
    if (!isReady || !cameraInitialized || !stream) {
      if (cameraError) {
        setError(cameraError);
      }
      return;
    }

    const unsubscribe = subscribe('HandGestureDetector');
    console.log('HandGestureDetector: Using shared camera stream');

    // Create video element for processing
    const video = document.createElement('video');
    video.srcObject = stream;
    video.autoplay = true;
    video.playsInline = true;
    videoElementRef.current = video;

    video.play().then(() => {
      console.log('HandGestureDetector: Video playing, starting detection loop');
      detectGestures();
    }).catch(err => {
      console.warn('HandGestureDetector: Video play failed:', err);
    });

    return () => {
      unsubscribe();
      // Don't stop stream - managed by SharedCameraProvider
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isReady, stream, cameraInitialized, cameraError]);

  // Reset finger count when component becomes inactive
  useEffect(() => {
    if (!isActive) {
      setFingerCount(0);
    }
  }, [isActive]);

  // Count extended fingers from hand landmarks (SIMPLE and RELIABLE algorithm)
  const countFingers = (landmarks) => {
    if (!landmarks || landmarks.length === 0) return 0;

    const hand = landmarks[0]; // First hand detected
    let count = 0;

    // MediaPipe hand landmark indices:
    // 0: Wrist, 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky
    
    // SIMPLE APPROACH: Compare fingertip Y position with MCP (knuckle) Y position
    // In camera coordinates, Y increases downward, so extended finger has LOWER Y value
    
    // 1. Check THUMB (compare tip[4] with IP joint[3])
    // Thumb extended if tip is farther from wrist than IP joint
    const thumbTip = hand[4];
    const thumbIp = hand[3];
    const wrist = hand[0];
    
    const thumbTipDist = Math.hypot(thumbTip.x - wrist.x, thumbTip.y - wrist.y);
    const thumbIpDist = Math.hypot(thumbIp.x - wrist.x, thumbIp.y - wrist.y);
    
    if (thumbTipDist > thumbIpDist * 1.3) {
      count++;
    }

    // 2. Check INDEX finger - tip[8] vs MCP[5]
    if (hand[8].y < hand[5].y - 0.08) count++;
    
    // 3. Check MIDDLE finger - tip[12] vs MCP[9]
    if (hand[12].y < hand[9].y - 0.08) count++;
    
    // 4. Check RING finger - tip[16] vs MCP[13]
    if (hand[16].y < hand[13].y - 0.08) count++;
    
    // 5. Check PINKY finger - tip[20] vs MCP[17]
    if (hand[20].y < hand[17].y - 0.08) count++;

    return count;
  };



  // Detect gestures continuously - always draw video, only detect when active
  const detectGestures = () => {
    if (!gestureRecognizer || !videoElementRef.current) return;

    const video = videoElementRef.current;
    const canvas = canvasRef.current;

    if (video.readyState === video.HAVE_ENOUGH_DATA && canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      // Always draw video to canvas (keeps camera feed visible)
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Only run gesture detection when active
      if (isActiveRef.current) {
        const now = Date.now();
        const results = gestureRecognizer.recognizeForVideo(video, now);

        if (results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0];
          const hand = landmarks;

          // Draw all landmarks as small green circles
          ctx.fillStyle = '#00FF00';
          landmarks.forEach(landmark => {
            ctx.beginPath();
            ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 3, 0, 2 * Math.PI);
            ctx.fill();
          });

          // Draw connections between joints
          ctx.strokeStyle = '#00FF00';
          ctx.lineWidth = 2;
          const connections = [
            [0,1],[1,2],[2,3],[3,4],
            [0,5],[5,6],[6,7],[7,8],
            [0,9],[9,10],[10,11],[11,12],
            [0,13],[13,14],[14,15],[15,16],
            [0,17],[17,18],[18,19],[19,20],
            [5,9],[9,13],[13,17]
          ];
          connections.forEach(([start, end]) => {
            ctx.beginPath();
            ctx.moveTo(hand[start].x * canvas.width, hand[start].y * canvas.height);
            ctx.lineTo(hand[end].x * canvas.width, hand[end].y * canvas.height);
            ctx.stroke();
          });

          // Highlight fingertips of extended fingers in RED
          const wrist = hand[0];
          const thumbTip = hand[4];
          const thumbIp = hand[3];
          const thumbTipDist = Math.hypot(thumbTip.x - wrist.x, thumbTip.y - wrist.y);
          const thumbIpDist = Math.hypot(thumbIp.x - wrist.x, thumbIp.y - wrist.y);

          if (thumbTipDist > thumbIpDist * 1.3) {
            ctx.fillStyle = '#FF0000';
            ctx.beginPath();
            ctx.arc(thumbTip.x * canvas.width, thumbTip.y * canvas.height, 12, 0, 2 * Math.PI);
            ctx.fill();
            ctx.fillStyle = '#FFFFFF';
            ctx.beginPath();
            ctx.arc(thumbTip.x * canvas.width, thumbTip.y * canvas.height, 8, 0, 2 * Math.PI);
            ctx.fill();
            ctx.fillStyle = '#FF0000';
            ctx.font = 'bold 18px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('👍', thumbTip.x * canvas.width, thumbTip.y * canvas.height + 6);
          }

          const fingerData = [
            { tip: 8, mcp: 5, num: '1' },
            { tip: 12, mcp: 9, num: '2' },
            { tip: 16, mcp: 13, num: '3' },
            { tip: 20, mcp: 17, num: '4' }
          ];
          fingerData.forEach(finger => {
            if (hand[finger.tip].y < hand[finger.mcp].y - 0.08) {
              ctx.fillStyle = '#FF0000';
              ctx.beginPath();
              ctx.arc(hand[finger.tip].x * canvas.width, hand[finger.tip].y * canvas.height, 12, 0, 2 * Math.PI);
              ctx.fill();
              ctx.fillStyle = '#FFFFFF';
              ctx.beginPath();
              ctx.arc(hand[finger.tip].x * canvas.width, hand[finger.tip].y * canvas.height, 8, 0, 2 * Math.PI);
              ctx.fill();
              ctx.fillStyle = '#FF0000';
              ctx.font = 'bold 18px Arial';
              ctx.textAlign = 'center';
              ctx.fillText(finger.num, hand[finger.tip].x * canvas.width, hand[finger.tip].y * canvas.height + 6);
            }
          });

          // Count fingers and emit
          const count = countFingers(results.landmarks);
          setFingerCount(count);
          if (onGestureDetected) onGestureDetected(count);
        } else {
          // No hand detected
          setFingerCount(0);
          if (onGestureDetected) onGestureDetected(0);
        }
      }
      // When not active, just shows the camera feed (no detection overlay)
    }

    animationFrameRef.current = requestAnimationFrame(detectGestures);
  };

  // Always show the camera panel (standby or active mode)
  // This keeps the camera stream alive and MediaPipe initialized across phases

  // If gesture recognizer or camera failed, show status with retry
  if (error || cameraError || !isReady) {
    const hasError = error || cameraError;
    return (
      <div className="gesture-panel gesture-panel-inactive">
        <div className="gesture-panel-status">
          <span className="gesture-status-icon">{hasError ? '📷' : '🖐️'}</span>
          <p className="gesture-status-text">
            {hasError
              ? 'කැමරාව සම්බන්ධ කළ නොහැක'
              : cameraInitialized
                ? 'ඇඟිලි හඳුනාගැනීම පූරණය වෙමින්...'
                : 'කැමරාව ආරම්භ වෙමින්...'}
          </p>
          <p className="gesture-status-sub" style={{ fontSize: '0.65rem', maxWidth: 240, lineHeight: 1.3 }}>
            {hasError ? (error || cameraError) : 'Hand gestures loading...'}
          </p>
          {hasError && (
            <button
              onClick={() => {
                setError(null);
                if (retryCamera) retryCamera();
              }}
              style={{
                marginTop: 8,
                padding: '6px 16px',
                borderRadius: 12,
                border: 'none',
                background: 'linear-gradient(135deg, #ffa726, #ff9800)',
                color: 'white',
                fontWeight: 700,
                fontSize: '0.8rem',
                cursor: 'pointer',
                boxShadow: '0 2px 8px rgba(255,152,0,0.3)'
              }}
            >
              🔄 නැවත උත්සාහ කරන්න (Retry)
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={`gesture-panel ${!isActive ? 'gesture-panel-standby' : ''}`}>
      {/* Camera Feed - Always visible */}
      <div className="gesture-camera-feed">
        <canvas
          ref={canvasRef}
          width="640"
          height="480"
          className="gesture-canvas"
          style={{ transform: 'scaleX(-1)' }}
        />

        {/* Big finger count badge - always visible */}
        <div className={`gesture-finger-badge ${isActive && fingerCount > 0 ? 'gesture-finger-active' : 'gesture-finger-waiting'}`}>
          <span className="gesture-finger-count">{isActive ? fingerCount : '👁️'}</span>
          <span className="gesture-finger-label">{isActive ? '🖐️' : ''}</span>
        </div>

        {/* Hand detection status */}
        <div className={`gesture-detect-status ${!isActive ? 'standby' : fingerCount > 0 ? 'detected' : 'searching'}`}>
          {!isActive
            ? '📷 කැමරාව සූදානම්'
            : fingerCount > 0
              ? '✅ අත හඳුනාගෙන ඇත'
              : '🔍 අත පෙන්වන්න...'}
        </div>
      </div>

      {/* Finger guide strip - shows which fingers = which answer */}
      <div className="gesture-finger-guide">
        {[1, 2, 3, 4].map(num => (
          <div key={num} className={`gesture-guide-item ${isActive && fingerCount === num ? 'guide-active' : ''}`}>
            <span className="guide-num">{num}</span>
            <span className="guide-hand">{['☝️', '✌️', '🤟', '🖖'][num - 1]}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HandGestureDetector;
