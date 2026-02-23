import React, { useEffect, useRef, useState } from 'react';
import { GestureRecognizer, FilesetResolver } from '@mediapipe/tasks-vision';
import { useSharedCamera } from './SharedCameraProvider';

const HandGestureDetector = ({ onGestureDetected, isActive }) => {
  const { stream, error: cameraError, isInitialized: cameraInitialized, subscribe } = useSharedCamera();
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

  // Initialize MediaPipe Gesture Recognizer
  useEffect(() => {
    const initializeGestureRecognizer = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        
        const recognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        
        setGestureRecognizer(recognizer);
        setIsReady(true);
        console.log('HandGestureDetector: Gesture recognizer initialized successfully');
      } catch (err) {
        console.warn('HandGestureDetector: Could not initialize gesture recognizer:', err.message);
        console.log('HandGestureDetector: Hand gesture control will be disabled. Gaze tracking is still active.');
        // Don't set error - fail silently and let gaze tracking work
        setIsReady(false);
      }
    };

    initializeGestureRecognizer();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Use shared camera
  useEffect(() => {
    if (!isActive || !isReady || !cameraInitialized || !stream) {
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
    video.autoPlay = true;
    video.playsInline = true;
    videoElementRef.current = video;

    video.play().then(() => {
      console.log('HandGestureDetector: Video playing, starting detection');
      detectGestures();
    });

    return () => {
      unsubscribe();
      // Don't stop stream - managed by SharedCameraProvider
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive, isReady, stream, cameraInitialized, cameraError]);

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



  // Detect gestures continuously
  const detectGestures = () => {
    if (!gestureRecognizer || !videoElementRef.current || !isActive) return;

    const video = videoElementRef.current;
    const canvas = canvasRef.current;

    if (video.readyState === video.HAVE_ENOUGH_DATA) {
      const now = Date.now();
      const results = gestureRecognizer.recognizeForVideo(video, now);

      // Draw video to canvas
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Draw hand landmarks
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
            [0,1],[1,2],[2,3],[3,4], // Thumb
            [0,5],[5,6],[6,7],[7,8], // Index
            [0,9],[9,10],[10,11],[11,12], // Middle
            [0,13],[13,14],[14,15],[15,16], // Ring
            [0,17],[17,18],[18,19],[19,20], // Pinky
            [5,9],[9,13],[13,17] // Palm
          ];
          connections.forEach(([start, end]) => {
            ctx.beginPath();
            ctx.moveTo(hand[start].x * canvas.width, hand[start].y * canvas.height);
            ctx.lineTo(hand[end].x * canvas.width, hand[end].y * canvas.height);
            ctx.stroke();
          });

          // Highlight fingertips of extended fingers in RED (using simple logic)
          const wrist = hand[0];
          
          // Check and highlight thumb
          const thumbTip = hand[4];
          const thumbIp = hand[3];
          const thumbTipDist = Math.hypot(thumbTip.x - wrist.x, thumbTip.y - wrist.y);
          const thumbIpDist = Math.hypot(thumbIp.x - wrist.x, thumbIp.y - wrist.y);
          
          if (thumbTipDist > thumbIpDist * 1.3) {
            ctx.fillStyle = '#FF0000';
            ctx.beginPath();
            ctx.arc(thumbTip.x * canvas.width, thumbTip.y * canvas.height, 12, 0, 2 * Math.PI);
            ctx.fill();
            // Add white circle in center
            ctx.fillStyle = '#FFFFFF';
            ctx.beginPath();
            ctx.arc(thumbTip.x * canvas.width, thumbTip.y * canvas.height, 8, 0, 2 * Math.PI);
            ctx.fill();
            // Add text
            ctx.fillStyle = '#FF0000';
            ctx.font = 'bold 18px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('👍', thumbTip.x * canvas.width, thumbTip.y * canvas.height + 6);
          }

          // Check and highlight other fingers - simple Y comparison
          const fingerData = [
            { tip: 8, mcp: 5, num: '1' },   // Index
            { tip: 12, mcp: 9, num: '2' },  // Middle
            { tip: 16, mcp: 13, num: '3' }, // Ring
            { tip: 20, mcp: 17, num: '4' }  // Pinky
          ];
          
          fingerData.forEach(finger => {
            if (hand[finger.tip].y < hand[finger.mcp].y - 0.08) {
              // Draw red circle
              ctx.fillStyle = '#FF0000';
              ctx.beginPath();
              ctx.arc(hand[finger.tip].x * canvas.width, hand[finger.tip].y * canvas.height, 12, 0, 2 * Math.PI);
              ctx.fill();
              // Draw white circle inside
              ctx.fillStyle = '#FFFFFF';
              ctx.beginPath();
              ctx.arc(hand[finger.tip].x * canvas.width, hand[finger.tip].y * canvas.height, 8, 0, 2 * Math.PI);
              ctx.fill();
              // Add number label
              ctx.fillStyle = '#FF0000';
              ctx.font = 'bold 18px Arial';
              ctx.textAlign = 'center';
              ctx.fillText(finger.num, hand[finger.tip].x * canvas.width, hand[finger.tip].y * canvas.height + 6);
            }
          });

          // Count fingers and emit real-time updates
          const count = countFingers(results.landmarks);
          setFingerCount(count);

          // Emit continuous updates to parent component
          // Parent will handle the confirmation logic and progress
          if (isActive && onGestureDetected) {
            onGestureDetected(count); // Emit every frame (including 0)
          }
        } else {
          // No hand detected
          setFingerCount(0);
          if (isActive && onGestureDetected) {
            onGestureDetected(0); // Emit 0 when no hand
          }
        }
      }
    }

    animationFrameRef.current = requestAnimationFrame(detectGestures);
  };

  // Don't show error UI - fail silently if gesture recognizer doesn't load
  // Gaze tracking will still work

  if (!isActive) {
    return null;
  }

  // If gesture recognizer failed to initialize, show simplified message
  if (error || !isReady) {
    return (
      <div className="bg-gradient-to-br from-gray-100 to-gray-200 p-4 rounded-2xl shadow-lg">
        <div className="text-center text-gray-600">
          <p className="text-sm">👁️ ඇස් ලුහුබැඳීම සක්‍රීයයි (Gaze tracking active)</p>
          <p className="text-xs mt-1 text-gray-500">අත් ඉශාරා අක්‍රියයි (Hand gestures unavailable)</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-blue-100 to-purple-100 p-4 rounded-2xl shadow-lg">
      <h3 className="text-xl font-bold text-purple-800 mb-3 text-center flex items-center justify-center gap-2">
        <span>🖐️</span> ඔබේ ඇඟිලි පෙන්වන්න
      </h3>
      
      {/* Camera preview */}
      <div className="relative mb-3">
        <canvas
          ref={canvasRef}
          width="640"
          height="480"
          className="w-full rounded-xl border-4 border-purple-400 shadow-lg"
          style={{ transform: 'scaleX(-1)' }} // Mirror the video
        />
        
        {/* Finger count overlay */}
        {fingerCount > 0 && (
          <div className="absolute top-2 right-2 bg-blue-500 text-white px-4 py-2 rounded-full font-bold text-xl shadow-xl">
            {fingerCount} 🖐️
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="bg-white bg-opacity-90 p-3 rounded-lg">
        <p className="text-center text-gray-700 font-semibold mb-2">
          පිළිතුරු තෝරන්න (Choose answer):
        </p>
        <div className="grid grid-cols-4 gap-2 text-center text-sm">
          <div className="bg-blue-200 p-2 rounded">
            <div className="text-2xl">1️⃣</div>
            <div className="text-xs text-gray-700">පළමු</div>
          </div>
          <div className="bg-green-200 p-2 rounded">
            <div className="text-2xl">2️⃣</div>
            <div className="text-xs text-gray-700">දෙවන</div>
          </div>
          <div className="bg-yellow-200 p-2 rounded">
            <div className="text-2xl">3️⃣</div>
            <div className="text-xs text-gray-700">තෙවන</div>
          </div>
          <div className="bg-pink-200 p-2 rounded">
            <div className="text-2xl">4️⃣</div>
            <div className="text-xs text-gray-700">සිව්වන</div>
          </div>
        </div>
      </div>

      {!isReady && (
        <div className="text-center mt-2">
          <p className="text-blue-600 animate-pulse">කැමරාව සකස් වෙමින්... (Loading camera...)</p>
        </div>
      )}
    </div>
  );
};

export default HandGestureDetector;
