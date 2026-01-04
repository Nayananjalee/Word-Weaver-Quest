import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { Hands, HAND_CONNECTIONS } from '@mediapipe/hands';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { Camera } from '@mediapipe/camera_utils';

function HandGestureReader({ onGesture }) {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraError, setCameraError] = useState(null);

  const handleUserMediaError = (error) => {
    console.error("Failed to acquire camera feed:", error);
    setCameraError("Camera not found. Please connect a webcam and grant permission.");
  };

  useEffect(() => {
    if (cameraError) return;

    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    hands.onResults(onResults);

    if (webcamRef.current && webcamRef.current.video) {
      const camera = new Camera(webcamRef.current.video, {
        onFrame: async () => {
          await hands.send({ image: webcamRef.current.video });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }

    function onResults(results) {
      const canvasElement = canvasRef.current;
      const canvasCtx = canvasElement.getContext('2d');
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

      if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
          drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
          drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
          
          // Simple gesture detection (e.g., open palm)
          const isPalmOpen = landmarks[4].y < landmarks[3].y && landmarks[8].y < landmarks[6].y;
          if (isPalmOpen) {
            onGesture('wave'); // Trigger 'wave' gesture
          }
        }
      }
      canvasCtx.restore();
    }
  }, [onGesture, cameraError]);

  return (
    <div className="relative flex justify-center items-center w-[640px] h-[480px] bg-gray-200 rounded-lg">
      {!cameraError && (
        <>
          <Webcam
            ref={webcamRef}
            audio={false}
            className="absolute mx-auto left-0 right-0 text-center z-10 w-full h-full"
            onUserMediaError={handleUserMediaError}
          />
          <canvas
            ref={canvasRef}
            className="absolute mx-auto left-0 right-0 text-center z-10 w-full h-full"
          />
        </>
      )}
      {cameraError && (
        <div className="text-red-500 p-4 bg-red-100 rounded-lg text-center">
          <p className="font-bold">Camera Error</p>
          <p>{cameraError}</p>
        </div>
      )}
    </div>
  );
}

export default HandGestureReader;
