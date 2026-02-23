import React, { createContext, useContext, useState, useEffect, useRef } from 'react';

/**
 * SharedCameraProvider
 * 
 * Provides a single camera stream that can be shared by multiple components
 * (GazeTracker, HandGestureDetector, etc.) to avoid conflicts.
 */

const CameraContext = createContext(null);

export const useSharedCamera = () => {
  const context = useContext(CameraContext);
  if (!context) {
    throw new Error('useSharedCamera must be used within SharedCameraProvider');
  }
  return context;
};

export const SharedCameraProvider = ({ children }) => {
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const videoRef = useRef(null);
  const subscribersRef = useRef(new Set());

  // Initialize camera once
  useEffect(() => {
    const initializeCamera = async () => {
      try {
        console.log('SharedCamera: Requesting camera access...');
        
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          }
        });

        console.log('SharedCamera: Camera access granted!');
        setStream(mediaStream);
        setIsInitialized(true);
        setError(null);
      } catch (err) {
        console.error('SharedCamera: Failed to initialize camera:', err);
        
        let errorMessage = 'Camera initialization failed.';
        if (err.name === 'NotAllowedError') {
          errorMessage = 'කැමරා අවසර දෙන්න (Camera access denied). Please allow camera in browser settings.';
        } else if (err.name === 'NotFoundError') {
          errorMessage = 'කැමරාවක් හමු නොවීය (No camera found).';
        } else if (err.name === 'NotReadableError') {
          errorMessage = 'කැමරාව වෙනත් යෙදුමකින් භාවිතා වේ (Camera in use by another app).';
        } else {
          errorMessage = `කැමරා දෝෂය: ${err.message}`;
        }
        
        setError(errorMessage);
        setIsInitialized(false);
      }
    };

    initializeCamera();

    // Cleanup on unmount
    return () => {
      if (stream) {
        console.log('SharedCamera: Stopping camera stream...');
        stream.getTracks().forEach(track => track.stop());
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Set video srcObject when stream is available
  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(err => {
        console.error('SharedCamera: Error playing video:', err);
      });
    }
  }, [stream]);

  // Subscribe to camera stream
  const subscribe = (componentName) => {
    console.log(`SharedCamera: ${componentName} subscribed`);
    subscribersRef.current.add(componentName);
    
    return () => {
      console.log(`SharedCamera: ${componentName} unsubscribed`);
      subscribersRef.current.delete(componentName);
    };
  };

  const value = {
    stream,
    error,
    isInitialized,
    subscribe
  };

  return (
    <CameraContext.Provider value={value}>
      {children}
      
      {/* Hidden video element for camera stream */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: 'none' }}
      />
    </CameraContext.Provider>
  );
};

export default SharedCameraProvider;
