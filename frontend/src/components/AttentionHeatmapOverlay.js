import React, { useEffect, useRef, useState } from 'react';
import API_BASE_URL from '../config';
import './AttentionHeatmapOverlay.css';

/**
 * AttentionHeatmapOverlay Component
 * 
 * Renders attention heatmap visualization using Canvas API.
 * Displays color-coded attention zones and gaze path.
 * 
 * Features:
 * - 10x10 grid heatmap with color gradients
 * - Gaze path visualization (spaghetti plot)
 * - Hotspot indicators
 * - Real-time updates
 * - Multiple color schemes (hot, cool, rainbow)
 */

const AttentionHeatmapOverlay = ({ 
  userId, 
  colorScheme = 'hot',
  showGazePath = true,
  showHotspots = true,
  opacity = 0.6,
  updateInterval = 5000, // Update every 5 seconds
  isVisible = true
}) => {
  const canvasRef = useRef(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch heatmap data from backend
  const fetchHeatmapData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/attention-heatmap/${userId}`);
      
      if (!response.ok) {
        console.warn('Failed to fetch heatmap data:', response.status);
        setIsLoading(false);
        return;
      }
      
      const data = await response.json();
      
      // Validate data structure
      if (data && typeof data === 'object') {
        setHeatmapData(data);
      }
      
      setIsLoading(false);
    } catch (err) {
      console.error('Failed to fetch heatmap data:', err);
      setIsLoading(false);
    }
  };

  // Initial fetch and periodic updates
  useEffect(() => {
    if (!userId || !isVisible) return;

    fetchHeatmapData();
    const interval = setInterval(fetchHeatmapData, updateInterval);

    return () => clearInterval(interval);
  }, [userId, updateInterval, isVisible]);

  // Render heatmap when data changes
  useEffect(() => {
    if (!heatmapData || !canvasRef.current) return;

    renderHeatmap();
  }, [heatmapData, colorScheme, showGazePath, showHotspots, opacity]);

  // Render heatmap to canvas
  const renderHeatmap = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    // Set canvas size to match display size
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw heatmap cells
    if (heatmapData.visualizations?.heatmap_overlay) {
      drawHeatmapCells(ctx, canvas, heatmapData.visualizations.heatmap_overlay);
    }

    // Draw gaze path
    if (showGazePath && heatmapData.visualizations?.gaze_path) {
      drawGazePath(ctx, canvas, heatmapData.visualizations.gaze_path);
    }

    // Draw hotspots
    if (showHotspots && heatmapData.heatmap_data?.hotspots) {
      drawHotspots(ctx, canvas, heatmapData.heatmap_data.hotspots);
    }
  };

  // Draw heatmap cells
  const drawHeatmapCells = (ctx, canvas, overlayData) => {
    if (!overlayData || !overlayData.cells) return;
    
    const { cells } = overlayData;

    cells.forEach(cell => {
      if (!cell || typeof cell.x === 'undefined' || typeof cell.y === 'undefined') return;
      
      const x = cell.x * canvas.width;
      const y = cell.y * canvas.height;
      const width = (cell.width || 0.1) * canvas.width;
      const height = (cell.height || 0.1) * canvas.height;

      // Get color based on intensity and scheme
      const color = getColorForIntensity(cell.intensity, colorScheme);
      
      ctx.fillStyle = color;
      ctx.globalAlpha = opacity * cell.intensity;
      ctx.fillRect(x, y, width, height);
      ctx.globalAlpha = 1.0;
    });
  };

  // Draw gaze path
  const drawGazePath = (ctx, canvas, gazePathData) => {
    if (!gazePathData) return;
    
    const { points, lines } = gazePathData;

    if (!lines || lines.length === 0) return;

    // Draw connecting lines
    ctx.strokeStyle = 'rgba(0, 150, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';

    lines.forEach((line, idx) => {
      if (!line || !line.start || !line.end) return;
      
      const startX = line.start.x * canvas.width;
      const startY = line.start.y * canvas.height;
      const endX = line.end.x * canvas.width;
      const endY = line.end.y * canvas.height;

      // Fade older lines
      const alpha = 0.5 * (idx / lines.length);
      ctx.strokeStyle = `rgba(0, 150, 255, ${alpha})`;

      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    });

    // Draw gaze points
    if (!points || points.length === 0) return;
    
    points.forEach((point, idx) => {
      if (!point || typeof point.x === 'undefined' || typeof point.y === 'undefined') return;
      
      const x = point.x * canvas.width;
      const y = point.y * canvas.height;
      const alpha = point.opacity || (idx / points.length);

      ctx.fillStyle = `rgba(0, 200, 255, ${alpha})`;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  // Draw hotspots
  const drawHotspots = (ctx, canvas, hotspots) => {
    if (!hotspots || hotspots.length === 0) return;
    
    hotspots.slice(0, 5).forEach((hotspot, idx) => {
      if (!hotspot || typeof hotspot.center_x === 'undefined' || typeof hotspot.center_y === 'undefined') return;
      
      const centerX = hotspot.center_x * canvas.width;
      const centerY = hotspot.center_y * canvas.height;

      // Draw pulsing circle
      const radius = 30 - (idx * 5);
      
      ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.stroke();

      // Draw hotspot number
      ctx.fillStyle = 'rgba(255, 215, 0, 1)';
      ctx.font = 'bold 16px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`#${idx + 1}`, centerX, centerY);
    });
  };

  // Get color based on intensity and scheme
  const getColorForIntensity = (intensity, scheme) => {
    const clampedIntensity = Math.max(0, Math.min(1, intensity));

    switch (scheme) {
      case 'hot':
        // Black -> Red -> Yellow -> White
        if (clampedIntensity < 0.33) {
          const t = clampedIntensity / 0.33;
          return `rgb(${Math.round(255 * t)}, 0, 0)`;
        } else if (clampedIntensity < 0.66) {
          const t = (clampedIntensity - 0.33) / 0.33;
          return `rgb(255, ${Math.round(255 * t)}, 0)`;
        } else {
          const t = (clampedIntensity - 0.66) / 0.34;
          return `rgb(255, 255, ${Math.round(255 * t)})`;
        }

      case 'cool':
        // Black -> Blue -> Cyan -> White
        if (clampedIntensity < 0.33) {
          const t = clampedIntensity / 0.33;
          return `rgb(0, 0, ${Math.round(255 * t)})`;
        } else if (clampedIntensity < 0.66) {
          const t = (clampedIntensity - 0.33) / 0.33;
          return `rgb(0, ${Math.round(255 * t)}, 255)`;
        } else {
          const t = (clampedIntensity - 0.66) / 0.34;
          return `rgb(${Math.round(255 * t)}, 255, 255)`;
        }

      case 'rainbow':
        // Full rainbow spectrum
        const hue = (1 - clampedIntensity) * 280; // 280 degrees (blue to red)
        return `hsl(${hue}, 100%, 50%)`;

      default:
        return `rgba(255, 0, 0, ${clampedIntensity})`;
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="attention-heatmap-overlay-container">
      <canvas
        ref={canvasRef}
        className="attention-heatmap-canvas"
        style={{ opacity: isLoading ? 0 : 1 }}
      />

      {isLoading && (
        <div className="heatmap-loading-indicator">
          <div className="loading-spinner"></div>
          <p>Loading attention heatmap...</p>
        </div>
      )}

      {/* Heatmap legend */}
      {!isLoading && heatmapData && (
        <div className="heatmap-legend">
          <div className="legend-title">Attention Intensity</div>
          <div className="legend-gradient" data-scheme={colorScheme}>
            <span className="legend-label">Low</span>
            <span className="legend-label">High</span>
          </div>
        </div>
      )}

      {/* Focus quality indicator */}
      {!isLoading && heatmapData?.visualizations?.focus_dashboard && (
        <div className="focus-quality-indicator">
          <div className="focus-score">
            {Math.round(heatmapData.visualizations.focus_dashboard.focus_quality)}
          </div>
          <div className="focus-label">
            {heatmapData.visualizations.focus_dashboard.quality_label}
          </div>
        </div>
      )}
    </div>
  );
};

export default AttentionHeatmapOverlay;
