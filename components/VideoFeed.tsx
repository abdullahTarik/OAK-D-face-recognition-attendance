
import React from 'react';
import { Card, CardHeader, CardContent } from './common/Card';
import { CameraIcon } from './Icons';

export const VideoFeed: React.FC = () => {
    return (
        <Card className="overflow-hidden">
            <CardHeader title="Live Camera Feed" icon={<CameraIcon />}>
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-gray-400 font-medium">LIVE</span>
                </div>
            </CardHeader>
            <CardContent className="!p-0">
                <div className="relative aspect-video bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center overflow-hidden group">
                    {/* The backend provides an MJPEG stream at this endpoint */}
                    <img 
                        src="/video_feed" 
                        alt="Live video feed" 
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.style.display = 'none';
                          const parent = target.parentElement;
                          if(parent) {
                            const errorDiv = document.createElement('div');
                            errorDiv.className = "text-center p-8";
                            errorDiv.innerHTML = `
                                <svg class="w-16 h-16 text-red-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                                <p class="text-red-400 font-semibold">Video feed failed to load</p>
                                <p class="text-gray-500 text-sm mt-2">Please ensure the backend server is running</p>
                            `;
                            parent.appendChild(errorDiv);
                          }
                        }}
                    />
                    {/* Overlay gradient for better text visibility */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent pointer-events-none"></div>
                </div>
            </CardContent>
        </Card>
    );
};
