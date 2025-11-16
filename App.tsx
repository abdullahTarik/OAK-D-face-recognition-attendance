
import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { VideoFeed } from './components/VideoFeed';
import { ControlPanel } from './components/ControlPanel';
import { AttendanceTable } from './components/AttendanceTable';
import { UserManagement } from './components/UserManagement';
import { useAppData } from './hooks/useAppData';
import { Spinner } from './components/common/Spinner';

const App: React.FC = () => {
    const { data, loading, error } = useAppData();
    const [message, setMessage] = useState<string | null>(null);

    useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const mess = urlParams.get('mess');
        if (mess) {
            setMessage(mess);
            // Clear message from URL
            window.history.replaceState({}, document.title, window.location.pathname);
        }
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white">
                <Spinner />
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-gray-900 text-red-400">
                <div className="text-center">
                    <h2 className="text-2xl font-bold mb-2">Failed to load application data</h2>
                    <p>{error}</p>
                    <p className="mt-4 text-sm text-gray-400">Please ensure the backend server is running and accessible.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-gray-100">
            {/* Background decoration */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
                <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
            </div>
            
            <div className="relative z-10">
                <Header
                    totalRegistered={data?.totalreg ?? 0}
                    dateToday={data?.datetoday2 ?? ''}
                />
                <main className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto">
                    {message && (
                        <div className="glass rounded-xl border border-blue-500/30 bg-blue-500/10 text-blue-200 px-6 py-4 relative mb-6 fade-in shadow-lg" role="alert">
                            <div className="flex items-center gap-3">
                                <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                </svg>
                                <span className="flex-1">{message}</span>
                                <button 
                                    onClick={() => setMessage(null)} 
                                    className="text-blue-300 hover:text-blue-100 transition-colors p-1 rounded hover:bg-blue-500/20"
                                >
                                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    )}
                    
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                        {/* Left Column - Video and Controls */}
                        <div className="lg:col-span-8 flex flex-col gap-6">
                            <VideoFeed />
                            <ControlPanel autoAttendanceEnabled={data?.auto_attendance ?? false} />
                        </div>
                        
                        {/* Right Column - Attendance and Users */}
                        <div className="lg:col-span-4 flex flex-col gap-6">
                            <AttendanceTable records={data?.attendance ?? []} />
                            <UserManagement 
                                users={data?.users ?? []} 
                                settings={data?.settings} 
                            />
                        </div>
                    </div>
                </main>
            </div>
            
            <style>{`
                @keyframes blob {
                    0% { transform: translate(0px, 0px) scale(1); }
                    33% { transform: translate(30px, -50px) scale(1.1); }
                    66% { transform: translate(-20px, 20px) scale(0.9); }
                    100% { transform: translate(0px, 0px) scale(1); }
                }
                .animate-blob {
                    animation: blob 7s infinite;
                }
                .animation-delay-2000 {
                    animation-delay: 2s;
                }
                .animation-delay-4000 {
                    animation-delay: 4s;
                }
            `}</style>
        </div>
    );
};

export default App;
