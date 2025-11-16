
import React from 'react';
import { UsersIcon } from './Icons';

interface HeaderProps {
    totalRegistered: number;
    dateToday: string;
}

export const Header: React.FC<HeaderProps> = ({ totalRegistered, dateToday }) => {
    return (
        <header className="glass border-b border-white/10 sticky top-0 z-50 shadow-lg">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-20">
                    <div className="flex items-center space-x-4">
                        <div className="relative">
                            <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-3 rounded-xl shadow-lg transform hover:scale-105 transition-transform">
                                <svg className="h-7 w-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                                </svg>
                            </div>
                            {totalRegistered > 0 && (
                                <div className="absolute -top-1 -right-1 bg-red-500 text-white text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center pulse-glow">
                                    {totalRegistered}
                                </div>
                            )}
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                                OAK-D Smart Attendance
                            </h1>
                            <p className="text-xs text-gray-400 mt-0.5">AI-Powered Face Recognition System</p>
                        </div>
                    </div>
                    <div className="flex items-center space-x-6">
                        <div className="hidden md:flex items-center space-x-3 glass px-4 py-2 rounded-lg border border-white/10">
                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                            <span className="text-sm font-medium text-gray-300">{totalRegistered} Users</span>
                        </div>
                        <div className="hidden lg:flex items-center space-x-2 text-sm glass px-4 py-2 rounded-lg border border-white/10">
                            <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            <span className="text-gray-300 font-medium">{dateToday}</span>
                        </div>
                    </div>
                </div>
            </div>
        </header>
    );
};
