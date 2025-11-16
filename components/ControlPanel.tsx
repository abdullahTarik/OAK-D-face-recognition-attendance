
import React from 'react';
import { Card, CardContent, CardHeader } from './common/Card';
import { Button } from './common/Button';
import { Input } from './common/Input';
import { PlayIcon, UserPlusIcon, UsersIcon } from './Icons';

interface ControlPanelProps {
    autoAttendanceEnabled: boolean;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({ autoAttendanceEnabled }) => {
    return (
        <Card>
            <CardHeader title="System Controls" icon={<UsersIcon />} />
            <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Enrollment Form */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 mb-4">
                            <div className="w-1 h-6 bg-gradient-to-b from-indigo-500 to-purple-500 rounded-full"></div>
                            <h4 className="font-semibold text-white text-lg">Enroll New User</h4>
                        </div>
                        <form action="/add" method="POST" className="space-y-4">
                            <div className="space-y-3">
                                <Input
                                    name="newusername"
                                    placeholder="Enter full name"
                                    required
                                />
                                <Input
                                    name="newuserid"
                                    placeholder="Enter user ID"
                                    required
                                />
                            </div>
                            <Button type="submit" className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700" icon={<UserPlusIcon />}>
                                Start Enrollment
                            </Button>
                        </form>
                    </div>

                    {/* Action Buttons */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 mb-4">
                            <div className="w-1 h-6 bg-gradient-to-b from-blue-500 to-cyan-500 rounded-full"></div>
                            <h4 className="font-semibold text-white text-lg">Quick Actions</h4>
                        </div>
                        <div className="space-y-3">
                            <Button 
                                href="/start" 
                                variant="primary" 
                                className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all" 
                                icon={<PlayIcon />}
                            >
                                Take Attendance
                            </Button>
                            <Button 
                                href="/listusers" 
                                variant="secondary" 
                                className="w-full bg-gray-700 hover:bg-gray-600" 
                                icon={<UsersIcon />}
                            >
                                View All Users
                            </Button>
                            <a 
                                href="/toggle_auto_attendance"
                                className={`w-full inline-flex items-center justify-center gap-2 px-4 py-3 font-semibold text-sm rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 shadow-lg hover:shadow-xl transform hover:scale-105 ${
                                    autoAttendanceEnabled 
                                    ? 'bg-gradient-to-r from-red-600 to-rose-600 text-white hover:from-red-700 hover:to-rose-700 focus:ring-red-500' 
                                    : 'bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-700 hover:to-emerald-700 focus:ring-green-500'
                                }`}
                            >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    {autoAttendanceEnabled ? (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    ) : (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    )}
                                </svg>
                                {autoAttendanceEnabled ? 'Disable Auto Attendance' : 'Enable Auto Attendance'}
                            </a>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};
