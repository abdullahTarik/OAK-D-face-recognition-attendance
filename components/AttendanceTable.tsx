
import React from 'react';
import { Card, CardHeader, CardContent } from './common/Card';
import { ListIcon } from './Icons';
import { AttendanceRecord } from '../types';

interface AttendanceTableProps {
    records: AttendanceRecord[];
}

export const AttendanceTable: React.FC<AttendanceTableProps> = ({ records }) => {
    return (
        <Card>
            <CardHeader title="Today's Attendance" icon={<ListIcon />}>
                <div className="flex items-center gap-2">
                    <span className="text-sm font-bold bg-gradient-to-r from-green-500 to-emerald-500 text-white px-3 py-1.5 rounded-full shadow-lg">
                        {records.length} Present
                    </span>
                </div>
            </CardHeader>
            <CardContent className="!p-0">
                <div className="overflow-x-auto max-h-96 custom-scrollbar">
                    {records.length > 0 ? (
                        <table className="w-full text-sm">
                            <thead className="text-xs text-gray-300 uppercase bg-gradient-to-r from-gray-800/80 to-gray-700/80 sticky top-0 backdrop-blur-sm">
                                <tr>
                                    <th scope="col" className="px-6 py-4 font-semibold">Name</th>
                                    <th scope="col" className="px-6 py-4 font-semibold">Roll ID</th>
                                    <th scope="col" className="px-6 py-4 font-semibold">Time</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/5">
                                {records.map((record, index) => (
                                    <tr 
                                        key={`${record.roll}-${index}`} 
                                        className="hover:bg-white/5 transition-colors group"
                                    >
                                        <td className="px-6 py-4 font-medium text-white group-hover:text-indigo-300 transition-colors">
                                            <div className="flex items-center gap-2">
                                                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                                                {record.name}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-gray-300">{record.roll}</td>
                                        <td className="px-6 py-4 text-gray-400 font-mono text-xs">{record.time}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div className="text-center py-12 px-6">
                            <div className="w-16 h-16 bg-gray-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
                                <svg className="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                            </div>
                            <p className="text-gray-400 font-medium">No attendance records</p>
                            <p className="text-gray-500 text-sm mt-1">Records will appear here after attendance is taken</p>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
};
