
import React from 'react';
import { User } from '../types';
import { Button } from './common/Button';
import { TrashIcon } from './Icons';

interface UserListProps {
    users: User[];
}

export const UserList: React.FC<UserListProps> = ({ users }) => {
    const handleDelete = (userIdentifier: string) => {
        if (window.confirm(`Are you sure you want to delete user ${userIdentifier.replace('_', ' ')}? This action cannot be undone.`)) {
            window.location.href = `/deleteuser?user=${userIdentifier}`;
        }
    };

    return (
        <div className="space-y-3 max-h-96 overflow-y-auto pr-2 custom-scrollbar">
            {users.length > 0 ? users.map(user => (
                <div key={user.identifier} className="flex items-center justify-between glass border border-white/10 p-4 rounded-xl hover:border-indigo-500/50 transition-all group">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">
                            {user.name.charAt(0).toUpperCase()}
                        </div>
                        <div>
                            <p className="font-semibold text-white group-hover:text-indigo-300 transition-colors">{user.name}</p>
                            <p className="text-xs text-gray-400">ID: {user.roll}</p>
                        </div>
                    </div>
                    <Button
                        variant="ghost"
                        className="!p-2 text-red-400 hover:bg-red-500/20 hover:text-red-300 rounded-lg"
                        onClick={() => handleDelete(user.identifier)}
                        icon={<TrashIcon />}
                    >
                        <span className="sr-only">Delete</span>
                    </Button>
                </div>
            )) : (
                <div className="text-center py-12">
                    <div className="w-16 h-16 bg-gray-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg className="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M15 21a6 6 0 00-9-5.197m0 0A5.975 5.975 0 0112 13a5.975 5.975 0 013 5.197m-3 0V21" />
                        </svg>
                    </div>
                    <p className="text-gray-400 font-medium">No users registered</p>
                    <p className="text-gray-500 text-sm mt-1">Enroll users to get started</p>
                </div>
            )}
        </div>
    );
};
