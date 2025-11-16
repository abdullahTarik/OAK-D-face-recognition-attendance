
import React, { useState } from 'react';
import { Card, CardHeader, CardContent } from './common/Card';
import { UsersIcon, SettingsIcon } from './Icons';
import { UserList } from './UserList';
import { SettingsForm } from './SettingsForm';
import { User, Settings } from '../types';

interface UserManagementProps {
    users: User[];
    settings: Settings | undefined;
}

type Tab = 'users' | 'settings';

export const UserManagement: React.FC<UserManagementProps> = ({ users, settings }) => {
    const [activeTab, setActiveTab] = useState<Tab>('users');

    const renderContent = () => {
        if (!settings) {
            return <p className="text-center text-gray-500 p-8">Settings not available.</p>;
        }
        switch (activeTab) {
            case 'users':
                return <UserList users={users} />;
            case 'settings':
                return <SettingsForm settings={settings} />;
            default:
                return null;
        }
    };

    const TabButton: React.FC<{ tabName: Tab; label: string; icon: React.ReactNode }> = ({ tabName, label, icon }) => (
        <button
            onClick={() => setActiveTab(tabName)}
            className={`flex items-center gap-2 px-5 py-3 text-sm font-semibold rounded-t-xl border-b-2 transition-all duration-200 ${
                activeTab === tabName
                    ? 'border-indigo-500 text-indigo-400 bg-gradient-to-b from-indigo-500/10 to-transparent'
                    : 'border-transparent text-gray-400 hover:text-white hover:border-gray-500/50 hover:bg-white/5'
            }`}
        >
            <div className={activeTab === tabName ? 'text-indigo-400' : 'text-gray-500'}>{icon}</div>
            {label}
        </button>
    );

    return (
        <Card>
            <div className="border-b border-white/10 bg-gradient-to-r from-white/5 to-transparent">
                <div className="flex -mb-px px-2">
                    <TabButton tabName="users" label="Registered Users" icon={<UsersIcon />} />
                    <TabButton tabName="settings" label="System Settings" icon={<SettingsIcon />} />
                </div>
            </div>
            <CardContent>
                {renderContent()}
            </CardContent>
        </Card>
    );
};
