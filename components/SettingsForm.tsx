
import React from 'react';
import { Settings } from '../types';
import { Input } from './common/Input';
import { Button } from './common/Button';

interface SettingsFormProps {
    settings: Settings;
}

const SettingsInput: React.FC<{ label: string; name: keyof Settings; value: number; type?: string, step?: string, helpText?: string }> = ({ label, name, value, type = "number", step, helpText }) => (
    <div className="space-y-2">
        <label htmlFor={name} className="block text-sm font-semibold text-gray-300">{label}</label>
        <Input 
            id={name}
            name={name}
            type={type}
            defaultValue={value}
            step={step}
        />
        {helpText && <p className="text-xs text-gray-500 flex items-center gap-1">
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
            </svg>
            {helpText}
        </p>}
    </div>
);


export const SettingsForm: React.FC<SettingsFormProps> = ({ settings }) => {
    return (
        <form action="/settings" method="POST" className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <SettingsInput 
                    label="Auto Attendance Interval" 
                    name="auto_attendance_interval_minutes" 
                    value={settings.auto_attendance_interval_minutes}
                    helpText="Time in minutes between automatic checks."
                />
                <SettingsInput 
                    label="Images to Capture" 
                    name="nimgs" 
                    value={settings.nimgs}
                    helpText="Number of images per user enrollment."
                />
                <SettingsInput 
                    label="Recognition Timeout" 
                    name="pipeline_timeout" 
                    value={settings.pipeline_timeout}
                    helpText="Duration in seconds for a manual session."
                />
                <SettingsInput 
                    label="Face Stability Time" 
                    name="stable_time" 
                    value={settings.stable_time}
                    type="number"
                    step="0.1"
                    helpText="Seconds the face must be still to capture."
                />
                <SettingsInput 
                    label="Max Face Movement" 
                    name="max_center_movement" 
                    value={settings.max_center_movement}
                    step="0.5"
                    helpText="Max movement in pixels to be 'stable'."
                />
                <SettingsInput 
                    label="Match Distance Threshold" 
                    name="match_distance_threshold" 
                    value={settings.match_distance_threshold}
                    step="100"
                    helpText="Lower is stricter. Controls recognition accuracy."
                />
            </div>
            <div className="pt-4 flex justify-end border-t border-white/10 mt-6">
                <Button type="submit" className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700">
                    Save Settings
                </Button>
            </div>
        </form>
    );
};
