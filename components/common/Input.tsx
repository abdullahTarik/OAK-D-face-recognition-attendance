
import React, { InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    label?: string;
}

export const Input: React.FC<InputProps> = ({ label, id, ...props }) => {
    return (
        <div>
            {label && <label htmlFor={id} className="block text-sm font-medium text-gray-300 mb-2">{label}</label>}
            <input
                id={id}
                className="w-full glass border border-white/10 text-white rounded-xl shadow-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 sm:text-sm px-4 py-3 transition-all duration-200 hover:border-white/20"
                {...props}
            />
        </div>
    );
};
