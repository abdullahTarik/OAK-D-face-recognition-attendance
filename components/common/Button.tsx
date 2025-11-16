
import React, { ReactNode } from 'react';

interface ButtonProps {
    children: ReactNode;
    onClick?: () => void;
    type?: 'button' | 'submit' | 'reset';
    variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
    className?: string;
    disabled?: boolean;
    href?: string;
    icon?: ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
    children,
    onClick,
    type = 'button',
    variant = 'primary',
    className = '',
    disabled = false,
    href,
    icon,
}) => {
    const baseClasses = 'inline-flex items-center justify-center gap-2 px-5 py-3 font-semibold text-sm rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95';
    const variantClasses = {
        primary: 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 focus:ring-blue-500',
        secondary: 'glass border border-white/10 text-gray-200 hover:bg-white/10 focus:ring-gray-500',
        danger: 'bg-gradient-to-r from-red-600 to-rose-600 text-white hover:from-red-700 hover:to-rose-700 focus:ring-red-500',
        ghost: 'bg-transparent text-gray-300 hover:bg-white/5 hover:text-white',
    };
    const disabledClasses = 'opacity-50 cursor-not-allowed transform-none hover:scale-100';

    const classes = `${baseClasses} ${variantClasses[variant]} ${disabled ? disabledClasses : ''} ${className}`;

    if (href) {
        return (
            <a href={href} className={classes}>
                {icon}
                {children}
            </a>
        );
    }

    return (
        <button type={type} onClick={onClick} className={classes} disabled={disabled}>
            {icon}
            {children}
        </button>
    );
};
