
import React, { ReactNode } from 'react';

interface CardProps {
    children: ReactNode;
    className?: string;
}

export const Card: React.FC<CardProps> = ({ children, className }) => {
    return (
        <div className={`glass rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 fade-in ${className}`}>
            {children}
        </div>
    );
};

interface CardHeaderProps {
    title: string;
    icon?: ReactNode;
    children?: ReactNode;
}

export const CardHeader: React.FC<CardHeaderProps> = ({ title, icon, children }) => {
    return (
        <div className="p-5 sm:p-6 border-b border-white/10 flex justify-between items-center bg-gradient-to-r from-white/5 to-transparent">
            <div className="flex items-center gap-3">
                {icon && <div className="text-indigo-400">{icon}</div>}
                <h3 className="text-lg font-semibold text-white">{title}</h3>
            </div>
            <div>{children}</div>
        </div>
    );
};

interface CardContentProps {
    children: ReactNode;
    className?: string;
}

export const CardContent: React.FC<CardContentProps> = ({ children, className }) => {
    return <div className={`p-5 sm:p-6 ${className}`}>{children}</div>;
};
